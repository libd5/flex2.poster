"""
Crawl images from Bing, given keywords.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import time
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from urllib.parse import quote

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def image_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_existing_hashes(data_root: str) -> Set[str]:
    hashes = set()
    for path in glob.glob(os.path.join(data_root, "*")):
        name = os.path.basename(path)
        stem, _ = os.path.splitext(name)
        if len(stem) == 64:
            hashes.add(stem)
    return hashes


def extension_from_image(content: bytes, img_url: str) -> Optional[str]:
    for ext in VALID_EXT:
        if img_url.lower().split("?")[0].endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext

    try:
        with Image.open(io.BytesIO(content)) as img:
            fmt = (img.format or "").lower()
    except Exception:
        return None

    if fmt in ("jpeg", "jpg"):
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt == "webp":
        return ".webp"
    return None


def save_image(content: bytes, img_url: str, data_root: str, seen_hashes: Set[str]) -> Optional[str]:
    digest = image_hash(content)
    if digest in seen_hashes:
        return None

    ext = extension_from_image(content, img_url)
    if ext is None:
        return None

    filename = f"{digest}{ext}"
    filepath = os.path.join(data_root, filename)
    if os.path.exists(filepath):
        seen_hashes.add(digest)
        return None

    try:
        with Image.open(io.BytesIO(content)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            if ext == ".jpg":
                img.save(filepath, "JPEG", quality=95)
            else:
                img.save(filepath)
    except Exception:
        return None

    seen_hashes.add(digest)
    return filepath


def create_session(use_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_proxy
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    max_retries: int,
    timeout: float,
    retry_base_delay: float,
) -> Optional[requests.Response]:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}")
            return response
        except Exception as e:
            last_error = e
            if attempt + 1 < max_retries:
                sleep_s = retry_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
    print(f"Request failed after {max_retries} retries: {url}\n  {last_error}")
    return None


def parse_image_urls(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for img_tag in soup.find_all("a", class_="iusc"):
        meta = img_tag.get("m")
        if not meta:
            continue
        try:
            data = json.loads(meta)
        except json.JSONDecodeError:
            continue
        img_url = data.get("murl", "")
        if not img_url or "bing.com" in img_url:
            continue
        if not any(img_url.lower().split("?")[0].endswith(ext) for ext in VALID_EXT):
            continue
        urls.append(img_url)
    return urls


def crawl_images(
    keyword: str,
    data_root: str,
    num_samples: int,
    delay: float,
    max_retries: int,
    retry_base_delay: float,
    max_page_failures: int,
    use_proxy: bool,
) -> int:
    os.makedirs(data_root, exist_ok=True)
    session = create_session(use_proxy=use_proxy)

    seen_hashes = load_existing_hashes(data_root)
    downloaded = len(seen_hashes)
    page = 1
    page_failures = 0
    empty_pages = 0

    pbar = tqdm(initial=downloaded, total=num_samples, desc=f"Crawling '{keyword}'")
    if downloaded:
        print(f"Resuming with {downloaded} existing images in {data_root}")

    while downloaded < num_samples:
        url = (
            f"https://www.bing.com/images/search?q={quote(keyword)}"
            f"&form=HDRSC2&first={(page - 1) * 35 + 1}"
        )
        response = request_with_retry(
            session,
            url,
            max_retries=max_retries,
            timeout=20,
            retry_base_delay=retry_base_delay,
        )
        if response is None:
            page_failures += 1
            if page_failures >= max_page_failures:
                print(f"Stopping after {max_page_failures} consecutive page failures.")
                break
            page += 1
            time.sleep(delay * 2)
            continue

        page_failures = 0
        if response.status_code != 200:
            print(f"Search page {page} returned HTTP {response.status_code}, skipping.")
            page += 1
            time.sleep(delay)
            continue

        image_urls = parse_image_urls(response.text)
        if not image_urls:
            empty_pages += 1
            if empty_pages >= 5:
                print("No more images found.")
                break
            page += 1
            time.sleep(delay)
            continue

        empty_pages = 0
        for img_url in image_urls:
            if downloaded >= num_samples:
                break

            img_resp = request_with_retry(
                session,
                img_url,
                max_retries=max_retries,
                timeout=15,
                retry_base_delay=retry_base_delay,
            )
            if img_resp is None or img_resp.status_code != 200 or not img_resp.content:
                continue

            saved_path = save_image(img_resp.content, img_url, data_root, seen_hashes)
            if saved_path is None:
                continue

            downloaded += 1
            pbar.update(1)
            time.sleep(random.uniform(0.05, 0.2))

        page += 1
        time.sleep(delay + random.uniform(0, 0.5))

    pbar.close()
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Crawl images from Bing image search.")
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between search pages")
    parser.add_argument("--max_retries", type=int, default=5, help="Retries per HTTP request")
    parser.add_argument("--retry_base_delay", type=float, default=2.0, help="Base delay for exponential backoff")
    parser.add_argument(
        "--max_page_failures",
        type=int,
        default=10,
        help="Stop only after this many consecutive search-page failures",
    )
    parser.add_argument(
        "--use_proxy",
        action="store_true",
        help="Use system proxy env vars (disabled by default to avoid broken proxies)",
    )
    args = parser.parse_args()

    total = crawl_images(
        keyword=args.keyword,
        data_root=args.data_root,
        num_samples=args.num_samples,
        delay=args.delay,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        max_page_failures=args.max_page_failures,
        use_proxy=args.use_proxy,
    )
    print(f"Saved {total} images to {args.data_root}")


if __name__ == "__main__":
    main()
