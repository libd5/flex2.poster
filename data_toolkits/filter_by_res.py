import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def check_resolution(task):
    image_path, min_resolution = task
    try:
        with Image.open(image_path) as img:
            passed = img.width >= min_resolution and img.height >= min_resolution
    except Exception as e:
        return image_path, False, str(e)
    return image_path, passed, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--min_resolution", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images = sorted(data_root.glob("*.jpg"))
    if not images:
        raise ValueError(f"No .jpg files found in {data_root}")

    num_workers = args.num_workers or os.cpu_count() or 1
    tasks = [(str(image), args.min_resolution) for image in images]

    rejected_images = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(check_resolution, tasks, chunksize=64)
        for image_path, passed, error in tqdm(results, total=len(tasks), desc="Filtering by resolution"):
            if error:
                print(f"Error reading {image_path}: {error}")
            if not passed:
                rejected_images.append(image_path)

    for image in tqdm(rejected_images, desc="Removing rejected"):
        os.remove(image)

    print(
        f"Done. total={len(images)}, kept={len(images) - len(rejected_images)}, "
        f"removed={len(rejected_images)}"
    )


if __name__ == "__main__":
    main()
