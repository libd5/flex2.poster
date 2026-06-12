"""Build ai-toolkit bilingual caption JSON from captions/ and captions_zh/."""

import argparse
import glob
import json
import os


def clean_caption(text: str) -> str:
    text = text.strip().replace("\n", ", ").replace("\r", ", ")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return ", ".join(parts)


def build_bilingual_json(
    data_root: str,
    output_path: str,
    require_canny: bool = True,
) -> dict:
    images_dir = os.path.join(data_root, "images")
    captions_en_dir = os.path.join(data_root, "captions")
    captions_zh_dir = os.path.join(data_root, "captions_zh")
    canny_dir = os.path.join(data_root, "canny")

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(
        f
        for f in glob.glob(os.path.join(images_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_exts
    )

    canny_stems = None
    if require_canny:
        canny_stems = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(canny_dir, "*"))
            if os.path.isfile(f)
        }

    data = {}
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        en_path = os.path.join(captions_en_dir, stem + ".txt")
        zh_path = os.path.join(captions_zh_dir, stem + ".txt")

        if not os.path.exists(en_path) or not os.path.exists(zh_path):
            continue
        if require_canny and stem not in canny_stems:
            continue

        with open(en_path, "r", encoding="utf-8") as f:
            en = clean_caption(f.read())
        with open(zh_path, "r", encoding="utf-8") as f:
            zh = clean_caption(f.read())

        if not en or not zh:
            continue

        data[os.path.abspath(img_path)] = {
            "caption": en,
            "caption_short": zh,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/captions_bilingual.json",
    )
    parser.add_argument(
        "--no_require_canny",
        action="store_true",
        help="Include pairs even if canny map is missing",
    )
    args = parser.parse_args()

    data = build_bilingual_json(
        data_root=args.data_root,
        output_path=args.output_path,
        require_canny=not args.no_require_canny,
    )
    print(f"Wrote {len(data)} entries to {args.output_path}")
