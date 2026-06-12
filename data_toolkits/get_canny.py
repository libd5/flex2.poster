import cv2
import numpy as np
from controlnet_aux import CannyDetector
import argparse
import os 
import glob 
from PIL import Image
def generate_canny(image_path, output_path, low_threshold=100, high_threshold=200):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取: {image_path}")
        return False
    canny_detector = CannyDetector()
    edge_map = canny_detector(
        image, 
        low_threshold=low_threshold, 
        high_threshold=high_threshold
    )
    Image.fromarray(edge_map).save(output_path)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/hd_course_covers")
    parser.add_argument("--output_dir", type=str, default="./data/canny_maps")
    parser.add_argument("--low_threshold", type=int, default=100)
    parser.add_argument("--high_threshold", type=int, default=200)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    for img_path in images:
        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        generate_canny(img_path, out_path, args.low_threshold, args.high_threshold)