"""
make_sample_dataset.py — Sinh dataset YOLO mẫu nhỏ để chạy thử train.

Tạo các ảnh tổng hợp đơn giản (hình tròn / hình vuông màu trên nền trắng)
kèm nhãn YOLO tương ứng. Đủ để xác nhận luồng train chạy thông —
KHÔNG dùng cho mô hình thật.

Tạo cấu trúc:
  data/
    images/train/*.jpg  (+ labels/train/*.txt)
    images/val/*.jpg     (+ labels/val/*.txt)
  data.yaml  (2 class: circle, square)

Dùng:
  pip install pillow
  python scripts/make_sample_dataset.py --n-train 40 --n-val 10
"""
import argparse
import os
import random

from PIL import Image, ImageDraw

CLASSES = ["circle", "square"]
IMG = 640


def draw_one(path_img: str, path_lbl: str):
    img = Image.new("RGB", (IMG, IMG), (245, 245, 247))
    d = ImageDraw.Draw(img)
    lines = []
    n_obj = random.randint(1, 3)
    for _ in range(n_obj):
        cls = random.randint(0, 1)
        w = random.randint(60, 160)
        h = w if cls == 1 else random.randint(60, 160)
        x = random.randint(0, IMG - w)
        y = random.randint(0, IMG - h)
        color = tuple(random.randint(40, 200) for _ in range(3))
        if cls == 0:
            d.ellipse([x, y, x + w, y + h], fill=color)
        else:
            d.rectangle([x, y, x + w, y + h], fill=color)
        # YOLO format: class cx cy w h (chuẩn hóa 0..1)
        cx = (x + w / 2) / IMG
        cy = (y + h / 2) / IMG
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w/IMG:.6f} {h/IMG:.6f}")
    img.save(path_img, "JPEG", quality=85)
    with open(path_lbl, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_split(root: str, split: str, n: int):
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        draw_one(os.path.join(img_dir, f"{split}_{i:04d}.jpg"),
                 os.path.join(lbl_dir, f"{split}_{i:04d}.txt"))
    print(f">> {split}: {n} ảnh + nhãn")


def write_yaml(root: str):
    path = os.path.join(os.path.dirname(root) or ".", "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(root)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    print(f">> data.yaml: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data", help="Thư mục dataset")
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-val", type=int, default=10)
    args = p.parse_args()

    random.seed(42)
    make_split(args.out, "train", args.n_train)
    make_split(args.out, "val", args.n_val)
    write_yaml(args.out)
    print(">> Xong. Thử train:")
    print("   python training/train.py --data data.yaml --model yolo26n.pt "
          "--epochs 3 --name smoke --skip-pull")
