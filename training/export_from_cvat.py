"""
export_from_cvat.py — Nối mắt xích CVAT → DVC → MinIO.

Tự động:
  1. Gọi CVAT REST API export task/project sang format YOLO (hoặc COCO).
  2. Giải nén vào thư mục dataset.
  3. dvc add + git commit + dvc push → version mới trên MinIO.

Chạy:
  python export_from_cvat.py --project-id 1 --format "YOLO 1.1" --tag v2
"""
import argparse
import io
import os
import subprocess
import time
import zipfile

import requests

CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.getenv("CVAT_USER", "admin")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD", "admin_pw")
DATA_DIR = os.getenv("DATA_DIR", "data")


def auth() -> str:
    """Đăng nhập CVAT, trả về token."""
    r = requests.post(
        f"{CVAT_URL}/api/auth/login",
        json={"username": CVAT_USER, "password": CVAT_PASSWORD},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["key"]


def export_project(token: str, project_id: int, fmt: str) -> bytes:
    """Yêu cầu export và poll cho tới khi sẵn sàng, trả về nội dung zip."""
    headers = {"Authorization": f"Token {token}"}
    params = {"format": fmt, "save_images": "true"}
    url = f"{CVAT_URL}/api/projects/{project_id}/dataset"

    # Khởi tạo export (CVAT trả 202 cho tới khi xong)
    while True:
        r = requests.get(url, params={**params, "action": "download"},
                         headers=headers, timeout=60)
        if r.status_code == 202:  # đang xử lý
            print("   ... đang chuẩn bị export")
            time.sleep(3)
            continue
        r.raise_for_status()
        return r.content


def unpack(content: bytes, dest: str):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        z.extractall(dest)
    print(f">> Giải nén vào {dest}")


def version_with_dvc(tag: str):
    """dvc add + commit + push."""
    subprocess.run(["dvc", "add", DATA_DIR], check=True)
    subprocess.run(["git", "add", f"{DATA_DIR}.dvc", ".gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", f"dataset {tag}"], check=True)
    subprocess.run(["git", "tag", tag], check=False)
    subprocess.run(["dvc", "push"], check=True)
    print(f">> Đã version dataset: {tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--format", default="YOLO 1.1",
                   help='CVAT export format, vd "YOLO 1.1" hoặc "COCO 1.0"')
    p.add_argument("--tag", default="v1", help="Nhãn version cho dataset")
    args = p.parse_args()

    print(">> Đăng nhập CVAT ...")
    tok = auth()
    print(f">> Export project {args.project_id} ({args.format}) ...")
    data = export_project(tok, args.project_id, args.format)
    unpack(data, DATA_DIR)
    version_with_dvc(args.tag)
    print(">> Hoàn tất. Giờ có thể chạy: python train.py --data data.yaml")
