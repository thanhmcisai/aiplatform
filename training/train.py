"""
train.py — Training pipeline cho Computer Vision (YOLO).

Luồng:
  1. Pull dataset đã version bằng DVC từ MinIO.
  2. Train model YOLO (Ultralytics) trên dataset.
  3. Log params/metrics/artifacts vào MLflow.
  4. Đăng ký model vào MLflow Model Registry.

Chạy:
  python train.py --data data.yaml --model yolo26n.pt --epochs 50 --name exp1
"""
import argparse
import os
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL = "cv-detector"


def pull_dataset():
    """Đồng bộ dataset từ DVC remote (MinIO)."""
    print(">> Pulling dataset via DVC ...")
    subprocess.run(["dvc", "pull", "-q"], check=True)


def split_val(data_yaml_path: str, val_ratio: float) -> None:
    """
    Nếu val rỗng và val_ratio > 0: chuyển ngẫu nhiên `val_ratio` ảnh
    từ images/train sang images/val (kèm file nhãn .txt tương ứng).
    Không phá val có sẵn — chỉ chia nếu val trống.
    """
    import random
    import shutil
    if val_ratio <= 0:
        return
    import yaml as _yaml
    yml_path = Path(data_yaml_path)
    if not yml_path.exists():
        return
    cfg = _yaml.safe_load(yml_path.read_text())
    root = Path(cfg.get("path") or yml_path.parent)
    train_imgs = root / "images" / "train"
    val_imgs = root / "images" / "val"
    if not train_imgs.exists():
        return
    if val_imgs.exists() and any(val_imgs.iterdir()):
        print(f">> Val đã có ảnh, bỏ qua val_split={val_ratio}")
        return
    val_imgs.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in train_imgs.iterdir() if p.is_file())
    n_val = max(1, int(len(imgs) * val_ratio))
    random.seed(42)
    chosen = random.sample(imgs, min(n_val, len(imgs)))
    for img in chosen:
        shutil.move(str(img), str(val_imgs / img.name))
        lbl = root / "labels" / "train" / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(root / "labels" / "val" / lbl.name))
    print(f">> Đã chia {len(chosen)}/{len(imgs)} ảnh sang val")


def resolve_model(model: str) -> str:
    """
    Trả về đường dẫn model để train.
    - Nếu là tên checkpoint Ultralytics (vd yolo26n.pt) hoặc file có sẵn: dùng luôn.
    - Nếu dạng 'base/xxx.pt' (mô hình nền trong MinIO bucket models): tải về.
    """
    if not model.startswith("base/"):
        return model  # yolo26n.pt hoặc đường dẫn local

    # Mô hình nền nạp sẵn: tải từ MinIO bucket 'models'
    import boto3
    local = Path("base_models") / Path(model).name
    if local.exists():
        return str(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )
    print(f">> Tải mô hình nền {model} từ MinIO ...")
    s3.download_file("models", model, str(local))
    return str(local)


def train(args):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("cv-detection")

    with mlflow.start_run(run_name=args.name) as run:
        # 1. Log hyperparameters
        mlflow.log_params({
            "base_model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "val_split": args.val_split,
            "data": args.data,
        })

        # Chia val nếu được yêu cầu
        if args.val_split > 0:
            split_val(args.data, args.val_split)

        # Tag dataset + version để dataset-api truy ngược "run nào dùng version nào"
        if args.dataset:
            mlflow.set_tag("dataset", args.dataset)
        if args.dataset_version:
            mlflow.set_tag("dataset_version", args.dataset_version)

        # 2. Train (resolve_model tải mô hình nền từ MinIO nếu cần)
        model = YOLO(resolve_model(args.model))
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            project="runs",
            name=args.name,
            exist_ok=True,
        )

        # 3. Log metrics
        m = results.results_dict
        mlflow.log_metrics({
            "mAP50": m.get("metrics/mAP50(B)", 0.0),
            "mAP50_95": m.get("metrics/mAP50-95(B)", 0.0),
            "precision": m.get("metrics/precision(B)", 0.0),
            "recall": m.get("metrics/recall(B)", 0.0),
        })

        # 4. Log weights + register
        # Lấy đúng thư mục lưu từ kết quả train (Ultralytics chèn 'detect/'
        # vào đường dẫn, nên không hard-code 'runs/<name>').
        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"Không thấy best.pt tại {best}")
        mlflow.log_artifact(str(best), artifact_path="weights")

        model_uri = f"runs:/{run.info.run_id}/weights"
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL)
        # Đóng mắt xích "chưa set stage": tự đưa lên Staging để chờ duyệt
        MlflowClient().transition_model_version_stage(
            name=REGISTERED_MODEL, version=mv.version, stage="Staging",
        )
        print(f">> Registered '{REGISTERED_MODEL}' v{mv.version} → Staging "
              f"(run {run.info.run_id})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml", help="YOLO dataset config")
    p.add_argument("--model", default="yolo26n.pt", help="Base model checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="exp1", help="Run / experiment name")
    p.add_argument("--task", default="detect", choices=["detect", "classify"],
                   help="Loại bài toán")
    p.add_argument("--patience", type=int, default=20,
                   help="Số epoch không cải thiện rồi dừng sớm")
    p.add_argument("--val-split", type=float, default=0.0, dest="val_split",
                   help="Tỉ lệ ảnh train chuyển sang val (0 = không chia, dùng val có sẵn)")
    p.add_argument("--dataset", default=None,
                   help="Tên dataset (để dataset-api truy ngược)")
    p.add_argument("--dataset-version", default=None, dest="dataset_version",
                   help="Phiên bản dataset (vd v1, v2...)")
    p.add_argument("--skip-pull", action="store_true", help="Bỏ qua dvc pull")
    args = p.parse_args()

    if not args.skip_pull:
        pull_dataset()
    train(args)
