"""
app.py — dataset-api: quản lý dataset đúng nghĩa (không chỉ file).

Cấu trúc MinIO mong đợi (do export_from_cvat.py push):
  bucket: datasets
    <dataset_name>/<version>/data.yaml
    <dataset_name>/<version>/images/train/*.jpg
    <dataset_name>/<version>/images/val/*.jpg
    <dataset_name>/<version>/labels/train/*.txt
    <dataset_name>/<version>/labels/val/*.txt

API gộp các nguồn:
  - MinIO: liệt kê dataset/version, đếm ảnh, đọc data.yaml (class)
  - MLflow: tra cứu run đã dùng dataset+version nào, mAP bao nhiêu

Endpoint:
  GET /datasets                          — danh sách dataset
  GET /datasets/{name}                   — chi tiết dataset (các version)
  GET /datasets/{name}/{version}         — chi tiết một version đầy đủ
  GET /datasets/{name}/{version}/preview — danh sách ảnh mẫu (presigned URL)
"""
import io
import os
from collections import defaultdict
from datetime import datetime

import boto3
import mlflow
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
BUCKET = os.getenv("DATASET_BUCKET", "datasets")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

app = FastAPI(title="Dataset API")

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")


def _s3():
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET)


def _list_versions() -> dict[str, list[dict]]:
    """Quét bucket datasets, gom thành {dataset_name: [{version, last_modified}]}."""
    s3 = _s3()
    out: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=BUCKET):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                # Mong đợi: <dataset>/<version>/...
                if len(parts) < 3:
                    continue
                name, version = parts[0], parts[1]
                pair = (name, version)
                if pair in seen:
                    continue
                seen.add(pair)
                out[name].append({
                    "version": version,
                    "last_modified": obj["LastModified"].isoformat(),
                })
    except Exception:  # noqa: BLE001
        return {}
    # Sắp version mới nhất trước
    for v in out.values():
        v.sort(key=lambda x: x["last_modified"], reverse=True)
    return dict(out)


def _read_yaml(name: str, version: str) -> dict:
    """Đọc data.yaml của một version để lấy class."""
    s3 = _s3()
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{name}/{version}/data.yaml")
        return yaml.safe_load(obj["Body"].read())
    except Exception:  # noqa: BLE001
        return {}


def _count_images(name: str, version: str) -> dict[str, int]:
    """Đếm ảnh train/val cho một version."""
    s3 = _s3()
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        prefix = f"{name}/{version}/images/{split}/"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
                counts[split] += len(page.get("Contents", []))
        except Exception:  # noqa: BLE001
            pass
    return counts


def _runs_for(name: str, version: str) -> list[dict]:
    """Tìm các MLflow run đã dùng dataset+version này."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        exps = client.search_experiments()
        runs = []
        for exp in exps:
            # Tìm run có tag dataset = name và dataset_version = version
            found = client.search_runs(
                [exp.experiment_id],
                filter_string=(f"tags.dataset = '{name}' "
                               f"and tags.dataset_version = '{version}'"),
                max_results=10,
            )
            for r in found:
                runs.append({
                    "run_id": r.info.run_id,
                    "name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                    "map50": r.data.metrics.get("mAP50", 0),
                    "started": datetime.fromtimestamp(
                        r.info.start_time / 1000).strftime("%d/%m"),
                    "epochs": int(r.data.params.get("epochs", 0)),
                })
        return runs
    except Exception:  # noqa: BLE001
        return []


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets():
    """Danh sách dataset kèm version mới nhất."""
    all_ds = _list_versions()
    out = []
    for name, versions in all_ds.items():
        latest = versions[0] if versions else None
        info = {"name": name, "version_count": len(versions),
                "latest_version": latest["version"] if latest else None,
                "latest_at": latest["last_modified"] if latest else None}
        if latest:
            yml = _read_yaml(name, latest["version"])
            counts = _count_images(name, latest["version"])
            info["image_count"] = counts["train"] + counts["val"]
            info["class_count"] = len(yml.get("names", []) or yml.get("names", {}))
        out.append(info)
    return {"datasets": out}


@app.get("/datasets/{name}")
def get_dataset(name: str):
    """Chi tiết một dataset: liệt kê các version."""
    all_ds = _list_versions()
    if name not in all_ds:
        raise HTTPException(404, "Không tìm thấy dataset")
    return {"name": name, "versions": all_ds[name]}


@app.get("/datasets/{name}/{version}")
def get_version(name: str, version: str):
    """Chi tiết đầy đủ một version: số ảnh, class, lịch sử train."""
    yml = _read_yaml(name, version)
    counts = _count_images(name, version)
    raw_names = yml.get("names", [])
    classes = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
    return {
        "name": name, "version": version,
        "train": counts["train"], "val": counts["val"],
        "total": counts["train"] + counts["val"],
        "classes": classes, "class_count": len(classes),
        "runs": _runs_for(name, version),
    }


@app.get("/datasets/{name}/{version}/preview")
def preview(name: str, version: str, n: int = 6):
    """N ảnh mẫu kèm nhãn (presigned URL + danh sách bbox + tên class)."""
    s3 = _s3()
    yml = _read_yaml(name, version)
    raw_names = yml.get("names", [])
    classes = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)

    prefix = f"{name}/{version}/images/train/"
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=n)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Không liệt kê được ảnh: {exc}") from exc

    items = []
    for obj in resp.get("Contents", [])[:n]:
        key = obj["Key"]
        fname = key.split("/")[-1]
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET, "Key": key},
                ExpiresIn=600)
        except Exception:  # noqa: BLE001
            continue

        # Đọc file nhãn cùng tên (đổi đuôi .jpg → .txt, đổi thư mục images/ → labels/)
        label_key = key.replace("/images/", "/labels/", 1).rsplit(".", 1)[0] + ".txt"
        boxes = []
        try:
            obj_lbl = s3.get_object(Bucket=BUCKET, Key=label_key)
            raw = obj_lbl["Body"].read().decode("utf-8", errors="ignore")
            for ln in raw.strip().splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    boxes.append({
                        "cls": int(parts[0]),
                        "cx": float(parts[1]), "cy": float(parts[2]),
                        "w": float(parts[3]), "h": float(parts[4]),
                    })
                except ValueError:
                    continue
        except Exception:  # noqa: BLE001
            pass

        items.append({"name": fname, "url": url, "boxes": boxes})

    return {"images": items, "classes": classes}
