"""
app.py — minio-api: REST tối thiểu bọc MinIO (S3) + UI duyệt dữ liệu.

Endpoint:
  GET    /buckets                       — danh sách bucket
  POST   /buckets/{bucket}              — tạo bucket
  DELETE /buckets/{bucket}              — xóa bucket (rỗng)
  GET    /buckets/{bucket}/objects?prefix=  — liệt kê object (theo prefix)
  GET    /buckets/{bucket}/download?key=    — link tải tạm (presigned)
  DELETE /buckets/{bucket}/objects?key=     — xóa object
  POST   /buckets/{bucket}/upload           — upload file (multipart)
  GET    /health

UI tại /  (static/index.html).
"""
import os

import boto3
from botocore.client import Config
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "minioadmin")

app = FastAPI(title="MinIO API", description="REST tối thiểu bọc S3")

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")


def s3():
    return boto3.client(
        "s3", endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_USER, aws_secret_access_key=MINIO_PASSWORD,
        config=Config(signature_version="s3v4"),
    )


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/buckets")
def list_buckets():
    try:
        r = s3().list_buckets()
        return {"buckets": [b["Name"] for b in r.get("Buckets", [])]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"MinIO lỗi: {exc}")


@app.post("/buckets/{bucket}")
def create_bucket(bucket: str):
    try:
        s3().create_bucket(Bucket=bucket)
        return {"bucket": bucket, "created": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Không tạo được bucket: {exc}")


@app.delete("/buckets/{bucket}")
def delete_bucket(bucket: str):
    try:
        s3().delete_bucket(Bucket=bucket)
        return {"bucket": bucket, "deleted": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Không xóa được (bucket phải rỗng): {exc}")


@app.get("/buckets/{bucket}/objects")
def list_objects(bucket: str, prefix: str = ""):
    try:
        r = s3().list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        folders = [p["Prefix"] for p in r.get("CommonPrefixes", [])]
        files = [{
            "key": o["Key"],
            "size": o["Size"],
            "size_h": human_size(o["Size"]),
        } for o in r.get("Contents", []) if o["Key"] != prefix]
        return {"bucket": bucket, "prefix": prefix,
                "folders": folders, "files": files}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"MinIO lỗi: {exc}")


@app.get("/buckets/{bucket}/download")
def download_url(bucket: str, key: str):
    try:
        url = s3().generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=600)
        return {"url": url}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"MinIO lỗi: {exc}")


@app.delete("/buckets/{bucket}/objects")
def delete_object(bucket: str, key: str):
    try:
        s3().delete_object(Bucket=bucket, Key=key)
        return {"deleted": key}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Không xóa được: {exc}")


@app.post("/buckets/{bucket}/upload")
async def upload(bucket: str, prefix: str = "", file: UploadFile = File(...)):
    try:
        key = f"{prefix}{file.filename}"
        s3().upload_fileobj(file.file, bucket, key)
        return {"uploaded": key}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Upload thất bại: {exc}")
