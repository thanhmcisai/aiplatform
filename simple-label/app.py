"""
app.py — simple-label: gán nhãn đơn giản, ghi thẳng định dạng YOLO vào MinIO.

Thiết kế: BỎ QUA hoàn toàn khái niệm Project/Task/Job của CVAT.
Đơn vị duy nhất là "dự án" (project) = một dataset đang gán nhãn:
  - tên (tự sinh slug)
  - danh sách class
  - các ảnh (lưu MinIO ngay khi upload)
  - các file nhãn (lưu MinIO khi người dùng bấm "Lưu")

Cấu trúc lưu trên MinIO (KHỚP với dataset-api hiện có để khỏi sửa hai bên):
  bucket: datasets
    <slug>/v1/data.yaml
    <slug>/v1/images/train/<file>
    <slug>/v1/labels/train/<file>.txt   (nếu đã gán)
    <slug>/.label-meta.json              (state nội bộ: class, progress)

Endpoint:
  GET    /projects                       — danh sách dự án
  POST   /projects                       — tạo {name, classes:[..]}
  GET    /projects/{slug}                — chi tiết: classes, images[+labeled?]
  POST   /projects/{slug}/images         — upload ảnh (multipart)
  POST   /projects/{slug}/labels/{img}   — lưu nhãn 1 ảnh {boxes:[{cls,cx,cy,w,h}]}
  GET    /projects/{slug}/image/{img}    — presigned URL của một ảnh
  GET    /projects/{slug}/labels/{img}   — đọc nhãn của một ảnh (để load lại)
  POST   /projects/{slug}/classes        — sửa danh sách class
"""
import io
import json
import os
import re
import unicodedata
from typing import List, Optional

import boto3
import yaml
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
BUCKET = os.getenv("DATASET_BUCKET", "datasets")
META = ".label-meta.json"

app = FastAPI(title="Simple Label")
_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")


def _s3():
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET)


def _slug(name: str) -> str:
    """Bỏ dấu, lowercase, gạch nối — dùng làm tên thư mục MinIO."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_str).strip("-").lower()
    return s or "du-an"


def _meta_key(slug: str) -> str:
    return f"{slug}/{META}"


def _read_meta(slug: str) -> Optional[dict]:
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=_meta_key(slug))
        return json.loads(obj["Body"].read())
    except ClientError:
        return None


def _write_meta(slug: str, meta: dict):
    _s3().put_object(
        Bucket=BUCKET, Key=_meta_key(slug),
        Body=json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json")


def _write_yaml(slug: str, classes: List[str]):
    """data.yaml để train.py + dataset-api đọc."""
    data = {
        "path": f"/data/{slug}/v1",
        "train": "images/train", "val": "images/train",  # mặc định gộp
        "names": {i: c for i, c in enumerate(classes)},
    }
    _s3().put_object(
        Bucket=BUCKET, Key=f"{slug}/v1/data.yaml",
        Body=yaml.safe_dump(data, allow_unicode=True).encode("utf-8"),
        ContentType="text/yaml")


class CreateProject(BaseModel):
    name: str
    classes: List[str]
    task: str = "detect"  # "detect" (bounding box) hoặc "classify"


class Boxes(BaseModel):
    boxes: List[dict]


class ClassifyLabel(BaseModel):
    cls: int  # index trong meta["classes"]


class SetClasses(BaseModel):
    classes: List[str]


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/projects")
def list_projects():
    """Liệt kê các dự án gán nhãn (có file .label-meta.json)."""
    s3 = _s3()
    out = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/" + META):
                    slug = obj["Key"].split("/")[0]
                    meta = _read_meta(slug)
                    if meta:
                        out.append({
                            "slug": slug, "name": meta.get("name", slug),
                            "classes": meta.get("classes", []),
                            "image_count": meta.get("image_count", 0),
                            "labeled_count": meta.get("labeled_count", 0),
                        })
    except ClientError:
        pass
    return {"projects": out}


@app.post("/projects")
def create_project(body: CreateProject):
    if not body.classes:
        raise HTTPException(400, "Cần ít nhất một loại (class)")
    if body.task not in ("detect", "classify"):
        raise HTTPException(400, "task phải là 'detect' hoặc 'classify'")
    slug = _slug(body.name)
    if _read_meta(slug):
        raise HTTPException(409, "Tên dự án đã tồn tại")
    meta = {"name": body.name, "slug": slug, "task": body.task,
            "classes": body.classes,
            "image_count": 0, "labeled_count": 0,
            "images": [], "labels": {}}
    _write_meta(slug, meta)
    if body.task == "detect":
        _write_yaml(slug, body.classes)
    return {"slug": slug, "name": body.name,
            "classes": body.classes, "task": body.task}


@app.get("/projects/{slug}")
def get_project(slug: str):
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    return meta


@app.post("/projects/{slug}/classes")
def set_classes(slug: str, body: SetClasses):
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    meta["classes"] = body.classes
    _write_meta(slug, meta)
    _write_yaml(slug, body.classes)
    return {"classes": body.classes}


@app.post("/projects/{slug}/images")
async def upload_images(slug: str, files: List[UploadFile] = File(...)):
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    task = meta.get("task", "detect")
    s3 = _s3()
    added = []
    for f in files:
        data = await f.read()
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", f.filename or "img.jpg")
        if task == "detect":
            key = f"{slug}/v1/images/train/{safe}"
        else:
            # classify: ảnh chưa gán nằm trong _unlabeled, khi gán sẽ chuyển
            key = f"{slug}/v1/_unlabeled/{safe}"
        s3.put_object(Bucket=BUCKET, Key=key, Body=data,
                      ContentType=f.content_type or "image/jpeg")
        if safe not in meta["images"]:
            meta["images"].append(safe)
            added.append(safe)
    meta["image_count"] = len(meta["images"])
    _write_meta(slug, meta)
    return {"added": added, "total": meta["image_count"]}


def _image_key(slug: str, img: str, meta: dict) -> str:
    """Vị trí S3 của một ảnh (detect: images/train; classify: theo class hoặc _unlabeled)."""
    if meta.get("task", "detect") == "detect":
        return f"{slug}/v1/images/train/{img}"
    cls_idx = meta.get("labels", {}).get(img)
    if cls_idx is not None and 0 <= cls_idx < len(meta["classes"]):
        return f"{slug}/v1/train/{_slug(meta['classes'][cls_idx])}/{img}"
    return f"{slug}/v1/_unlabeled/{img}"


@app.get("/projects/{slug}/image/{img}")
def get_image(slug: str, img: str):
    """Trả thẳng bytes ảnh (proxy qua MinIO).

    Trước đây trả presigned URL trỏ tới minio:9000 — trình duyệt (nhất là máy
    khác trong mạng, qua gateway) không phân giải được hostname nội bộ này nên
    ảnh không hiện. Stream thẳng qua service giúp mọi request đi qua /label/.
    """
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=_image_key(slug, img, meta))
        data = obj["Body"].read()
        ctype = obj.get("ContentType") or "image/jpeg"
    except ClientError as exc:
        raise HTTPException(404, "Không tìm thấy ảnh") from exc
    return Response(content=data, media_type=ctype,
                    headers={"Cache-Control": "no-store"})


@app.delete("/projects/{slug}/image/{img}")
def delete_image(slug: str, img: str):
    """Xoá một ảnh khỏi dataset: file ảnh + file nhãn (nếu có) + cập nhật meta."""
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    s3 = _s3()
    for key in (_image_key(slug, img, meta), _label_key(slug, img)):
        try:
            s3.delete_object(Bucket=BUCKET, Key=key)
        except ClientError:
            pass
    if img in meta.get("images", []):
        meta["images"].remove(img)
    meta.get("labels", {}).pop(img, None)
    if img in meta.get("labeled", []):
        meta["labeled"].remove(img)
    meta["image_count"] = len(meta.get("images", []))
    meta["labeled_count"] = (len(meta["labeled"]) if "labeled" in meta
                             else len(meta.get("labels", {})))
    _write_meta(slug, meta)
    return {"deleted": img, "total": meta["image_count"]}


@app.post("/projects/{slug}/classify/{img}")
def classify(slug: str, img: str, body: ClassifyLabel):
    """Gán class cho cả ảnh (classification). Di chuyển file đến thư mục class."""
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    if meta.get("task") != "classify":
        raise HTTPException(400, "Dự án này dùng cho detection, không phải classify")
    if not (0 <= body.cls < len(meta["classes"])):
        raise HTTPException(400, "Class index không hợp lệ")

    s3 = _s3()
    # Tìm vị trí hiện tại của ảnh
    current = meta.get("labels", {}).get(img)
    if current is not None and 0 <= current < len(meta["classes"]):
        src_key = f"{slug}/v1/train/{_slug(meta['classes'][current])}/{img}"
    else:
        src_key = f"{slug}/v1/_unlabeled/{img}"

    new_cls_name = _slug(meta["classes"][body.cls])
    dst_key = f"{slug}/v1/train/{new_cls_name}/{img}"

    if src_key == dst_key:
        return {"ok": True, "unchanged": True}

    # MinIO không có "rename" — phải copy rồi delete
    try:
        s3.copy_object(Bucket=BUCKET, Key=dst_key,
                       CopySource={"Bucket": BUCKET, "Key": src_key})
        s3.delete_object(Bucket=BUCKET, Key=src_key)
    except ClientError as exc:
        raise HTTPException(500, f"Lỗi di chuyển: {exc}") from exc

    labels = meta.get("labels", {})
    is_new = img not in labels
    labels[img] = body.cls
    meta["labels"] = labels
    if is_new:
        meta["labeled_count"] = meta.get("labeled_count", 0) + 1
    _write_meta(slug, meta)
    return {"ok": True, "class": meta["classes"][body.cls],
            "labeled_count": meta["labeled_count"]}


def _label_key(slug: str, img: str) -> str:
    base = img.rsplit(".", 1)[0]
    return f"{slug}/v1/labels/train/{base}.txt"


@app.post("/projects/{slug}/labels/{img}")
def save_labels(slug: str, img: str, body: Boxes):
    meta = _read_meta(slug)
    if not meta:
        raise HTTPException(404, "Không tìm thấy dự án")
    lines = []
    for b in body.boxes:
        try:
            lines.append(f"{int(b['cls'])} {float(b['cx']):.6f} "
                         f"{float(b['cy']):.6f} {float(b['w']):.6f} "
                         f"{float(b['h']):.6f}")
        except (KeyError, ValueError):
            continue
    content = "\n".join(lines) + ("\n" if lines else "")
    _s3().put_object(
        Bucket=BUCKET, Key=_label_key(slug, img),
        Body=content.encode("utf-8"), ContentType="text/plain")
    # Cập nhật progress
    labeled = meta.get("labeled", [])
    if img not in labeled:
        labeled.append(img)
        meta["labeled"] = labeled
        meta["labeled_count"] = len(labeled)
        _write_meta(slug, meta)
    return {"ok": True, "boxes": len(lines),
            "labeled_count": meta["labeled_count"]}


@app.get("/projects/{slug}/labels/{img}")
def load_labels(slug: str, img: str):
    """Đọc lại nhãn của một ảnh (để load khi quay lại)."""
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=_label_key(slug, img))
        raw = obj["Body"].read().decode("utf-8", errors="ignore")
    except ClientError:
        return {"boxes": []}
    boxes = []
    for ln in raw.strip().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append({"cls": int(parts[0]), "cx": float(parts[1]),
                          "cy": float(parts[2]), "w": float(parts[3]),
                          "h": float(parts[4])})
        except ValueError:
            continue
    return {"boxes": boxes}
