"""
app.py — trainer-api: trigger train & export bất đồng bộ, theo dõi job.

Train là tác vụ lâu nên KHÔNG chạy đồng bộ trong request HTTP. Thay vào đó:
  - POST tạo job → trả job_id ngay, train chạy ở thread nền.
  - UI poll GET /jobs/{id} để xem tiến độ.

Endpoint:
  POST /train            — bắt đầu train (body: name, epochs, data, model)
  POST /export           — export CVAT + DVC version (body: project_id, tag, format)
  GET  /jobs             — danh sách job
  GET  /jobs/{job_id}    — trạng thái + log đuôi của một job
  GET  /health

UI tại /  (static/index.html).

LƯU Ý: train chạy ngay trong container này. Cần GPU + thư viện train
(ultralytics, mlflow, dvc) cài trong image. Không có GPU thì train CPU
rất chậm — chỉ hợp smoke-test.
"""
import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO = os.getenv("REPO_DIR", "/workspace")  # nơi chứa training/, data/
BASE_MODELS_FILE = os.getenv(
    "BASE_MODELS_FILE", os.path.join(REPO, "base-models", "base-models.json"))
DATA_DIR = os.path.join(REPO, "data")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "datasets")
DATASET_API = os.getenv("DATASET_API_URL", "http://dataset-api:8400")
app = FastAPI(title="Trainer API", description="Trigger train/export bất đồng bộ")

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")

# Lưu job trong bộ nhớ. Đủ cho 1 server; production nên dùng Redis/DB.
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


class TrainBody(BaseModel):
    name: str = "exp1"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 20
    val_split: float = 0.0
    data: str = "data.yaml"
    model: str = "yolo26n.pt"
    base_model: str | None = None  # id mô hình nền nạp sẵn (nếu fine-tune)
    dataset: str | None = None     # tên dataset (để truy ngược trong dataset-api)
    dataset_version: str | None = None
    task: str = "detect"           # "detect" hoặc "classify"
    skip_pull: bool = False


def _load_base_models() -> list[dict]:
    """Đọc danh sách mô hình nền nạp sẵn từ base-models.json."""
    try:
        with open(BASE_MODELS_FILE) as f:
            return json.load(f).get("base_models", [])
    except Exception:  # noqa: BLE001
        return []


@app.get("/base-models")
def base_models():
    """Liệt kê mô hình nền nạp sẵn để fine-tune."""
    return {"base_models": _load_base_models()}


def _s3():
    import boto3
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))


@app.get("/datasets")
def datasets():
    """Liệt kê dataset có sẵn để huấn luyện (mượn dataset-api)."""
    import requests
    try:
        r = requests.get(f"{DATASET_API}/datasets", timeout=8)
        return r.json()
    except Exception:  # noqa: BLE001
        return {"datasets": []}


def _prepare_dataset(name: str, version: str) -> tuple[str, str]:
    """Tải dataset <name>/<version> từ MinIO về local, ép path tuyệt đối trong
    data.yaml, và suy ra task. Trả (đường_dẫn_data.yaml, task)."""
    import yaml as _yaml
    s3 = _s3()
    prefix = f"{name}/{version}/"
    local_root = os.path.join(DATA_DIR, "_train", name, version)
    os.makedirs(local_root, exist_ok=True)
    found = False
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=DATASET_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel or key.endswith("/"):
                continue
            dst = os.path.join(local_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(DATASET_BUCKET, key, dst)
            found = True
    if not found:
        raise HTTPException(404, f"Dataset {name}/{version} rỗng hoặc không tồn tại")
    yaml_path = os.path.join(local_root, "data.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(400, "Dataset thiếu data.yaml")
    # Ép path tuyệt đối để YOLO tìm đúng images/labels (data.yaml gốc lưu path khác)
    cfg = _yaml.safe_load(open(yaml_path)) or {}
    cfg["path"] = local_root
    with open(yaml_path, "w") as f:
        _yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    # Suy ra task từ cấu trúc: có labels/ → detect; có train/<class>/ → classify
    task = "detect" if os.path.isdir(os.path.join(local_root, "labels")) else "classify"
    return yaml_path, task


class ExportBody(BaseModel):
    project_id: int
    tag: str = "v1"
    format: str = "YOLO 1.1"


def _run_job(job_id: str, cmd: list[str], cwd: str):
    """Chạy lệnh ở thread nền, ghi log đuôi + cập nhật trạng thái."""
    job = JOBS[job_id]
    job["status"] = "running"
    tail = deque(maxlen=200)
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        job["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            job["log"] = list(tail)
            # Bắt tiến độ epoch của YOLO: "  3/50" dạng epoch hiện tại.
            # Chỉ nhận token có ĐÚNG một dấu '/' để không nuốt nhầm dấu thời gian
            # kiểu "2026/05/29" (3 phần) — vốn làm vỡ a, b = tok.split('/').
            for tok in line.split():
                if tok.count("/") == 1 and tok.replace("/", "").isdigit():
                    a, b = tok.split("/")
                    if b == str(job.get("epochs", 0)) and int(b) > 0:
                        job["progress"] = round(int(a) / int(b), 3)
                    break
        code = proc.wait()
        job["status"] = "done" if code == 0 else "failed"
        job["progress"] = 1.0 if code == 0 else job.get("progress", 0)
        job["exit_code"] = code
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["finished_at"] = time.time()


def _start(kind: str, cmd: list[str], cwd: str, extra: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        JOBS[job_id] = {
            "id": job_id, "kind": kind, "status": "queued",
            "progress": 0.0, "log": [], "started_at": time.time(),
            **extra,
        }
    t = threading.Thread(target=_run_job, args=(job_id, cmd, cwd), daemon=True)
    t.start()
    return job_id


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train(body: TrainBody):
    # Nếu chọn mô hình nền nạp sẵn → dùng đường dẫn base/ trong MinIO,
    # train.py sẽ tự tải về và fine-tune từ đó.
    model_arg = body.model
    if body.base_model:
        bm = next((m for m in _load_base_models()
                   if m["id"] == body.base_model), None)
        if not bm:
            raise HTTPException(404, "Không tìm thấy mô hình nền")
        model_arg = bm["file"]  # vd "base/object-detector.pt"

    # Nếu chọn dataset cụ thể: tải từ MinIO về local, dùng data.yaml đã tải
    # và bỏ qua dvc pull (dataset không nằm trong DVC mà do simple-label/
    # dataset-api ghi thẳng vào bucket 'datasets').
    data_arg = body.data
    task = body.task
    skip_pull = body.skip_pull
    if body.dataset:
        version = body.dataset_version or "v1"
        data_arg, task = _prepare_dataset(body.dataset, version)
        skip_pull = True

    cmd = ["python", "training/train.py",
           "--data", data_arg, "--model", model_arg,
           "--epochs", str(body.epochs),
           "--imgsz", str(body.imgsz),
           "--batch", str(body.batch),
           "--patience", str(body.patience),
           "--val-split", str(body.val_split),
           "--name", body.name,
           "--task", task]
    if body.dataset:
        cmd += ["--dataset", body.dataset]
    if body.dataset_version:
        cmd += ["--dataset-version", body.dataset_version]
    if skip_pull:
        cmd.append("--skip-pull")
    job_id = _start("train", cmd, REPO,
                    {"name": body.name, "epochs": body.epochs})
    return {"job_id": job_id, "status": "queued"}


@app.post("/export")
def export(body: ExportBody):
    cmd = ["python", "training/export_from_cvat.py",
           "--project-id", str(body.project_id),
           "--format", body.format, "--tag", body.tag]
    job_id = _start("export", cmd, REPO,
                    {"name": f"export {body.tag}", "epochs": 0})
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
def list_jobs():
    with _lock:
        jobs = sorted(JOBS.values(), key=lambda j: j["started_at"], reverse=True)
    # Không trả full log trong danh sách cho gọn
    return {"jobs": [{k: v for k, v in j.items() if k != "log"} for j in jobs]}


@app.get("/jobs/{job_id}")
def job_detail(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job
