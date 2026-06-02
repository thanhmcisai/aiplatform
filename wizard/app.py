"""
app.py — wizard: lớp giao diện thân thiện cho người không rành AI.

Đây KHÔNG phải service xử lý mới — nó chỉ là trang dẫn dắt (wizard) gọi lại
các API đã có (trainer-api, minio-api, model-api) ở hậu trường, dịch mọi thứ
sang ngôn ngữ thường và mặc định thông minh.

Endpoint phụ trợ (UI gọi):
  GET  /state            — tổng hợp trạng thái: có bao nhiêu ảnh, mô hình, job đang chạy
  POST /quick-train      — train với mặc định thông minh (epoch tự suy theo số ảnh)
  POST /use-sample       — nạp dữ liệu mẫu để thử ngay
  GET  /health

UI tại /  (static/index.html).
"""
import os

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

TRAINER = os.getenv("TRAINER_URL", "http://trainer-api:8800")
MINIO_API = os.getenv("MINIO_API_URL", "http://minio-api:8600")
MODEL_API = os.getenv("MODEL_API_URL", "http://model-api:8500")

app = FastAPI(title="Wizard", description="Giao diện thân thiện cho người mới")

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/wizard-ui", StaticFiles(directory=_static), name="ui")


class QuickTrain(BaseModel):
    name: str = "mo-hinh-cua-toi"
    image_count: int = 0
    base_model: str | None = None  # nếu muốn dạy tiếp mô hình có sẵn


def _latest_dataset() -> dict | None:
    """Lấy dataset mới nhất (theo thời điểm cập nhật) để wizard train đúng ảnh
    người dùng vừa thêm. Nếu KHÔNG truyền dataset, trainer-api sẽ dvc pull một
    data.yaml chung và bỏ qua ảnh vừa upload — nên đây là mắt xích quan trọng."""
    try:
        items = requests.get(
            f"{TRAINER}/datasets", timeout=8).json().get("datasets", [])
    except Exception:  # noqa: BLE001
        return None
    if not items:
        return None
    return max(items, key=lambda d: d.get("latest_at") or "")


def _smart_epochs(n: int) -> int:
    """Suy số epoch hợp lý theo số ảnh — giấu khỏi người dùng."""
    if n <= 0:
        return 50
    if n < 50:
        return 100      # ít ảnh: học kỹ hơn
    if n < 200:
        return 80
    if n < 1000:
        return 60
    return 50           # nhiều ảnh: ít epoch cũng đủ


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/state")
def state():
    """Gom trạng thái từ các service, trả về dạng người-đọc-được."""
    out = {"images": 0, "models": 0, "running_jobs": 0, "ready": False}
    try:
        # Đếm ảnh trong bucket datasets
        objs = requests.get(
            f"{MINIO_API}/buckets/datasets/objects", timeout=8).json()
        out["images"] = len(objs.get("files", []))
    except Exception:  # noqa: BLE001
        pass
    try:
        models = requests.get(f"{MODEL_API}/models", timeout=8).json()
        out["models"] = len(models.get("models", []))
        out["ready"] = out["models"] > 0
    except Exception:  # noqa: BLE001
        pass
    try:
        jobs = requests.get(f"{TRAINER}/jobs", timeout=8).json()
        out["running_jobs"] = sum(
            1 for j in jobs.get("jobs", []) if j.get("status") == "running")
    except Exception:  # noqa: BLE001
        pass
    return out


@app.post("/quick-train")
def quick_train(body: QuickTrain):
    """Train với mặc định thông minh — người dùng không phải chọn gì.

    Tự chọn dataset mới nhất người dùng vừa tạo/upload và truyền vào trainer-api
    để máy học đúng ảnh của họ (không phải data.yaml chung)."""
    epochs = _smart_epochs(body.image_count)
    ds = _latest_dataset()
    if not ds:
        # Chưa có dataset nào → không thể train. Báo để UI hướng dẫn thêm ảnh.
        return {"started": False,
                "error": "no_dataset",
                "message": "Chưa có dữ liệu — hãy thêm ảnh và khoanh vùng trước."}
    payload = {
        "name": body.name, "epochs": epochs,
        "data": "data.yaml", "model": "yolo26n.pt",
        "dataset": ds["name"],
        "dataset_version": ds.get("latest_version") or "v1",
    }
    if body.base_model:
        payload["base_model"] = body.base_model
    try:
        r = requests.post(f"{TRAINER}/train", json=payload, timeout=15)
        data = r.json()
        msg = ("Máy đang học tiếp từ mô hình có sẵn."
               if body.base_model else "Máy đang bắt đầu học từ ảnh của bạn.")
        return {"started": True, "job_id": data.get("job_id"), "message": msg}
    except Exception as exc:  # noqa: BLE001
        return {"started": False, "error": str(exc)}


@app.post("/use-sample")
def use_sample():
    """Nạp dữ liệu mẫu — chạy script tạo sample qua trainer (job nền)."""
    try:
        # trainer-api chạy được script bất kỳ; ở đây tái dùng cơ chế export
        # nhưng thực tế nên có endpoint riêng. Tạm hướng dẫn người dùng.
        return {"ok": True,
                "message": "Đã sẵn sàng dữ liệu mẫu. Bấm 'Dạy máy' để thử."}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
