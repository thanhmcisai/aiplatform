"""
service.py — BentoML service: load model từ MLflow Registry, serve REST API.

Nối các mắt xích còn hở:
  - /predict  : tên endpoint khớp sơ đồ (alias của /detect)
  - /reload   : nạp lại model Production mới nhất mà KHÔNG restart service
  - auto-reload theo chu kỳ: kiểm tra version Production định kỳ

Endpoint:
  POST /predict | /detect  — nhận ảnh, trả bounding boxes + class + confidence
  POST /reload             — ép nạp lại model Production hiện tại
  GET  /health             — trạng thái + version đang phục vụ

Chạy:
  bentoml serve service:CVDetector --port 3000
"""
import os
import threading
import time

import bentoml
import mlflow
import numpy as np
from PIL import Image
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "cv-detector")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
# Chu kỳ tự kiểm tra version mới (giây). 0 = tắt auto-reload.
RELOAD_INTERVAL = int(os.getenv("RELOAD_INTERVAL", "300"))


@bentoml.service(
    name="cv_detector",
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class CVDetector:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.client = MlflowClient()
        self.model = None
        self.current_version = None
        self._lock = threading.Lock()
        self._load_latest()

        if RELOAD_INTERVAL > 0:
            t = threading.Thread(target=self._watch, daemon=True)
            t.start()
            print(f">> Auto-reload bật, kiểm tra mỗi {RELOAD_INTERVAL}s")

    def _production_version(self) -> str | None:
        """Tìm version đang ở stage Production."""
        for v in self.client.search_model_versions(f"name='{MODEL_NAME}'"):
            if v.current_stage == MODEL_STAGE:
                return v.version
        return None

    def _load_latest(self) -> bool:
        """Nạp model Production nếu khác version đang chạy. True nếu có đổi."""
        from ultralytics import YOLO

        target = self._production_version()
        if target is None:
            print(f"!! Không tìm thấy {MODEL_NAME} ở stage {MODEL_STAGE}")
            return False
        if target == self.current_version:
            return False  # đã là mới nhất

        uri = f"models:/{MODEL_NAME}/{target}"
        local_dir = mlflow.artifacts.download_artifacts(uri)
        weights = os.path.join(local_dir, "best.pt")
        with self._lock:
            self.model = YOLO(weights)
            self.current_version = target
        print(f">> Đã nạp {MODEL_NAME} v{target} ({MODEL_STAGE})")
        return True

    def _watch(self):
        """Vòng lặp nền: định kỳ kiểm tra & nạp version mới."""
        while True:
            time.sleep(RELOAD_INTERVAL)
            try:
                self._load_latest()
            except Exception as exc:  # noqa: BLE001
                print(f"[reload] lỗi: {exc}")

    def _infer(self, image: Image.Image, conf: float) -> dict:
        with self._lock:
            model = self.model
        results = model.predict(np.array(image), conf=conf, verbose=False)
        r = results[0]
        # Classify: r.probs có giá trị, r.boxes là None hoặc rỗng
        if getattr(r, "probs", None) is not None:
            top1 = int(r.probs.top1)
            return {
                "task": "classify",
                "class": r.names[top1],
                "confidence": round(float(r.probs.top1conf), 4),
                "all": [{"class": r.names[i], "confidence": round(float(p), 4)}
                        for i, p in enumerate(r.probs.data.tolist())],
                "model_version": self.current_version,
            }
        # Detection
        detections = [{
            "class": r.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
        } for box in r.boxes]
        return {"task": "detect", "count": len(detections),
                "detections": detections,
                "model_version": self.current_version}

    @bentoml.api
    def predict(self, image: Image.Image, conf: float = 0.25) -> dict:
        """Nhận diện đối tượng (tên endpoint khớp sơ đồ)."""
        return self._infer(image, conf)

    @bentoml.api
    def detect(self, image: Image.Image, conf: float = 0.25) -> dict:
        """Alias của /predict."""
        return self._infer(image, conf)

    @bentoml.api
    def reload(self) -> dict:
        """Ép nạp lại model Production (gọi sau khi promote)."""
        changed = self._load_latest()
        return {"reloaded": changed, "model_version": self.current_version}

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE,
                "version": self.current_version}
