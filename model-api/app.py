"""
app.py — model-api: lớp REST tối thiểu bọc MLflow.

MLflow chạy nền làm engine (tracking + registry, lưu Postgres + MinIO),
không phơi UI. Các hệ thống khác chỉ gọi API gọn ở đây, không cần biết MLflow.

Endpoint:
  GET  /models                      — liệt kê model + version + stage + metrics
  GET  /models/{name}               — chi tiết các version của một model
  GET  /models/{name}/production     — version đang Production + link tải weights
  POST /models/{name}/promote        — đổi stage (body: {version, stage, min_map50?})
  GET  /runs?experiment=...          — liệt kê run + metrics (thay cho UI tracking)
  GET  /health

Chạy:
  uvicorn app:app --host 0.0.0.0 --port 8500
"""
import os
from typing import Optional

import mlflow
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BENTOML_URL = os.getenv("BENTOML_URL", "http://bentoml:3000")

app = FastAPI(title="Model API", description="Lớp REST tối thiểu bọc MLflow")
mlflow.set_tracking_uri(MLFLOW_URI)

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/try")
def try_page():
    """Trang 'Thử model' — kéo thả ảnh, gọi BentoML, vẽ kết quả."""
    return FileResponse(os.path.join(_static, "try.html"))


@app.post("/try/predict")
async def try_predict(request: Request):
    """Proxy ảnh sang BentoML để tránh CORS/credential từ trình duyệt."""
    body = await request.body()
    ct = request.headers.get("content-type", "application/octet-stream")
    try:
        r = requests.post(f"{BENTOML_URL}/predict",
                          data=body, headers={"Content-Type": ct}, timeout=60)
        return r.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Không gọi được model: {exc}") from exc


def client() -> MlflowClient:
    return MlflowClient()


def _map50(c: MlflowClient, run_id: str) -> float:
    try:
        return c.get_run(run_id).data.metrics.get("mAP50", 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


class PromoteBody(BaseModel):
    version: str
    stage: str = "Staging"
    min_map50: float = 0.85


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    c = client()
    out = []
    for m in c.search_registered_models():
        versions = []
        for v in c.search_model_versions(f"name='{m.name}'"):
            versions.append({
                "version": v.version,
                "stage": v.current_stage or "None",
                "mAP50": round(_map50(c, v.run_id), 4),
            })
        out.append({"name": m.name, "versions": versions})
    return {"models": out}


@app.get("/models/{name}")
def model_detail(name: str):
    c = client()
    vs = c.search_model_versions(f"name='{name}'")
    if not vs:
        raise HTTPException(404, f"Không tìm thấy model '{name}'")
    return {
        "name": name,
        "versions": [{
            "version": v.version,
            "stage": v.current_stage or "None",
            "mAP50": round(_map50(c, v.run_id), 4),
            "run_id": v.run_id,
        } for v in vs],
    }


@app.get("/models/{name}/production")
def production_version(name: str):
    c = client()
    for v in c.search_model_versions(f"name='{name}'"):
        if v.current_stage == "Production":
            uri = f"models:/{name}/Production"
            return {
                "name": name,
                "version": v.version,
                "mAP50": round(_map50(c, v.run_id), 4),
                "model_uri": uri,
            }
    raise HTTPException(404, f"'{name}' chưa có version Production")


@app.post("/models/{name}/promote")
def promote(name: str, body: PromoteBody):
    c = client()
    # Quality gate khi lên Production
    if body.stage == "Production":
        mv = c.get_model_version(name, body.version)
        score = _map50(c, mv.run_id)
        if score < body.min_map50:
            raise HTTPException(
                400, f"Từ chối: mAP50 {score:.4f} < ngưỡng {body.min_map50}")

    c.transition_model_version_stage(
        name=name, version=body.version, stage=body.stage,
        archive_existing_versions=(body.stage == "Production"),
    )

    reloaded = None
    if body.stage == "Production":
        try:
            r = requests.post(f"{BENTOML_URL}/reload", json={}, timeout=120)
            reloaded = r.json()
        except Exception as exc:  # noqa: BLE001
            reloaded = {"error": str(exc)}

    return {"name": name, "version": body.version, "stage": body.stage,
            "bentoml_reload": reloaded}


@app.get("/runs")
def list_runs(experiment: Optional[str] = None, limit: int = 50):
    c = client()
    exps = ([c.get_experiment_by_name(experiment)] if experiment
            else c.search_experiments())
    exps = [e for e in exps if e]
    if not exps:
        return {"runs": []}
    runs = c.search_runs([e.experiment_id for e in exps], max_results=limit)
    return {"runs": [{
        "run_id": r.info.run_id,
        "name": r.info.run_name,
        "status": r.info.status,
        "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
    } for r in runs]}
