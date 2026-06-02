"""
app.py — Management Dashboard (FastAPI), Apple-inspired UI.

Gom dữ liệu từ MLflow + MinIO thành một trang quản lý tập trung,
kèm các link nhanh tới CVAT / MLflow / MinIO / BentoML.
"""
import os

import boto3
import mlflow
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "minioadmin")

LINKS = {
    "cvat": os.getenv("CVAT_URL", "http://localhost:8080"),
    "train": os.getenv("TRAIN_URL", "http://localhost:8800/"),
    "mlflow": os.getenv("MLFLOW_UI_URL", "http://localhost:5000"),
    "minio": os.getenv("MINIO_UI_URL", "http://localhost:9001"),
    "bentoml": os.getenv("BENTOML_URL", "http://localhost:3000"),
}

# (tên, mô tả, url, icon Tabler, màu nền icon, màu icon)
SERVICES = [
    ("Huấn luyện", "Train model", LINKS["train"], "ti-player-play", "#fff4e5", "#e8930c"),
    ("CVAT", "Gán nhãn", LINKS["cvat"], "ti-tag", "#e8f0fe", "#1a73e8"),
    ("MLflow", "Tracking", LINKS["mlflow"], "ti-chart-line", "#e9f7ef", "#1d9e75"),
    ("MinIO", "Lưu trữ", LINKS["minio"], "ti-database", "#f3eefe", "#7f4ad8"),
    ("BentoML", "Serving API", LINKS["bentoml"], "ti-api", "#fdeee7", "#d85a30"),
]

app = FastAPI(title="CV MQOps Console")
mlflow.set_tracking_uri(MLFLOW_URI)

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static), name="static")


def get_stats() -> dict:
    """Thu thập số liệu từ MLflow + MinIO. Lỗi thì trả 0 thay vì sập."""
    stats = {"experiments": 0, "runs": 0, "models": 0, "production": 0,
             "datasets": 0, "storage_gb": 0.0, "registry": []}
    try:
        client = MlflowClient()
        exps = list(client.search_experiments())
        stats["experiments"] = len(exps)
        runs = 0
        for e in exps:
            runs += len(client.search_runs([e.experiment_id], max_results=1000))
        stats["runs"] = runs

        for m in client.search_registered_models():
            stats["models"] += 1
            for v in client.search_model_versions(f"name='{m.name}'"):
                if v.current_stage == "Production":
                    stats["production"] += 1
                stats["registry"].append({
                    "name": m.name, "version": v.version,
                    "stage": v.current_stage or "None",
                })
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] {exc}")

    try:
        s3 = boto3.client(
            "s3", endpoint_url=f"http://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_USER, aws_secret_access_key=MINIO_PASSWORD,
        )
        total = 0
        ds_prefixes = set()
        for obj in s3.list_objects_v2(Bucket="datasets").get("Contents", []):
            total += obj["Size"]
            ds_prefixes.add(obj["Key"].split("/")[0])
        for b in ("mlflow", "models"):
            for obj in s3.list_objects_v2(Bucket=b).get("Contents", []):
                total += obj["Size"]
        stats["datasets"] = len(ds_prefixes)
        stats["storage_gb"] = round(total / 1e9, 1)
    except Exception as exc:  # noqa: BLE001
        print(f"[minio] {exc}")

    return stats


def badge(stage: str) -> str:
    cls = {"Production": "badge-prod", "Staging": "badge-staging"}.get(
        stage, "badge-none")
    return f'<span class="badge {cls}">{stage}</span>'


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def api_stats():
    return get_stats()


@app.get("/", response_class=HTMLResponse)
def home():
    s = get_stats()

    metrics = [
        ("Datasets", f'{s["datasets"]}', "", "đã version"),
        ("Experiments", f'{s["experiments"]}', "", f'{s["runs"]} runs'),
        ("Models", f'{s["models"]}', "", f'{s["production"]} production'),
        ("Storage", f'{s["storage_gb"]}', " GB", "MinIO"),
    ]
    metric_html = "".join(
        f'<div class="metric reveal"><p class="metric-label">{label}</p>'
        f'<p class="metric-value">{val}<span class="metric-unit">{unit}</span></p>'
        f'<p class="metric-foot">{foot}</p></div>'
        for label, val, unit, foot in metrics
    )

    tile_html = "".join(
        f'<a class="tile reveal" href="{url}" target="_blank">'
        f'<div class="tile-ico" style="background:{bg};color:{fg}">'
        f'<i class="ti {icon}" aria-hidden="true"></i></div>'
        f'<div><p class="tile-name">{name}</p>'
        f'<p class="tile-desc">{desc}</p></div></a>'
        for name, desc, url, icon, bg, fg in SERVICES
    )

    rows = "".join(
        f'<div class="row"><div><p class="row-name">{r["name"]}</p>'
        f'<p class="row-meta">v{r["version"]}</p></div>{badge(r["stage"])}</div>'
        for r in s["registry"][:10]
    ) or ('<div class="row"><p class="row-meta">'
          'Chưa có model nào trong registry</p></div>')

    return PAGE.format(metrics=metric_html, tiles=tile_html, rows=rows)


PAGE = """<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CV MQOps Console</title>
<link rel="preconnect" href="https://cdnjs.cloudflare.com">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/3.17.0/tabler-icons.min.css">
<link rel="stylesheet" href="static/style.css">
</head><body>
<div class="topbar"><div class="topbar-inner">
  <div class="brand"><i class="ti ti-stack-2 brand-ico" aria-hidden="true"></i>
  <span class="brand-name">CV MQOps Console</span></div>
  <span class="status"><span class="status-dot"></span>Tất cả dịch vụ hoạt động</span>
</div></div>

<div class="wrap">
  <div class="hero">
    <h1>Overview</h1>
    <p>Computer Vision MLOps · một máy chủ</p>
  </div>

  <div class="grid grid-4 metrics">{metrics}</div>

  <p class="section-label">Dịch vụ</p>
  <div class="grid grid-4 services">{tiles}</div>

  <p class="section-label">Model registry</p>
  <div class="panel">{rows}</div>

  <div class="actions">
    <button class="btn btn-primary"
      onclick="location.href='{train}'">Huấn luyện model</button>
    <button class="btn btn-ghost"
      onclick="location.href='{mlflow}'">Mở MLflow</button>
    <button class="btn btn-ghost"
      onclick="location.href='{cvat}'">Mở CVAT</button>
    <button class="btn btn-ghost"
      onclick="location.href='{bento}'">Mở API docs</button>
  </div>
</div>
</body></html>"""

PAGE = (PAGE.replace("{train}", LINKS["train"])
        .replace("{mlflow}", LINKS["mlflow"])
        .replace("{cvat}", LINKS["cvat"])
        .replace("{bento}", LINKS["bentoml"]))
