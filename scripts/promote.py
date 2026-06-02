"""
promote.py — Nối mắt xích Registry → Production qua CLI.

Tự động hóa việc đổi stage model, có cổng chất lượng (quality gate):
chỉ promote lên Production nếu mAP50 vượt ngưỡng.

Chạy:
  python promote.py --version 8 --to Staging
  python promote.py --version 8 --to Production --min-map50 0.90
"""
import argparse
import os

import mlflow
import requests
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "cv-detector")
BENTOML_URL = os.getenv("BENTOML_URL", "http://localhost:3000")


def get_map50(client: MlflowClient, version: str) -> float:
    """Lấy metric mAP50 của run gắn với version này."""
    mv = client.get_model_version(MODEL_NAME, version)
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("mAP50", 0.0)


def promote(version: str, to_stage: str, min_map50: float):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Quality gate cho Production
    if to_stage == "Production":
        score = get_map50(client, version)
        print(f">> mAP50 của v{version}: {score:.4f} (ngưỡng {min_map50})")
        if score < min_map50:
            raise SystemExit(
                f"X  Từ chối promote: mAP50 {score:.4f} < {min_map50}")

    # Tự động chuyển model Production cũ sang Archived
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage=to_stage,
        archive_existing_versions=(to_stage == "Production"),
    )
    print(f">> {MODEL_NAME} v{version} → {to_stage}")

    # Đóng mắt xích cuối: báo BentoML nạp lại model ngay
    if to_stage == "Production":
        try:
            r = requests.post(f"{BENTOML_URL}/reload", json={}, timeout=120)
            print(f">> BentoML reload: {r.json()}")
        except Exception as exc:  # noqa: BLE001
            print(f"!! Không gọi được BentoML reload ({exc}). "
                  f"Service sẽ tự nạp ở chu kỳ kế tiếp.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--to", default="Staging",
                   choices=["Staging", "Production", "Archived", "None"])
    p.add_argument("--min-map50", type=float, default=0.85,
                   help="Ngưỡng tối thiểu để lên Production")
    args = p.parse_args()
    promote(args.version, args.to, args.min_map50)
