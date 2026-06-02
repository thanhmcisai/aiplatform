# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-hosted, single-server MLOps stack for Computer Vision, orchestrated entirely by `docker-compose.yml`. The end-to-end loop:

```
annotate (CVAT / simple-label) → version datasets (DVC → MinIO) → train (YOLO/Ultralytics)
  → track + registry (MLflow) → serve REST (BentoML)
```

PostgreSQL holds metadata, MinIO is object storage, Keycloak + oauth2-proxy provide SSO, and Caddy is the single-port gateway. Comments, docs, and all user-facing UI are in **Vietnamese** — match that language when editing them.

This is **not a git repository** and has **no test suite or lint config**. "Testing" means running `scripts/smoke-test.sh` against a live stack. The `# noqa: BLE001` markers throughout follow ruff conventions even though no ruff config is checked in.

## Common commands

```bash
# Bring up / tear down the whole stack
docker compose up -d
docker compose ps
docker compose logs -f <service>

# After editing one self-written service (dashboard, model-api, minio-api,
# kc-admin-api, dataset-api, simple-label, trainer-api, wizard):
docker compose up -d --build <service>

# Health-check every service (run ~1-2 min after `up`)
bash scripts/smoke-test.sh

# Generate a synthetic YOLO dataset for smoke-testing the train path (no CVAT needed)
python scripts/make_sample_dataset.py --n-train 40 --n-val 10
```

Training, serving, and promotion can run on the host (defaulting to `localhost` URLs) or inside containers (env vars override to Docker-DNS hostnames like `http://mlflow:5000`):

```bash
# Train (host): set MLFLOW/AWS/S3 env vars first — see README §4. --skip-pull skips `dvc pull`.
python training/train.py --data data.yaml --model yolo26n.pt --epochs 50 --name exp1

# Export CVAT labels + version with DVC in one step
python training/export_from_cvat.py --project-id 1 --format "YOLO 1.1" --tag v2

# Promote a registry version (quality gate + auto-reload of BentoML)
python scripts/promote.py --version 8 --to Production --min-map50 0.90

# Serve
bentoml serve serving/service:CVDetector --port 3000
```

There is no single-test runner because there are no unit tests; the smallest verification loop is: sample dataset → `train.py --epochs 3 --skip-pull` → `curl localhost:8500/models`. See `GETTING_STARTED.md` for the full smoke-test walkthrough.

## Architecture: the "API-wrapper" pattern

The defining design choice is that **heavy third-party UIs are hidden behind self-written FastAPI services** that expose a clean Vietnamese REST API *and* serve their own HTML UI. MLflow's UI port is unmapped, and the MinIO/Keycloak consoles are kept only for debugging. External callers and end users go through the wrappers:

| Wrapper (port) | Wraps | Gateway path |
|---|---|---|
| `model-api` (8500) | MLflow tracking + registry | `/models/` |
| `minio-api` (8600) | MinIO S3 (via boto3) | `/minio/` |
| `kc-admin-api` (8700) | Keycloak Admin REST | `/users/` |
| `dataset-api` (8400) | MinIO + MLflow combined into a "dataset" concept | `/datasets/` |
| `trainer-api` (8800) | runs `train.py`/`export_from_cvat.py` as background jobs | `/train/` |
| `simple-label` (8300) | beginner annotator, writes YOLO straight to MinIO | `/label/` |
| `wizard` (8900) | goal-oriented homepage, calls the other APIs | `/` |
| `dashboard` (8000) | overview pane | `/dashboard/` |

Every self-written service is the **same shape**: a FastAPI `app.py`, a `static/` dir with HTML, and a near-identical `python:3.11-slim` + uvicorn `Dockerfile`. `GET /` returns `static/index.html`, `/ui` mounts the static dir, and the rest are JSON endpoints. When adding a service, copy this structure and add both a `build:` entry in `docker-compose.yml` and a `handle_path` block in `gateway/Caddyfile`.

## Gateway, SSO, and sub-path constraints

`gateway/Caddyfile` (Caddy on :80) is the single entry point. It uses `forward_auth` against `oauth2-proxy` so every route requires a Keycloak login — **except `/api/*`** (BentoML), which is intentionally left open for machine-to-machine calls. It strips `X-Frame-Options` so UIs can be embedded in the studio shell's iframes.

Because gateway routes use `handle_path` (which strips the prefix), **each service must work correctly when served under a sub-path** (e.g. `/models/`). This is why MLflow runs with `--static-prefix` and MinIO needs `MINIO_BROWSER_REDIRECT_URL` when unified — relative asset paths break otherwise.

`/` serves the beginner `wizard`; `/studio/` serves the full expert sidebar shell (`gateway/shell/index.html`).

## The "closed-loop" automation (recurring theme)

Comments repeatedly talk about "nối mắt xích" (closing the gaps) — the pipeline is wired to need no manual MLflow/BentoML clicking:

- `train.py` auto-registers the trained model as `cv-detector` and transitions it to **Staging**. Experiment name is `cv-detection`; the registered model name is always `cv-detector` (`REGISTERED_MODEL`).
- Promotion (`promote.py` or `model-api`'s `/promote`) enforces a **mAP50 quality gate**, auto-archives the previous Production version, then **calls BentoML `/reload`**.
- `serving/service.py` (`CVDetector`) reloads the Production model with **zero restart** — either via the `/reload` endpoint or a background thread polling every `RELOAD_INTERVAL` seconds. It serves both `/predict` and its alias `/detect`, and auto-detects detection vs. classification from the YOLO result (`r.probs` present ⇒ classify).
- MLflow uses **legacy stage transitions** (Staging/Production/Archived), valid on the pinned MLflow 2.16.2 — don't replace these with the alias/tag API.

## Training specifics

- Default model is **YOLO26** (`yolo26n.pt`, auto-downloaded). Supports `--task detect` (default) and `--task classify`.
- `resolve_model()` in `train.py`: a `base/...pt` model arg is downloaded from the MinIO `models` bucket for fine-tuning; anything else (e.g. `yolo26n.pt`) is used as-is. Base models are declared in `base-models/base-models.json` and must be `.pt` files (ONNX/TF cannot be fine-tuned).
- `trainer-api` runs training **inside its own container** as in-memory background threads (the `JOBS` dict — lost on restart). GPU is opt-in: uncomment the `deploy.resources` block in `docker-compose.yml` (needs nvidia-docker); CPU works but is smoke-test-only speed.
- DVC remote is the MinIO `datasets` bucket (`training/dvc-config-example`). `--dataset`/`--dataset-version` are logged as MLflow tags so `dataset-api` can trace which run used which dataset version.

## Storage & data layout

- **Postgres** hosts three DBs created on first boot by `scripts/init-multi-db.sh`: `mlflow`, `cvat`, `keycloak`.
- **MinIO buckets** (created by the `minio-init` one-shot): `datasets` (DVC remote), `mlflow` (artifacts), `models` (base models under `base/`).
- `simple-label` and the dataset layout write the standard YOLO structure (`images/{train,val}`, `labels/{train,val}`, `data.yaml`) directly into the `datasets` bucket so `train.py` and `dataset-api` can consume it without conversion.

## Optional add-ons

- **SAM auto-annotation**: bring up `serverless/docker-compose.serverless.yml` then `bash serverless/deploy-sam.sh` (see `serverless/README.md`).
- All secrets live in `.env` — every password is a placeholder and must be changed before production.
