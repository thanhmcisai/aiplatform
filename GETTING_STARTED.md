# Hướng dẫn khởi động & smoke-test

Tài liệu này giúp bạn lần đầu đưa hệ thống lên và chạy thử trọn vòng dataset → train, phát hiện sớm chỗ vướng trước khi dùng dữ liệu/GPU thật.

## Bước 0 — Khởi động stack

```bash
cd cv-mlops
# Đổi mật khẩu trong .env trước khi chạy thật
docker compose up -d
```

Lần đầu sẽ lâu: build 4 service tự viết (dashboard, model-api, minio-api, kc-admin-api) và kéo image nặng (CVAT, Keycloak, MLflow). Đợi 2-3 phút.

Theo dõi:
```bash
docker compose ps              # tất cả nên ở trạng thái "running"/"healthy"
docker compose logs -f keycloak # xem realm import xong chưa
```

## Bước 1 — Smoke-test: kiểm tra mọi service đã lên

```bash
bash scripts/smoke-test.sh
```

Script gọi health-check từng service. Nếu service nặng chưa kịp lên, đợi thêm 1-2 phút rồi chạy lại. Mong đợi: tất cả `[OK]`, model-api trả `{"models":[]}` (registry trống là đúng), minio-api liệt kê 3 bucket (`datasets`, `models`, `mlflow`).

Nếu một service `[FAIL]` kéo dài, xem log:
```bash
docker compose logs <tên-service>
```

## Bước 2 — Tạo dataset mẫu (không cần CVAT)

Để xác nhận luồng train chạy thông trước khi gán nhãn thật:

```bash
pip install pillow ultralytics
python scripts/make_sample_dataset.py --n-train 40 --n-val 10
```

Sinh dataset YOLO mẫu (hình tròn/vuông màu) trong `data/` kèm `data.yaml` 2 class.

## Bước 3 — Train thử vài epoch (CPU cũng được)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

python training/train.py --data data.yaml --model yolo26n.pt \
  --epochs 3 --name smoke --skip-pull
```

`--skip-pull` bỏ qua `dvc pull` (dataset mẫu đã có sẵn tại chỗ). Train xong, script tự log metrics + đăng ký model `cv-detector` lên Staging trong MLflow.

Kiểm chứng:
```bash
curl http://localhost:8500/models     # nên thấy cv-detector, stage Staging
```
Hoặc mở http://localhost (đăng nhập) → mục "Mô hình".

> Lưu ý: nếu máy không tải được `yolo26n.pt` (mạng chặn GitHub releases), đổi `--model yolo26n.yaml` để train từ kiến trúc (chỉ dùng cho smoke-test, không phải mô hình thật).

## Bước 4 — Khi đã thông luồng: chuyển sang dữ liệu thật

1. Gán nhãn trên CVAT (http://localhost qua mục "Gán nhãn").
2. Cấu hình DVC một lần (xem `training/dvc-config-example`), rồi:
   ```bash
   python training/export_from_cvat.py --project-id 1 --format "YOLO 1.1" --tag v1
   ```
3. Train thật trên **máy có GPU** (CPU quá chậm cho train thực tế):
   ```bash
   python training/train.py --data data.yaml --epochs 100 --name exp1
   ```
4. Promote khi đạt: `python scripts/promote.py --version 1 --to Production --min-map50 0.90`

## Các lỗi tích hợp thường gặp lần đầu

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| smoke-test FAIL ở CVAT/Keycloak | Service nặng, chưa kịp lên — đợi thêm rồi chạy lại |
| Keycloak vòng lặp redirect | Sai `KC_HOSTNAME`/relative path — xem README phần SSO |
| train.py báo không thấy best.pt | Đã sửa: lấy từ `results.save_dir`. Cập nhật bản mới nhất |
| MinIO tải file lỗi từ trình duyệt | presigned URL dùng hostname nội bộ — đặt `MINIO_ENDPOINT` thành địa chỉ ngoài |
| model-api trả lỗi 502 | MLflow chưa lên — `docker compose logs mlflow` |
