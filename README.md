# CV MLOps Stack

Hệ thống MLOps hoàn chỉnh cho Computer Vision, chạy trên **một server** bằng Docker Compose. Toàn bộ self-host, không phụ thuộc dịch vụ ngoài.

```
Gán nhãn (CVAT) → Version (DVC) → Train (YOLO) → Tracking/Registry (MLflow) → Serve API (BentoML)
                         �‾‾‾‾‾‾‾‾  MinIO (storage) + PostgreSQL (metadata)  ‾‾‾‾‾‾‾‾↾
                                   Dashboard — quản lý tập trung
```

## Thành phần

| Dịch vụ | Vai trò | URL |
|---|---|---|
| CVAT | Gán nhãn ảnh (bbox, polygon, segmentation) | http://localhost:8080 |
| MLflow | Tracking experiment + Model Registry | http://localhost:5000 |
| MinIO | Object storage (dataset + artifacts) | http://localhost:9001 |
| BentoML | Serve model thành REST API | http://localhost:3000 |
| Dashboard | Quản lý tập trung | http://localhost:8000 |
| PostgreSQL | Metadata cho MLflow & CVAT | :5432 |

## Cấu trúc thư mục

```
cv-mlops/
├── docker-compose.yml        # toàn bộ stack
├── .env                      # cấu hình mật khẩu (đổi trước khi production)
├── scripts/
│   └── init-multi-db.sh      # tạo nhiều DB trong Postgres
├── dashboard/                # FastAPI dashboard
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/                 # training pipeline
│   ├── train.py              # YOLO + DVC + MLflow
│   ├── data.yaml             # cấu hình dataset
│   ├── dvc-config-example
│   └── requirements.txt
└── serving/                  # BentoML serving
    ├── service.py
    └── bentofile.yaml
```

## 1. Khởi động hạ tầng

```bash
cd cv-mlops
# Đổi mật khẩu trong .env trước!
docker compose up -d
```

Kiểm tra:
```bash
docker compose ps          # 5 dịch vụ chạy
open http://localhost:8000 # dashboard
```

Buckets MinIO (`datasets`, `mlflow`, `models`) được tạo tự động.

## 2. Gán nhãn (CVAT)

1. Vào http://localhost:8080, đăng nhập (`admin` / mật khẩu trong `.env`).
2. Tạo project → upload ảnh → gán nhãn.

## 3. Export + version dataset (tự động)

Thay vì export tay rồi chạy DVC, dùng script nối liền cả hai:

```bash
cd training
pip install -r requirements.txt
dvc init   # chỉ lần đầu — xem dvc-config-example để cấu hình remote MinIO

export CVAT_URL=http://localhost:8080 CVAT_USER=admin CVAT_PASSWORD=admin_pw
python export_from_cvat.py --project-id 1 --format "YOLO 1.1" --tag v2
```

Script tự: gọi CVAT API export → giải nén vào `data/` → `dvc add` + `git commit` + `dvc push`. Mỗi lần chạy = một version dataset mới trên MinIO.

## 4. Train model

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

python train.py --data data.yaml --model yolo26n.pt --epochs 50 --name exp1
```

Script tự động: `dvc pull` dataset → train → log metrics/weights vào MLflow → đăng ký model `cv-detector` và **tự đưa lên Staging**.

Xem kết quả ở http://localhost:5000.

## 5. Promote model (CLI, có quality gate)

Thay vì vào MLflow UI bấm tay:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export BENTOML_URL=http://localhost:3000

# Lên Production, chỉ qua nếu mAP50 >= 0.90
python ../scripts/promote.py --version 8 --to Production --min-map50 0.90
```

Script tự: kiểm tra mAP50 (từ chối nếu dưới ngưỡng) → đổi stage → archive Production cũ → **gọi `/reload` của BentoML** để API nạp model mới ngay lập tức.

## 6. Serve API (BentoML, tự reload)

```bash
cd serving
pip install bentoml ultralytics mlflow boto3 pillow
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export RELOAD_INTERVAL=300   # tự kiểm tra version Production mới mỗi 5 phút

bentoml serve service:CVDetector --port 3000
```

Gọi API từ hệ thống khác (endpoint `/predict` khớp sơ đồ; `/detect` là alias):
```bash
curl -X POST http://localhost:3000/predict \
  -F "image=@test.jpg" -F "conf=0.3"
```

Trả về (kèm `model_version` để truy vết):
```json
{"count": 2, "model_version": "8", "detections": [
  {"class": "person", "confidence": 0.94, "bbox": [12.0, 30.0, 220.0, 410.0]}
]}
```

Service tự nạp model Production mới theo 2 cách: gọi `/reload` ngay sau promote, hoặc tự kiểm tra mỗi `RELOAD_INTERVAL` giây — **không cần restart**.

Build thành Docker image để deploy:
```bash
bentoml build
bentoml containerize cv_detector:latest
```

## Vòng lặp cải tiến

Gán nhãn thêm → DVC version mới → train lại → so sánh trên MLflow → promote nếu tốt hơn → BentoML tự load version Production mới.

## Hợp nhất giao diện (tùy chọn)

Mặc định mỗi service có cổng riêng (cách đơn giản nhất). Nếu muốn trải nghiệm "một ứng dụng", bật gateway:

```bash
docker compose up -d gateway
# Truy cập tất cả qua MỘT cổng:
open http://localhost
```

Gateway (Caddy) làm 2 việc: gom mọi UI về cổng 80, và phục vụ một shell có sidebar — bấm chuyển giữa CVAT / MLflow / MinIO / dashboard mà không nhảy tab, mỗi service render trong iframe.

Lưu ý kỹ thuật khi hợp nhất:
- Caddy gỡ header `X-Frame-Options` để các UI nhúng được trong iframe.
- MLflow và MinIO dùng đường dẫn tuyệt đối nội bộ; nếu CSS/JS của chúng lỗi khi chạy dưới `/mlflow/` hoặc `/minio/`, cần đặt base-path: MLflow chạy với `--static-prefix /mlflow`, MinIO với biến `MINIO_BROWSER_REDIRECT_URL`.
- Đăng nhập mỗi service vẫn riêng. Để thực sự liền mạch (một lần đăng nhập), cần thêm SSO (vd Authentik/Keycloak) trước gateway — đây là bước nâng cao, chỉ nên làm khi lên production thật.

Nếu chỉ chạy nội bộ một server cho team nhỏ, dùng từng service riêng (mỗi cổng) là đủ và ít rắc rối nhất.

## SSO — đăng nhập một lần (Keycloak)

Bật toàn bộ stack có SSO:

```bash
docker compose up -d
open http://localhost          # tự chuyển tới trang đăng nhập tiếng Việt
# Đăng nhập: admin / admin (đổi ngay trong production)
```

Cách hoạt động: mọi truy cập qua gateway đều bị `oauth2-proxy` chặn; nếu chưa đăng nhập sẽ chuyển tới Keycloak. Đăng nhập một lần rồi dùng được CVAT, MLflow, MinIO, dashboard mà không phải nhập lại.

Cấu hình:
- Realm `mlops` và các client (CVAT, MLflow, MinIO, oauth2-proxy) được import tự động từ `sso/keycloak/realm-mlops.json`.
- Admin Keycloak: http://localhost:8180 (tài khoản trong `.env`).
- Trang đăng nhập dùng theme `mlops` (tiếng Việt, phong cách Apple) trong `sso/themes/`.

Để mỗi service thực sự nhận danh tính từ Keycloak (không chỉ chặn ở cổng), cấu hình OIDC trong từng service — xem tài liệu: CVAT hỗ trợ social auth, MLflow dùng `mlflow server --app-name basic-auth` + proxy header, MinIO bật OpenID trong console. README này dựng sẵn lớp chặn SSO ở gateway; tích hợp OIDC sâu vào từng service là bước tiếp theo tùy nhu cầu.

## Tiếng Việt hóa

Hai mức:
- Lớp vỏ (shell, dashboard, trang đăng nhập): Việt hóa 100%, vì ta sở hữu mã nguồn.
- CVAT: dịch best-effort bằng `sso/cvat-i18n/cvat-vi.js` — shell tự chèn script này khi mở CVAT, dịch các nhãn chính (Projects, Tasks, Create...). Bổ sung từ điển trong file đó khi cần.

Lưu ý thẳng: MLflow và MinIO gần như không hỗ trợ đa ngôn ngữ, nên giữ nguyên tiếng Anh. Việc dịch đè runtime chỉ là giải pháp tạm, dễ vỡ khi service cập nhật — đừng kỳ vọng phủ 100%.

## MLflow tối thiểu — chỉ dùng qua API

MLflow vẫn chạy nền làm engine (tracking + registry, lưu Postgres + MinIO) nhưng **không phơi UI ra ngoài** (đã gỡ port mapping). Trước nó là `model-api` — một lớp REST gọn để hệ thống khác gọi mà không cần học MLflow SDK.

Các endpoint của `model-api` (cổng 8500, hoặc qua gateway tại `/models`):

```bash
# Liệt kê model + version + stage + mAP50
curl http://localhost:8500/models

# Version đang Production của một model (kèm model_uri để serve)
curl http://localhost:8500/models/cv-detector/production

# Promote (có quality gate + tự reload BentoML)
curl -X POST http://localhost:8500/models/cv-detector/promote \
  -H "Content-Type: application/json" \
  -d '{"version":"8","stage":"Production","min_map50":0.90}'

# Liệt kê run + metrics (thay cho UI tracking)
curl http://localhost:8500/runs
```

Tài liệu API tương tác (Swagger) tại http://localhost:8500/docs.

Ngoài API, `model-api` còn phục vụ một **giao diện riêng** (thay cho UI MLflow) tại http://localhost:8500/ — hoặc qua gateway tại `/models/`. Giao diện này (tiếng Việt, phong cách Apple) liệt kê model với từng version, thanh mAP50, badge stage, nút promote lên Production (có quality gate) và nút lấy `model_uri`, cùng danh sách experiment gần đây. Toàn bộ gọi chính các endpoint REST ở trên, nên không cần MLflow UI.

`train.py` và `service.py` vẫn ghi/đọc trực tiếp MLflow nội bộ (qua mạng Docker `http://mlflow:5000`) — không đổi. Chỉ các hệ thống bên ngoài là chuyển sang gọi `model-api`. Nếu sau này muốn xem lại UI MLflow để debug, thêm tạm `ports: ["5000:5000"]` vào service `mlflow`.

## Giao diện riêng cho MinIO và Keycloak

Hai service nhỏ tự viết cung cấp giao diện đồng nhất (tiếng Việt, phong cách Apple), mỗi cái vừa là REST API vừa phục vụ UI — giống mô hình `model-api`:

- `minio-api` (cổng 8600, qua gateway `/minio/`) — trình duyệt dữ liệu: list bucket, duyệt thư mục, tạo bucket, upload, tải về (presigned URL), xóa. Bọc S3 API qua boto3.
- `kc-admin-api` (cổng 8700, qua gateway `/users/`) — quản lý user: xem, thêm, xóa, reset mật khẩu, gán/gỡ vai trò. Bọc Keycloak Admin REST API.

Console gốc của MinIO và admin Keycloak vẫn còn (cổng 9001 và 8180) để debug khi cần, nhưng sidebar giờ trỏ vào UI tự viết. Như vậy CVAT là service duy nhất còn dùng UI gốc cho người dùng cuối — cố ý giữ vì công cụ gán nhãn quá phức tạp để viết lại.

## Huấn luyện & tạo version dataset trên giao diện

Service `trainer-api` (cổng 8800, qua gateway `/train/`, sidebar "Huấn luyện") cho phép làm trọn trên UI mà không cần gõ lệnh:
- Nút "Huấn luyện model mới" → chạy `train.py` ở thread nền, trả job_id ngay.
- Nút "Tạo phiên bản dataset" → chạy `export_from_cvat.py` (export CVAT + DVC push).
- Danh sách job tự làm mới mỗi 3 giây, hiện tiến độ epoch, trạng thái (chờ/đang chạy/xong/lỗi) và lý do lỗi.

Train chạy ngay trong container `trainer-api`. Để train thật cần GPU: bỏ comment khối `deploy.resources` trong `docker-compose.yml` (cần nvidia-docker). Không có GPU thì vẫn chạy được trên CPU nhưng rất chậm — chỉ hợp smoke-test.

Cơ chế: train là tác vụ lâu nên không chạy đồng bộ trong request HTTP — service tạo job nền và UI poll trạng thái. Job lưu trong bộ nhớ (đủ cho 1 server; production nên chuyển sang Redis/DB để không mất khi restart).

## Model mặc định: YOLO26

Hệ thống dùng YOLO26 (model mới nhất của Ultralytics, ra tháng 1/2026) làm mặc định — `yolo26n.pt`, tự tải lần đầu dùng. YOLO26 có inference NMS-free end-to-end, tối ưu cho edge. Đổi sang model khác (yolo26s/m/l/x, hoặc YOLO11 cũ) chỉ cần đổi tham số `--model` khi train, hoặc ô model trên giao diện Huấn luyện.

## SAM auto-annotation (gán nhãn một-click)

CVAT hỗ trợ Segment Anything Model để gán nhãn bằng AI — click một phát ra mask thay vì vẽ tay. Bật bằng cách chạy stack kèm `serverless/docker-compose.serverless.yml` rồi `bash serverless/deploy-sam.sh`. Chi tiết trong `serverless/README.md`. Đây là cách thu hẹp khoảng cách với smart-annotation của các nền tảng thương mại, vẫn giữ self-host hoàn toàn.

## Chế độ đơn giản cho người mới (wizard)

Trang chủ (`http://localhost`) giờ là một wizard thân thiện cho người không rành AI, thay vì dashboard kỹ thuật. Service `wizard` (cổng 8900) dẫn dắt theo mục tiêu:

- Màn chào hỏi "Bạn muốn làm gì?" với lựa chọn bằng ngôn ngữ thường (dạy máy nhận diện, thử mô hình, quản lý ảnh, chế độ chuyên gia).
- Chọn "dạy máy" → 4 bước đánh số: đưa ảnh → khoanh vùng (SAM một-chạm) → máy học → dùng thử. Mỗi bước tự cập nhật trạng thái từ các API.
- Thuật ngữ kỹ thuật được dịch sang tiếng người: "epoch" ẩn đi (tự suy theo số ảnh), "mAP50" → "độ chính xác", trạng thái train → "Máy đang học, còn khoảng N phút".
- Mặc định thông minh: nút "Bắt đầu dạy máy" chạy với YOLO26n + số epoch tự suy; tùy chỉnh nằm sau "Tùy chọn nâng cao".
- Có nút dùng dữ liệu mẫu để thử ngay.

Wizard không thay thế công cụ cũ — nó nằm trên. Người rành vào "Chế độ chuyên gia" (`/studio/`) là sidebar đầy đủ như trước. Wizard chỉ gọi lại các API có sẵn (trainer-api, minio-api, model-api), không xử lý gì mới.

Giới hạn thành thật: wizard làm hệ thống dễ tiếp cận hơn nhiều, nhưng không biến train thành tức thì — train vẫn cần thời gian và GPU, gán nhãn vẫn cần công người (dù SAM giúp nhanh hơn). Đây là bản chất, không UI nào giấu được.

## Gán nhãn đơn giản (thay CVAT cho người mới)

Service `simple-label` (cổng 8300, qua gateway `/label/`, sidebar "Gán nhãn") hỗ trợ **hai loại bài toán** — người dùng chọn khi tạo dự án:

- **Nhận diện vật thể** (detection): kéo chuột vẽ bounding box, chọn class. Lưu định dạng YOLO chuẩn (`labels/train/*.txt`).
- **Phân loại ảnh** (classification): bấm class lớn để gán cho cả ảnh, có phím tắt 1-4. Lưu dạng thư mục theo class (`train/<class>/`).

Cả hai chế độ ghi thẳng vào MinIO với cấu trúc khớp luôn `train.py` — `--task detect` (mặc định, model `yolo26n.pt`) hoặc `--task classify` (model classification của YOLO). Wizard dẫn người mới đến simple-label thay vì CVAT. CVAT vẫn giữ ở "Gán nhãn nâng cao" cho polygon, keypoint, hoặc khi cần chia task nhiều người.

## Quản lý dataset (đúng nghĩa, không chỉ file)

Service `dataset-api` (cổng 8400, qua gateway `/datasets/`, sidebar "Dataset") quản lý dataset như khái niệm ML, không chỉ liệt kê file: liệt kê các bộ dữ liệu, version (v1/v2/v3), số ảnh train/val, danh sách class, ảnh mẫu với **bounding box từ file nhãn YOLO vẽ đè lên** (preview trực quan để kiểm tra nhãn không cần mở CVAT), và lịch sử train nào dùng version nào kèm mAP đạt được.

Nguồn dữ liệu: gom MinIO (bucket `datasets` theo cấu trúc `<tên>/<version>/data.yaml + images/labels`) và MLflow (tag `dataset` + `dataset_version` trên run). `train.py` tự log hai tag này khi truyền `--dataset` và `--dataset-version`, trainer-api tự truyền qua.

Sidebar tách thành hai mục: "Dataset" (cấu trúc, mục đích ML) và "Kho ảnh" (trình duyệt file MinIO thô — vẫn giữ cho thao tác cấp thấp). Người mới chỉ cần xem "Dataset"; kỹ sư cần đụng file gốc dùng "Kho ảnh".

## Train tiếp từ mô hình có sẵn (fine-tuning)

Hệ thống cho phép nạp sẵn các mô hình nền (file `.pt` của YOLO) để người dùng train tiếp bằng ảnh mới, thay vì luôn train từ đầu. Đặt file `.pt` vào MinIO bucket `models` dưới `base/`, khai báo trong `base-models/base-models.json`. Chi tiết trong `base-models/README.md`.

Khi train, giao diện "Huấn luyện" hiện một hộp chọn cách bắt đầu: hai thẻ "Bắt đầu từ đầu" (train model mới YOLO26) hoặc "Dùng mô hình có sẵn" (train tiếp). Chọn "có sẵn" → hiện danh sách mô hình nền để bấm. `train.py` tự tải file `.pt` đó từ MinIO và fine-tune từ trạng thái đã có. Nếu số class của dữ liệu mới khác mô hình nền, Ultralytics tự khởi tạo lại lớp đầu ra cho khớp, phần backbone vẫn kế thừa.

Lưu ý quan trọng: mô hình nền phải là `.pt` của YOLO/Ultralytics. Mô hình ONNX/TF chỉ chạy inference được, KHÔNG train tiếp được (đã bỏ trạng thái huấn luyện) — nếu chỉ có ONNX, cần tìm lại file `.pt` gốc.

## Lưu ý production

- Đổi toàn bộ mật khẩu trong `.env`.
- Thêm reverse proxy (Traefik/Nginx) + HTTPS trước các service.
- Bật xác thực cho dashboard và BentoML endpoint.
- Backup volume `pg_data` và `minio_data` định kỳ.
- Khi cần GPU/scale: chuyển training sang máy có GPU, hoặc nâng lên Kubernetes (KServe thay BentoML, Kubeflow cho pipeline).
