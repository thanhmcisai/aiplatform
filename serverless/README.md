# SAM auto-annotation cho CVAT

Tích hợp Segment Anything Model (SAM) để gán nhãn một-click trong CVAT — thay vì vẽ tay từng bounding box/polygon, click một phát ra mask. Đây là điểm mạnh nhất của các nền tảng thương mại như Ultralytics Platform, và CVAT hỗ trợ sẵn qua Nuclio.

## Cơ chế

SAM chạy như một "serverless function" do Nuclio quản lý. CVAT gọi function này qua HTTP mỗi khi bạn click trên ảnh. Kiến trúc:

```
CVAT (annotation UI)  ──HTTP──>  Nuclio  ──>  SAM function (container)
        click chuột                              trả về mask/bbox
```

## Cài đặt

### 1. Chạy stack kèm Nuclio

```bash
docker compose \
  -f docker-compose.yml \
  -f serverless/docker-compose.serverless.yml \
  up -d
```

### 2. Deploy SAM function

```bash
bash serverless/deploy-sam.sh
```

Script tự: tải `nuctl` (Nuclio CLI) nếu thiếu, lấy mã SAM từ repo CVAT, deploy function. Lần đầu sẽ lâu vì build container chứa model SAM (vài GB).

Chạy trên GPU (nhanh hơn nhiều) nếu có Nvidia Container Toolkit:
```bash
DEPLOY_MODE=gpu bash serverless/deploy-sam.sh
```

### 3. Dùng trong CVAT

1. Mở một annotation job.
2. Vào `AI Tools > Interactors`, chọn `Segment Anything` trong danh sách.
3. Click trái lên vật thể cần gán nhãn → SAM tạo mask. Click phải để loại vùng nền thừa.
4. Lưu job, export như bình thường.

## Lưu ý thực tế

- **CPU chạy được nhưng chậm** — mỗi click mất vài giây. Có GPU thì gần như tức thì. Cho dùng thật nên có GPU.
- `nuctl` phải khớp version Nuclio dashboard (mặc định 1.13.0). Nếu deploy lỗi, kiểm tra version trong `docker-compose.serverless.yml`.
- `CVAT_REF` trong script phải khớp version CVAT đang chạy (mặc định `v2.20.0`, khớp image trong compose chính). Nếu đổi version CVAT, sửa biến này.
- SAM cho ra **mask** (segmentation). Nếu chỉ cần bounding box, CVAT tự chuyển mask thành box khi export YOLO detection.
- Mô hình nặng: function SAM container vài GB, cần đủ dung lượng đĩa và RAM.

## Tùy chọn nâng cao

- Có thể deploy thêm các function khác (YOLO pre-label tự động cả ảnh, không cần click) theo cùng cách — xem `serverless/` trong repo CVAT.
- SAM2 có chế độ tracking cho video, nhưng cần Redis (CVAT đã có sẵn `cvat-redis` trong stack) và cấu hình thêm.
