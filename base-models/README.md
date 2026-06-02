# Mô hình nền nạp sẵn (fine-tuning)

Cho phép người dùng train tiếp (fine-tune) từ 2 mô hình bạn đã chuẩn bị, thay vì train từ đầu. Người dùng chọn một mô hình nền, thêm ảnh mới, hệ thống học tiếp từ trạng thái đã có.

## Yêu cầu quan trọng

Mô hình nền phải là **file `.pt` của YOLO/Ultralytics** (không phải ONNX/TF). ONNX chỉ chạy được inference, không train tiếp được — nếu bạn chỉ có ONNX, hãy tìm lại file `.pt` gốc lúc train ra nó.

## Cách nạp 2 mô hình của bạn

1. Đặt 2 file `.pt` gốc vào MinIO, bucket `models`, đúng đường dẫn ghi trong `base-models.json`:
   - `base/object-detector.pt`
   - `base/defect-classifier.pt`

   Có thể upload qua giao diện "Dữ liệu" (chọn bucket models, tạo thư mục base, tải lên), hoặc bằng lệnh:
   ```bash
   mc cp object-detector.pt myminio/models/base/object-detector.pt
   mc cp defect-classifier.pt myminio/models/base/defect-classifier.pt
   ```

2. Chỉnh `base-models.json` nếu cần (đổi tên hiển thị, mô tả, hoặc thêm mô hình nền khác).

3. Khởi động lại `trainer-api` để nó nạp danh sách mới.

## Người dùng dùng thế nào

- Trên giao diện "Huấn luyện": có ô "Mô hình nền" — chọn một trong các nền đã nạp, hoặc "Bắt đầu từ đầu (YOLO26)".
- Trên wizard (chế độ đơn giản): khi chọn "Dạy máy", có thể chọn "Dạy tiếp mô hình có sẵn" với tên dễ hiểu.
- Chọn nền → thêm ảnh đã gán nhãn → bấm train. Hệ thống fine-tune từ nền đó.

## Lưu ý kỹ thuật

- Fine-tune giữ lại kiến thức cũ và học thêm cái mới. Cần ít ảnh hơn train từ đầu, hội tụ nhanh hơn.
- Nếu thêm loại vật/lỗi mới (số class khác mô hình nền), Ultralytics tự khởi tạo lại lớp đầu ra (detection head) cho khớp số class mới — phần backbone vẫn kế thừa.
- Số class và định dạng nhãn của dữ liệu mới phải khai trong `data.yaml` như bình thường.
