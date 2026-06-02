# serving/Dockerfile — BentoML serving (CVDetector) phục vụ route /api.
#
# Image này NẶNG vì kéo theo torch + ultralytics. Để nhẹ và không cần GPU,
# cài torch BẢN CPU trước (tránh wheel CUDA vài GB), rồi mới tới ultralytics.
# service.py nạp model Production từ MLflow Registry (artifact trên MinIO).
FROM python:3.11-slim

WORKDIR /app

# libGL/libglib: phụ thuộc OpenCV của ultralytics. Chỉ cài tối thiểu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# torch CPU trước (index riêng), rồi các gói còn lại.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir \
        "bentoml>=1.2.0" \
        "ultralytics>=8.3.0" \
        "mlflow>=2.16.0" \
        "boto3>=1.34.0" \
        "pillow>=10.0.0"

COPY service.py .

EXPOSE 3000
# Khớp Caddyfile: gateway proxy /api/* → bentoml:3000
CMD ["bentoml", "serve", "service:CVDetector", "--host", "0.0.0.0", "--port", "3000"]
