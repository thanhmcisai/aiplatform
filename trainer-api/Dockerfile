FROM python:3.11-slim
WORKDIR /app
# OpenCV (do ultralytics kéo về) cần thư viện hệ thống GUI/X; nếu thiếu sẽ lỗi
# "libxcb.so.1 / libGL.so.1 cannot open shared object file" khi import cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
# Thư viện train nặng (ultralytics kéo torch). Build sẽ lâu.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY static ./static
EXPOSE 8800
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]
