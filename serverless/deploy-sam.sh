#!/usr/bin/env bash
# =============================================================
#  deploy-sam.sh — Deploy Segment Anything (SAM) vào CVAT qua Nuclio.
#
#  Sau khi chạy, mở CVAT → annotation job → AI Tools > Interactors,
#  chọn "Segment Anything". Click trái = chọn vật thể, click phải = bỏ nền.
#  Một cú click ra mask/bbox thay vì vẽ tay.
#
#  ĐIỀU KIỆN:
#   - Stack đã chạy KÈM serverless/docker-compose.serverless.yml
#   - Có 'nuctl' (Nuclio CLI). Script tự tải nếu thiếu.
#   - SAM trên CPU chạy được nhưng chậm; có GPU thì nhanh hơn nhiều
#     (đổi DEPLOY_MODE=gpu, cần Nvidia Container Toolkit).
#
#  Dùng:  bash serverless/deploy-sam.sh
# =============================================================
set -e

NUCTL_VERSION="${NUCTL_VERSION:-1.13.0}"
DEPLOY_MODE="${DEPLOY_MODE:-cpu}"     # cpu | gpu
CVAT_REF="${CVAT_REF:-v2.20.0}"       # khớp version cvat đang dùng
WORK="${WORK:-/tmp/cvat-sam}"

echo ">> 1/4 Kiểm tra nuctl (Nuclio CLI)..."
if ! command -v nuctl >/dev/null 2>&1; then
  echo "   Chưa có nuctl, đang tải v${NUCTL_VERSION}..."
  ARCH=$(uname -m); [ "$ARCH" = "x86_64" ] && ARCH=amd64
  curl -sLo /tmp/nuctl \
    "https://github.com/nuclio/nuclio/releases/download/${NUCTL_VERSION}/nuctl-${NUCTL_VERSION}-linux-${ARCH}"
  chmod +x /tmp/nuctl
  sudo ln -sf /tmp/nuctl /usr/local/bin/nuctl
fi
nuctl version

echo ">> 2/4 Lấy mã function SAM từ repo CVAT (ref ${CVAT_REF})..."
# Chỉ cần thư mục serverless, dùng sparse checkout cho nhẹ
rm -rf "$WORK"
git clone --depth 1 --branch "$CVAT_REF" --filter=blob:none --sparse \
  https://github.com/cvat-ai/cvat.git "$WORK"
cd "$WORK"
git sparse-checkout set serverless

echo ">> 3/4 Deploy SAM function (mode=${DEPLOY_MODE})..."
SAM_PATH="serverless/pytorch/facebookresearch/sam/nuclio"
if [ "$DEPLOY_MODE" = "gpu" ]; then
  nuctl deploy --project-name cvat \
    --path "$SAM_PATH" \
    --volume "$(pwd)/serverless/common:/opt/nuclio/common" \
    --platform local \
    --resource-limit nvidia.com/gpu=1 \
    --triggers '{"myHttpTrigger": {"maxWorkers": 1}}'
else
  nuctl deploy --project-name cvat \
    --path "$SAM_PATH" \
    --volume "$(pwd)/serverless/common:/opt/nuclio/common" \
    --platform local
fi

echo ">> 4/4 Kiểm tra function đã chạy:"
nuctl get function --platform local

echo ""
echo "============================================="
echo "  SAM đã deploy. Mở CVAT → annotation job →"
echo "  AI Tools > Interactors → chọn 'Segment Anything'."
echo "  Click trái: chọn vật thể · Click phải: bỏ nền."
echo "============================================="
