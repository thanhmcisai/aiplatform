#!/usr/bin/env bash
# =============================================================
#  smoke-test.sh — Kiểm tra từng service đã lên và phản hồi đúng.
#  Chạy SAU khi `docker compose up -d` và đợi ~1-2 phút.
#
#  Dùng: bash scripts/smoke-test.sh
# =============================================================
set -u

PASS=0
FAIL=0

check() {
  local name="$1" url="$2" expect="${3:-200}"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$url" 2>/dev/null)
  if [ "$code" = "$expect" ] || { [ "$expect" = "2xx" ] && [ "${code:0:1}" = "2" ]; }; then
    printf "  [OK]   %-16s %s (%s)\n" "$name" "$url" "$code"
    PASS=$((PASS+1))
  else
    printf "  [FAIL] %-16s %s (nhận %s, mong %s)\n" "$name" "$url" "$code" "$expect"
    FAIL=$((FAIL+1))
  fi
}

echo "== Kiểm tra service nội bộ (cổng trực tiếp) =="
check "dashboard"    "http://localhost:8001/health"
check "model-api"    "http://localhost:8500/health"
check "minio-api"    "http://localhost:8600/health"
check "kc-admin-api" "http://localhost:8700/health"
check "minio"        "http://localhost:9000/minio/health/live"
check "cvat-ui"      "http://localhost:8080/" "2xx"
check "keycloak"     "http://localhost:8180/auth/realms/mlops" "2xx"

echo ""
echo "== Kiểm tra qua gateway (cổng 8090) =="
echo "  (các route dưới gateway cần đăng nhập SSO; 302 = chuyển tới login là BÌNH THƯỜNG)"
check "shell"        "http://localhost:8090/" "302"
check "gw dashboard" "http://localhost:8090/dashboard/health" "302"

echo ""
echo "== Kiểm tra API model-api có gọi được MLflow không =="
MODELS=$(curl -s --max-time 8 http://localhost:8500/models 2>/dev/null)
if echo "$MODELS" | grep -q '"models"'; then
  echo "  [OK]   model-api → MLflow phản hồi: $MODELS"
  PASS=$((PASS+1))
else
  echo "  [FAIL] model-api không lấy được dữ liệu MLflow"
  echo "         (nếu registry trống, vẫn nên thấy {\"models\":[]})"
  FAIL=$((FAIL+1))
fi

echo ""
echo "== Kiểm tra minio-api liệt kê được bucket =="
BUCKETS=$(curl -s --max-time 8 http://localhost:8600/buckets 2>/dev/null)
if echo "$BUCKETS" | grep -q '"buckets"'; then
  echo "  [OK]   minio-api → bucket: $BUCKETS"
  PASS=$((PASS+1))
else
  echo "  [FAIL] minio-api không liệt kê được bucket"
  FAIL=$((FAIL+1))
fi

echo ""
echo "== Kiểm tra Nuclio (chỉ khi đã bật serverless cho SAM) =="
NUCLIO=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8070/ 2>/dev/null)
if [ "$NUCLIO" = "200" ]; then
  echo "  [OK]   Nuclio dashboard chạy (SAM auto-annotation khả dụng)"
else
  echo "  [--]   Nuclio chưa chạy (bình thường nếu không bật serverless)"
fi

echo ""
echo "============================================="
echo "  Kết quả: $PASS đạt, $FAIL lỗi"
if [ "$FAIL" -gt 0 ]; then
  echo "  Gợi ý: service nặng (CVAT, Keycloak) cần thêm thời gian khởi động."
  echo "  Đợi thêm 1-2 phút rồi chạy lại. Xem log: docker compose logs <service>"
  exit 1
fi
echo "  Tất cả service sẵn sàng. Có thể bắt đầu luồng dataset → train."
