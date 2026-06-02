/* =============================================================
   cvat-vi.js — Lớp dịch tiếng Việt cho CVAT (best-effort).

   CVAT chưa có gói tiếng Việt chính thức, nên ta dịch runtime
   bằng cách thay text các nhãn phổ biến nhất trên DOM.

   LƯU Ý: đây là giải pháp "best-effort". Mỗi lần CVAT cập nhật
   version, nhãn có thể đổi và cần bổ sung từ điển bên dưới.
   Không bao phủ 100% (tooltip động, thông báo lỗi runtime...).
   ============================================================= */
(function () {
  // Từ điển: text gốc (tiếng Anh) -> tiếng Việt
  const DICT = {
    "Projects": "Dự án",
    "Tasks": "Công việc",
    "Jobs": "Phiên gán nhãn",
    "Cloud Storages": "Lưu trữ đám mây",
    "Models": "Mô hình",
    "Create a new task": "Tạo công việc mới",
    "Create a new project": "Tạo dự án mới",
    "Create": "Tạo",
    "Submit": "Gửi",
    "Cancel": "Hủy",
    "Save": "Lưu",
    "Delete": "Xóa",
    "Open": "Mở",
    "Export": "Xuất",
    "Import": "Nhập",
    "Name": "Tên",
    "Labels": "Nhãn",
    "Add label": "Thêm nhãn",
    "Select files": "Chọn tệp",
    "Upload": "Tải lên",
    "Search": "Tìm kiếm",
    "Filter": "Lọc",
    "Sort by": "Sắp xếp theo",
    "Owner": "Chủ sở hữu",
    "Status": "Trạng thái",
    "Completed": "Hoàn thành",
    "In progress": "Đang thực hiện",
    "Annotation": "Gán nhãn",
    "Validation": "Kiểm định",
    "Save annotations": "Lưu nhãn",
    "Logout": "Đăng xuất",
    "Settings": "Cài đặt",
    "About": "Giới thiệu",
  };

  function translateNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const t = node.textContent.trim();
      if (DICT[t]) node.textContent = node.textContent.replace(t, DICT[t]);
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      // dịch placeholder và title
      if (node.placeholder && DICT[node.placeholder.trim()]) {
        node.placeholder = DICT[node.placeholder.trim()];
      }
      if (node.title && DICT[node.title.trim()]) {
        node.title = DICT[node.title.trim()];
      }
      node.childNodes.forEach(translateNode);
    }
  }

  function run() { translateNode(document.body); }

  // Dịch lần đầu + theo dõi DOM thay đổi (CVAT là SPA)
  if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);

  const obs = new MutationObserver((muts) => {
    for (const m of muts) m.addedNodes.forEach(translateNode);
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();
