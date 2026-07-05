"""
face_ai_patch.py
================
Patch tự động vào face_ai_pro.py:
  1. Ghi current_owner.txt trước sys.exit(0)
  2. Xoá menu CLI (chỉ giữ hàm run_smart_home để door_gui gọi)
  3. Bỏ phần gọi pygame / âm thanh ra module riêng (tuỳ chọn)

Chạy 1 lần: python face_ai_patch.py
"""

import re, shutil, os

SRC = "face_ai_pro.py"
BAK = "face_ai_pro.py.bak"

if not os.path.exists(SRC):
    print(f"[PATCH] Không tìm thấy {SRC}. Đặt file này cùng thư mục rồi chạy lại.")
    exit(1)

shutil.copy(SRC, BAK)
print(f"[PATCH] Đã sao lưu → {BAK}")

code = open(SRC, encoding="utf-8").read()

# ── PATCH 1: thêm ghi current_owner.txt trước sys.exit(0) ──
# Tìm đoạn:  cap.release()\n    cv2.destroyAllWindows()\n\n    import sys\n    sys.exit(0)
OLD = (
    "cap.release()\n"
    "        cv2.destroyAllWindows()\n"
    "\n"
    "                    import sys\n"
    "                    sys.exit(0)"
)
NEW = (
    "cap.release()\n"
    "        cv2.destroyAllWindows()\n"
    "\n"
    "                    # Ghi tên chủ nhà để door_gui / master đọc\n"
    "                    with open('current_owner.txt', 'w', encoding='utf-8') as _owf:\n"
    "                        _owf.write(owner_name)\n"
    "\n"
    "                    import sys\n"
    "                    sys.exit(0)"
)

if OLD in code:
    code = code.replace(OLD, NEW)
    print("[PATCH] ✓ Đã thêm ghi current_owner.txt")
else:
    # Fallback: tìm kiểu khác (indentation khác nhau)
    pattern = r'(cap\.release\(\)\s+cv2\.destroyAllWindows\(\))\s+(import sys\s+sys\.exit\(0\))'
    replacement = (
        r'\1\n\n                    # Ghi tên chủ nhà để door_gui / master đọc\n'
        r"                    with open('current_owner.txt', 'w', encoding='utf-8') as _owf:\n"
        r'                        _owf.write(owner_name)\n\n'
        r'                    \2'
    )
    new_code = re.sub(pattern, replacement, code)
    if new_code != code:
        code = new_code
        print("[PATCH] ✓ Đã thêm ghi current_owner.txt (regex fallback)")
    else:
        print("[PATCH] ⚠ Không tìm thấy đoạn sys.exit(0) — tìm thủ công:")
        print("        Thêm 3 dòng này TRƯỚC dòng 'import sys' trong secure_stage==2:")
        print("            with open('current_owner.txt', 'w', encoding='utf-8') as _owf:")
        print("                _owf.write(owner_name)")

open(SRC, "w", encoding="utf-8").write(code)
print(f"[PATCH] Đã lưu {SRC}")
print("\n[HƯỚNG DẪN] Nếu patch tự động không khớp, mở face_ai_pro.py")
print("tìm đoạn 'sys.exit(0)' trong khối 'elif secure_stage == 2:'")
print("và thêm thủ công:")
print("""
    with open('current_owner.txt', 'w', encoding='utf-8') as _owf:
        _owf.write(owner_name)
    import sys
    sys.exit(0)
""")