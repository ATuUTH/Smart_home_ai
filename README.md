# ⌂ Smart Home AI Hub (Hybrid Architecture & Hardware Automation)

Dự án xây dựng hệ thống quản lý nhà thông minh toàn diện phối hợp giữa tầng **Trí tuệ nhân tạo (Vision AI & Machine Learning)** trên Python và tầng **Điều khiển ngoại vi (Hardware Simulation)** trên Proteus thông qua giao tiếp nối tiếp UART. 

Hệ thống ứng dụng kiến trúc lai (Hybrid Model): Sử dụng Học sâu (Deep Learning) của Google MediaPipe để trích xuất đặc trưng hình học thời gian thực, kết hợp với các mô hình Học máy truyền thống (Scikit-Learn) để phân loại danh tính và dự đoán ngữ cảnh môi trường nhằm tối ưu hóa hiệu năng trên các thiết bị Edge AI hoặc máy tính cấu hình tầm trung.

---

## 🛠️ CÁC TÍNH NĂNG CỐT LÕI (CORE FEATURES)

### 1. Xác thực Face ID An ninh & Thử thách Liveness (Anti-Spoofing)

* **Trích xuất đặc trưng hình học:** Bắt luồng camera thời gian thực qua OpenCV, định vị ma trận 468 điểm mốc không gian 3D của khuôn mặt qua **MediaPipe Face Mesh**. Thuật toán tự động cô lập 10 điểm mốc cốt lõi (Core Indices) để tính toán ma trận khoảng cách Euclid tương đối, được chuẩn hóa qua khoảng cách giữa hai mắt (`base_dist`) nhằm triệt tiêu hoàn toàn sai số khi người dùng đứng xa/gần camera.

* **Phân loại danh tính:** Sử dụng mô hình **SVC (Support Vector Classifier)** đã được hiệu chuẩn xác suất (CalibratedClassifierCV) để nhận diện chính xác từng thành viên với ngưỡng tin cậy tối thiểu **70%**.

* **Chống giả mạo (Anti-Spoofing):** Hệ thống tích hợp thuật toán kiểm tra thực thể sống thông qua mã lệnh thử thách ngẫu nhiên (Nháy mắt liên tục - tỷ lệ EAR, Quay mặt sang trái/phải - tỷ lệ biến thiên trục ngang, Há miệng to). Người dùng bắt buộc phải vượt qua thử thách để chứng minh là người thật, loại bỏ hoàn toàn nguy cơ xâm nhập bằng ảnh chụp hoặc video giả mạo.

### 2. Bộ Dự Đoán Điều Hòa Thông Minh Cá Nhân Hóa (HabitEngine AI)

* **Thu thập thói quen nền:** Tự động thu thập thói quen điều chỉnh nhiệt độ điều hòa của từng chủ nhà cụ thể dựa trên ma trận ngữ cảnh gồm: Khung giờ trong ngày, Nhiệt độ môi trường ngoài trời (giả lập biến thiên), Nhiệt độ trong phòng thực tế (đọc trực tiếp từ cảm biến **LM35** qua chân Analog A0).

* **Thuật toán Hồi quy hỗn hợp (Ensemble Regressor):**
  * *Dưới 15 mẫu dữ liệu:* Sử dụng thuật toán hồi quy lân cận **KNeighborsRegressor** (với k = 3, tính trọng số theo khoảng cách).
  * *Từ 15 mẫu trở lên:* Kích hoạt mô hình hỗn hợp Ensemble phối hợp giữa **KNN Regressor (70%)** and **Linear Regression (30%)** để tính toán mức nhiệt độ tối ưu nhất cho chủ nhà, tự động gửi lệnh điều khiển xuống mạch ngoại vi mà không cần thao tác bấm nút vật lý.

### 3. Tương Tác Cử Chỉ Không Tiếp Xúc (Contactless Hand Gesture)

* Ứng dụng mô hình **MediaPipe Hands** để số hóa cấu trúc xương bàn tay.

* **Bộ lọc nhiễu chuyển động (Motion Buffer):** Sử dụng hàng đợi kép `deque(maxlen=15)` để lưu vết chuyển động ngón tay liên tục. Tích hợp thuật toán giám sát trạng thái ổn định (`stable_frames >= 10`), yêu cầu người dùng giữ nguyên cử chỉ tối thiểu 10 khung hình để tránh việc nhận diện sai khi tay di chuyển vô tình qua camera.

* **Bộ lệnh cử chỉ hỗ trợ:** Giơ 1 ngón (Bật/Tắt Đèn), Giơ 2 ngón (Bật/Tắt Quạt), Vuốt phải (Mở rèm), Vuốt trái (Đóng rèm), Nắm tay (Tắt toàn bộ hệ thống).

### 4. Điều Phối Máy Trạng Thái Khép Kín (Closed-loop State Machine)

Hệ thống vận hành theo một vòng đời tự động nghiêm ngặt để tiết kiệm tài nguyên CPU/GPU và bảo mật tối đa:
`Chờ quét mặt Face ID (Khoá) ➔ Xác thực thành công ➔ Khởi tạo luồng song song (Thread) cấu hình AC cá nhân hóa ➔ Mở phiên tương tác cử chỉ tay (Hub Panel) ➔ Đếm ngược thời gian chờ (Idle Timeout 400 frames) ➔ Tự giải phóng Camera & Tái khóa hệ thống về Giai đoạn 1.`

---

## 📁 CẤU TRÚC THƯ MỤC PROJECT

```text
├── door_gui.py             # Giao diện màn hình an ninh quét Face ID & Chống giả mạo
├── smart_hub_gui.py        # Giao diện trung tâm điều khiển Smart Hub, lõi quản lý cử chỉ và HabitEngine AI
├── ai_mat.py               # Lõi thuật toán xử lý Vision AI (Đăng ký khuôn mặt, Huấn luyện mô hình SVC)
├── smart_home_firmware.ino # Mã nguồn C++ nạp cho vi điều khiển Arduino Uno R3 trên Proteus
├── current_owner.txt       # Tệp tin tạm lưu danh tính chủ nhà vừa đăng nhập để liên thông dữ liệu
├── ac_habit_log.csv        # Cơ sở dữ liệu thói quen sử dụng điều hòa của các thành viên
├── AI_Evaluation_Report.txt# Báo cáo đánh giá hàn lâm (Accuracy, Precision, Recall) tự động xuất sau khi train AI
└── face_ocsvm_model.pkl    # File lưu trữ mô hình SVC nhận diện khuôn mặt sau khi huấn luyện xong
