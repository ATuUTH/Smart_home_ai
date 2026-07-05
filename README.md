# ⌂ Smart Home AI Hub (Hybrid Architecture & Hardware Automation)

Dự án xây dựng hệ thống quản lý nhà thông minh toàn diện, phối hợp giữa tầng **Trí tuệ nhân tạo** (Vision AI & Machine Learning) trên Python và tầng **Điều khiển ngoại vi** (Hardware Simulation) trên Proteus thông qua giao tiếp nối tiếp UART.

Hệ thống ứng dụng **kiến trúc lai (Hybrid Model)**: sử dụng Học sâu (Deep Learning) của Google MediaPipe để trích xuất đặc trưng hình học thời gian thực, kết hợp với các mô hình Học máy truyền thống (Scikit-Learn) để phân loại danh tính và dự đoán ngữ cảnh môi trường, nhằm tối ưu hóa hiệu năng trên các thiết bị Edge AI hoặc máy tính cấu hình tầm trung.

---

## 🛠️ CÁC TÍNH NĂNG CỐT LÕI (CORE FEATURES)

### 1. Xác thực Face ID An ninh & Thử thách Liveness (Anti-Spoofing)

- **Trích xuất đặc trưng hình học:** Bắt luồng camera thời gian thực qua OpenCV, định vị ma trận 468 điểm mốc không gian 3D của khuôn mặt qua MediaPipe Face Mesh. Thuật toán tự động cô lập 10 điểm mốc cốt lõi (Core Indices) để tính toán ma trận khoảng cách Euclid tương đối, được chuẩn hóa qua khoảng cách giữa hai mắt (`base_dist`) nhằm triệt tiêu hoàn toàn sai số khi người dùng đứng xa/gần camera.
- **Phân loại danh tính:** Sử dụng mô hình SVC (Support Vector Classifier) đã được hiệu chuẩn xác suất (`CalibratedClassifierCV`) để nhận diện chính xác từng thành viên với ngưỡng tin cậy tối thiểu **70%**.
- **Chống giả mạo (Anti-Spoofing):** Hệ thống tích hợp thuật toán kiểm tra thực thể sống thông qua mã lệnh thử thách ngẫu nhiên (nháy mắt liên tục - tỷ lệ EAR, quay mặt sang trái/phải - tỷ lệ biến thiên trục ngang, há miệng to). Người dùng bắt buộc phải vượt qua thử thách để chứng minh là người thật, loại bỏ hoàn toàn nguy cơ xâm nhập bằng ảnh chụp hoặc video giả mạo.

### 2. Bộ Dự Đoán Điều Hòa Thông Minh Cá Nhân Hóa (HabitEngine AI)

- **Thu thập thói quen nền:** Tự động thu thập thói quen điều chỉnh nhiệt độ điều hòa của từng chủ nhà cụ thể dựa trên ma trận ngữ cảnh gồm: khung giờ trong ngày, nhiệt độ môi trường ngoài trời (giả lập biến thiên), nhiệt độ trong phòng thực tế (đọc trực tiếp từ cảm biến LM35 qua chân Analog A0).
- **Thuật toán Hồi quy hỗn hợp (Ensemble Regressor):**
  - Dưới 15 mẫu dữ liệu: sử dụng thuật toán hồi quy lân cận `KNeighborsRegressor` (k = 3, tính trọng số theo khoảng cách).
  - Từ 15 mẫu trở lên: kích hoạt mô hình hỗn hợp Ensemble phối hợp giữa KNN Regressor (70%) và Linear Regression (30%) để tính toán mức nhiệt độ tối ưu nhất cho chủ nhà, tự động gửi lệnh điều khiển xuống mạch ngoại vi mà không cần thao tác bấm nút vật lý.

### 3. Tương Tác Cử Chỉ Không Tiếp Xúc (Contactless Hand Gesture)

- Ứng dụng mô hình MediaPipe Hands để số hóa cấu trúc xương bàn tay.
- **Bộ lọc nhiễu chuyển động (Motion Buffer):** sử dụng hàng đợi kép `deque(maxlen=15)` để lưu vết chuyển động ngón tay liên tục. Tích hợp thuật toán giám sát trạng thái ổn định (`stable_frames >= 10`), yêu cầu người dùng giữ nguyên cử chỉ tối thiểu 10 khung hình để tránh việc nhận diện sai khi tay di chuyển vô tình qua camera.
- **Hỗ trợ bộ lệnh cử chỉ:**
  - Giơ 1 ngón → Bật/Tắt Đèn
  - Giơ 2 ngón → Bật/Tắt Quạt
  - Vuốt phải → Mở rèm
  - Vuốt trái → Đóng rèm
  - Nắm tay → Tắt toàn bộ hệ thống

### 4. Điều Phối Máy Trạng Thái Khép Kín (Closed-loop State Machine)

Hệ thống vận hành theo một vòng đời tự động nghiêm ngặt để tiết kiệm tài nguyên CPU/GPU và bảo mật tối đa:

> Chờ quét mặt Face ID (Khóa) → Xác thực thành công → Khởi tạo luồng song song (Thread) cấu hình AC cá nhân hóa → Mở phiên tương tác cử chỉ tay (Hub Panel) → Đếm ngược thời gian chờ (Idle Timeout 400 frames) → Tự giải phóng Camera & Tái khóa hệ thống về Giai đoạn 1.

---

## 📁 CẤU TRÚC THƯ MỤC PROJECT

```text
├── ac_habit_log.csv        # Cơ sở dữ liệu thói quen điều chỉnh nhiệt độ của các thành viên
├── AI_Evaluation_Report.txt# Báo cáo đánh giá thuật toán (Accuracy, Precision, Recall) tự động xuất
├── bao_dong.mp3            # Âm thanh còi hú báo động đỏ khi phát hiện người lạ (Unknown)
├── chao_sep.mp3            # Âm thanh chào mừng chủ nhà khi vượt qua bộ lọc Anti-Spoofing thành công
├── current_owner.txt       # Tệp tin tạm lưu danh tính chủ nhà vừa đăng nhập để đồng bộ sang Hub
├── dang_quet.mp3           # Âm thanh nhắc nhở người dùng đứng yên để hệ thống bắt trắc học mặt
├── dataset_anhtu.csv       # Tệp dữ liệu vector trích xuất các điểm mốc khuôn mặt của Anh Tú
├── dataset_phat.csv        # Tệp dữ liệu vector trích xuất các điểm mốc khuôn mặt của Phát
├── door_gui.py             # Giao diện an ninh quét Face ID & Thử thách chống giả mạo ngoài cửa
├── face_ai_patch.py        # Module bản vá/bổ trợ cho luồng xử lý nhận diện khuôn mặt
├── face_ai_pro.py          # Lõi thuật toán Vision AI chính (Thu thập e-KYC, trích xuất hình học và SVC)
├── face_ai_pro.py.bak      # Bản sao lưu dự phòng an toàn của file lõi nhận diện khuôn mặt
├── face_ocsvm_model.pkl    # File bộ não lưu mô hình SVC nhận diện khuôn mặt sau khi huấn luyện xong
├── face_owner_data.pkl     # File lưu bộ đệm siêu dữ liệu e-KYC thô trước khi đưa vào phân lớp
├── lai_gan.mp3             # Âm thanh nhắc nhở người dùng tiến lại gần camera để đạt độ to hình học
├── smart_hub_gui.py        # Giao diện trung tâm Smart Hub điều khiển thiết bị & lõi HabitEngine AI
└── test_hand.py            # Module xử lý camera cử chỉ tay không tiếp xúc và số hóa lệnh UART
```

---

## 🔌 SƠ ĐỒ KẾT NỐI PHẦN CỨNG (PROTEUS SIMULATION)

| Tên Linh Kiện | Mã Linh Kiện Proteus | Chân Arduino Uno R3 | Vai Trò Trong Hệ Thống |
|---|---|---|---|
| Vi điều khiển | ARDUINO UNO R3 | - | Khối xử lý trung tâm, quản lý ngắt Serial và thiết bị ngoại vi |
| Cổng kết nối ảo | COMPIM | Pin 0 (RX), Pin 1 (TX) | Cầu nối UART nhận lệnh mã hóa chuỗi trực tiếp từ mã nguồn Python |
| Cảm biến nhiệt độ | LM35 | Pin Analog A0 | Đo nhiệt độ phòng thực tế (xuất điện áp tuyến tính 10mV/°C, không bị lỗi lag kẹt số NaN như các dòng cảm biến kỹ thuật số trong môi trường mô phỏng nặng) |
| Động cơ khóa rèm | MOTOR-SERVO | Pin 10 (Xuất PWM) | Giả lập cơ chế chốt khóa cửa sổ / đóng mở rèm (0° là Đóng, 90° là Mở) |
| Cơ cấu làm mát | MOTOR (DC Motor) | Pin 9 (Qua Transistor) | Giả lập hệ thống quạt thông gió hoặc Block máy lạnh |
| Mạch kích dòng | NPN Transistor (2N2222) | Cực B nối Pin 9 qua trở 1k | Mạch đệm dòng điện, bảo vệ chân Arduino khỏi dòng tải cao của động cơ |
| Diode bảo vệ | DIODE (1N4007) | Mắc song song ngược với Motor | Diode dập xung ngược (Flyback Diode) triệt tiêu dòng điện cảm ứng |
| Hệ thống đèn | LED-RED, LED-GREEN | Pin 8, Pin A1, Pin A2 | Hệ thống đèn chiếu sáng và đèn LED báo trạng thái an ninh (Red: Khóa, Green: Mở) |
| Màn hình hiển thị | LM016L (LCD 16x2) | Giao tiếp qua đường dây I2C | Hiển thị trạng thái thời gian thực, thông điệp chào mừng và số độ AC |

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & TRIỂN KHAI

### Bước 1: Khởi tạo môi trường Python

Yêu cầu Python phiên bản từ 3.8 đến 3.11. Cài đặt các thư viện lõi bằng dòng lệnh sau:

```bash
pip install opencv-python mediapipe numpy scikit-learn pillow pyserial pygame
```

### Bước 2: Cấu hình cổng COM ảo trên máy tính

1. Mở phần mềm tạo cổng COM ảo (Virtual Serial Port Driver - VSPD hoặc VSPM).
2. Tạo một cặp cổng COM ảo liên thông với nhau: `COM3` ➔ `COM4`.
3. Trong mã nguồn Python, cổng kết nối được cấu hình mặc định là `COM3`. Trong phần mềm Proteus, click đúp vào linh kiện COMPIM và cấu hình cổng kết nối là `COM4`, tốc độ Baudrate đặt chính xác mức `9600`.

### Bước 3: Biên dịch và nạp Firmware cho Arduino

1. Mở file `smart_home_firmware.ino` bằng Arduino IDE.
2. Nhấn tổ hợp phím `Ctrl + Alt + S` (hoặc chọn Sketch ➔ Export Compiled Binary) để phần mềm biên dịch ra file nhị phân `.hex`.
3. Sang phần mềm Proteus, click đúp chuột vào board mạch Arduino Uno R3. Tại ô Program File, bấm vào biểu tượng thư mục màu vàng và trỏ đường dẫn tới file `.hex` vừa được xuất ra.

---

## 🎮 CÁCH VẬN HÀNH HỆ THỐNG TRỰC TIẾP

### 1. Đăng ký thành viên & Huấn luyện AI Khuôn mặt (qua giao diện Smart Hub)

Từ phiên bản hiện tại, toàn bộ quy trình quét đăng ký khuôn mặt và huấn luyện lại mô hình **không còn chạy qua menu dòng lệnh của `ai_mat.py`** nữa, mà đã được tích hợp gọn vào khối **"Thiết lập AI"** ở cột bên trái giao diện `smart_hub_gui.py`, gồm ô nhập tên và 2 bước thao tác trực quan.

**Quy trình thao tác:**

1. Khởi chạy trực tiếp Hub ở chế độ quản trị (chưa cần đăng nhập Face ID) bằng lệnh: `python smart_hub_gui.py`.
2. Tại ô **"Tên"**, nhập tên thành viên muốn đăng ký (ví dụ: `anhtu`).
3. Nhấn bước **① Đăng ký khuôn mặt**: hệ thống tự động tắt camera cử chỉ tay (nếu đang bật) để giải phóng thiết bị, chờ 1.5 giây rồi gọi hàm `auto_collect_owner_data()` từ module AI khuôn mặt (ưu tiên `ai_mat.py` nếu có, mặc định dùng `face_ai_pro.py`) để bật camera và tự động thu thập đặc trưng khuôn mặt của thành viên vừa nhập tên, lưu ra file `dataset_<tên>.csv`.
4. Nhấn bước **② Lưu và cập nhật trạng thái**: chạy nền hàm `train_svc_multi_class()`, huấn luyện lại mô hình SVC trên toàn bộ dữ liệu hiện có, xuất ra file "não bộ" `face_ocsvm_model.pkl` kèm báo cáo `AI_Evaluation_Report.txt` (Precision, Recall...).
5. Danh sách **Thành viên** ở đầu cột trái sẽ tự động cập nhật theo các file `dataset_<tên>.csv` đang tồn tại trong thư mục dự án; mỗi thành viên có nút **✕** để xoá dữ liệu khuôn mặt khi cần (hệ thống sẽ nhắc train lại model sau khi xoá).
6. Toàn bộ trạng thái (đang chạy / thành công / lỗi) được hiển thị trực tiếp trên khung **Activity Log** phía dưới màn hình Hub, không cần theo dõi qua cửa sổ Console/Terminal nữa.

> 💡 Muốn kiểm tra riêng màn hình an ninh cửa mà không cần thao tác từ đầu, có thể nhấn nút **"🚪 Mở Door Panel"** ngay trong khối "Thiết lập AI" để chạy `door_gui.py` như một tiến trình con.

### 2. Bật Chế Độ Vận Hành Hệ Thống Toàn Phần

1. Trên phần mềm Proteus, nhấn nút Play ở góc trái phía dưới để khởi động mạch giả lập.
2. Khởi chạy tiến trình an ninh từ Python bằng lệnh: `python door_gui.py`.
3. Nhấn phím `SPACE` (khoảng trắng) để đánh thức Camera. Hệ thống sẽ quét khuôn mặt bằng mô hình đã huấn luyện ở Bước 1. Nếu nhận diện đúng thành viên, màn hình yêu cầu vượt qua thử thách chống giả mạo ngẫu nhiên.
4. Xác thực thành công ➔ hệ thống tự động ghi danh tính vào tệp `current_owner.txt`, đóng màn hình Cửa an ninh và gọi hàm `show_hub(owner_name)` để mở (hoặc cập nhật) giao diện Smart Home AI Hub đúng theo danh tính vừa đăng nhập.

Lúc này, luồng chạy ngầm điều hòa (`_auto_ac_loop`) bắt đầu hoạt động, cứ mỗi 8 giây lại đọc dữ liệu nhiệt độ từ con LM35 trên Proteus qua cổng Serial để AI tự động tinh chỉnh số độ C, đồng thời ghi nhớ thói quen thực tế mỗi khi người dùng bấm nút `+` hoặc `−` trên card "Điều hòa" của Smart Hub. Song song đó, camera cử chỉ tay cũng được tự động bật lên để sẵn sàng nhận lệnh điều khiển thiết bị.

### 3. Thao tác điều khiển thiết bị bằng Cử chỉ tay (Hand Gesture Control)

Khung **"Hand Gesture Camera"** ở cột phải Smart Hub vận hành theo cơ chế **khóa/mở khóa theo phiên** để tránh việc tay vô tình lướt qua camera làm thiết bị tự bật/tắt ngoài ý muốn:

1. **Trạng thái khóa (mặc định):** viền khung màu đỏ, dòng chữ nhắc "Giơ OK để mở khoá". Camera vẫn quan sát nhưng chưa nhận bất kỳ lệnh điều khiển nào.
2. **Mở khóa phiên điều khiển:** chụm đầu ngón cái và ngón trỏ lại gần nhau (khoảng cách nhỏ hơn ~20% kích thước bàn tay) trong khi vẫn xòe ít nhất một ngón giữa/áp út/út — giữ nguyên tư thế này liên tục trong khoảng 10 khung hình. Khi đủ điều kiện, viền khung chuyển sang màu xanh lá và một cửa sổ thời gian (khoảng 5–7 giây) được mở ra để nhận lệnh.
3. **Ra lệnh trong lúc đã mở khoá**, chọn một trong các cử chỉ sau:
   - ☝ Giơ đúng **1 ngón**, giữ yên ~10 khung hình → Bật/Tắt **Đèn**.
   - ✌ Giơ đúng **2 ngón**, giữ yên ~10 khung hình → Bật/Tắt **Quạt**.
   - 👉 **Vuốt tay sang phải** → gửi lệnh **Mở rèm** ngay lập tức (không cần giữ yên).
   - 👈 **Vuốt tay sang trái** → gửi lệnh **Đóng rèm** ngay lập tức.
   - ✊ **Nắm chặt bàn tay** (0 ngón), giữ yên ~10 khung hình → **Tắt toàn bộ** thiết bị.
4. **Tự khóa lại:** ngay sau khi một lệnh được thực thi (hoặc khi hết thời gian chờ của phiên mà không có cử chỉ hợp lệ nào), hệ thống tự động khóa lại về trạng thái ban đầu (viền đỏ). Muốn ra lệnh tiếp theo, người dùng phải lặp lại thao tác chụm tay mở khoá ở bước 2.
5. **Tự đóng camera khi rảnh (Idle Timeout):** nếu trong khoảng 400 khung hình (~20 giây) liên tục không phát sinh thao tác mở khoá/ra lệnh nào, khung Hand Gesture Camera sẽ tự giải phóng camera — đúng theo cơ chế Idle Timeout của máy trạng thái tổng đã mô tả ở phần Tính năng cốt lõi.

> 💡 Ngoài việc dùng ngay trong Smart Hub, có thể chạy độc lập module xử lý cử chỉ bằng lệnh `python test_hand.py` để kiểm thử riêng phần nhận diện tay và truyền lệnh UART xuống Arduino/Proteus mà không cần mở toàn bộ giao diện Smart Home AI Hub — rất hữu ích khi cần debug phần cứng.
