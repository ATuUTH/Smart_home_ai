import cv2
import math
import mediapipe as mp
import serial
import sys
from collections import deque

# ---------------- CẤU HÌNH CỔNG COM ----------------
try:
    arduino_port = serial.Serial('COM3', 9600, timeout=1) # Sửa lại đúng cổng COM của ông
    print("Đã kết nối thành công với cổng COM!")
except:
    arduino_port = None
    print("Chưa kết nối cổng COM, chạy AI chế độ TEST...")

# ---------------- CẤU HÌNH MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- BIẾN TOÀN CỤC: MÁY TRẠNG THÁI (STATE MACHINE) ---
frame_history = deque(maxlen=15)
is_awake = False
awake_timer = 0
stable_frames = 0
current_detect = ""

last_action = ""
action_display_timer = 0

# --- BIẾN MỚI CHO MASTER SCRIPT ---
# Nếu trong vòng 400 frames (~20 giây) không ai làm gì, tự tắt Hand AI để quay về Face AI
idle_timer = 400

# ---------------- HÀM TÍNH TOÁN GÓC ----------------
def calculate_angle(p1, p2, p3):
    a = math.dist(p2, p3)
    b = math.dist(p1, p3)
    c = math.dist(p1, p2)
    try:
        cos_b = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        cos_b = max(-1.0, min(1.0, cos_b))
        return math.degrees(math.acos(cos_b))
    except ZeroDivisionError:
        return 0

# ---------------- XỬ LÝ LOGIC CHÍNH ----------------
def process_hand_commands(lmList, hand_history):
    if not lmList or len(lmList) == 0:
        return None

    wrist = lmList[0][1:]
    thumb_tip = lmList[4][1:]
    index_tip = lmList[8][1:]
    middle_mcp = lmList[9][1:]

    hand_size = math.dist(wrist, middle_mcp)
    if hand_size == 0: hand_size = 1

    hand_history.append(middle_mcp)

    finger_joints = {
        "Thumb": (2, 3, 4), "Index": (5, 6, 8), "Middle": (9, 10, 12),
        "Ring": (13, 14, 16), "Pinky": (17, 18, 20)
    }

    open_fingers = {}
    total_fingers = 0
    for name, (p1, p2, p3) in finger_joints.items():
        angle = calculate_angle(lmList[p1][1:], lmList[p2][1:], lmList[p3][1:])
        threshold = 155 if name == "Thumb" else 145
        is_open = angle > threshold
        open_fingers[name] = is_open
        if is_open:
            total_fingers += 1

    pinch_dist = math.dist(thumb_tip, index_tip) / hand_size
    if pinch_dist < 0.20:
        if open_fingers["Middle"] or open_fingers["Ring"] or open_fingers["Pinky"]:
            return "S"

    if len(hand_history) == hand_history.maxlen:
        delta_x = hand_history[-1][0] - hand_history[0][0]
        swipe_threshold = hand_size * 0.45

        if delta_x > swipe_threshold:
            hand_history.clear()
            return "W"
        elif delta_x < -swipe_threshold:
            hand_history.clear()
            return "L"

    return str(total_fingers)

# ---------------- LUỒNG CHẠY CHÍNH ----------------
while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    raw_cmd = None

    # Tự động đếm ngược thời gian rảnh. Về 0 là thoát!
    idle_timer -= 1
    if idle_timer <= 0:
        print(">> Bảng điều khiển hết thời gian. Tự động đóng lại!")
        break # Thoát vòng lặp while để đóng Camera

    cv2.putText(img, f"Thoi gian cho: {idle_timer//20}s", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    border_color = (0, 255, 0) if is_awake else (0, 0, 255)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), border_color, 3)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        raw_cmd = process_hand_commands(lmList, frame_history)

        if raw_cmd:
            cv2.putText(img, f"AI thay: [{raw_cmd}]", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ==========================================
        # TRẠNG THÁI 1: KHÓA
        # ==========================================
        if not is_awake:
            cv2.putText(img, "HETHONG KHOA - Gio chu 'OK' de mo", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if raw_cmd == "S":
                stable_frames += 1
                cv2.putText(img, f'Dang mo khoa... {stable_frames}/10', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if stable_frames >= 10:
                    is_awake = True
                    awake_timer = 150 # Khoảng 5-7 giây để ra lệnh
                    stable_frames = 0
                    idle_timer = 400  # Đánh thức thành công thì reset bộ đếm thoát
                    frame_history.clear()
            else:
                stable_frames = 0

        # ==========================================
        # TRẠNG THÁI 2: ĐANG MỞ KHOÁ (RA LỆNH)
        # ==========================================
        else:
            awake_timer -= 1
            cv2.putText(img, f"DA MO KHOA! Ra lenh di... ({awake_timer//20}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if awake_timer <= 0:
                is_awake = False
                stable_frames = 0
                frame_history.clear()

            elif raw_cmd in ["W", "L"]:
                if arduino_port:
                    arduino_port.write(raw_cmd.encode())
                    print(f"➔ Đã truyền UART: {raw_cmd}")

                last_action = raw_cmd
                action_display_timer = 30
                is_awake = False
                idle_timer = 400 # Ra lệnh xong reset giờ để ông xem thông báo
                frame_history.clear()

            elif raw_cmd in ["0", "1", "2", "5"]:
                if raw_cmd == current_detect:
                    stable_frames += 1
                else:
                    current_detect = raw_cmd
                    stable_frames = 1

                cv2.putText(img, f'Dang chot: {raw_cmd} ({stable_frames}/10)', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if stable_frames >= 10:
                    if arduino_port:
                        arduino_port.write(raw_cmd.encode())
                        print(f"➔ Đã truyền UART: {raw_cmd}")

                    last_action = raw_cmd
                    action_display_timer = 30
                    is_awake = False
                    stable_frames = 0
                    idle_timer = 400
                    frame_history.clear()
            else:
                stable_frames = 0
    else:
        if len(frame_history) > 0: frame_history.popleft()
        stable_frames = 0

        if is_awake:
            awake_timer -= 1
            cv2.putText(img, f"DA MO KHOA! Ra lenh di... ({awake_timer//20}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if awake_timer <= 0:
                is_awake = False

    # --- HIỂN THỊ CHỮ TRÊN MÀN HÌNH ---
    if action_display_timer > 0:
        action_display_timer -= 1
        cv2.putText(img, f'>> VUA THUC HIEN: {last_action} <<', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

    cv2.imshow("SmartHome AI - Control Panel", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# ---------------- DỌN DẸP VÀ THOÁT ----------------
# Khi hết idle_timer, vòng lặp dừng lại và chạy xuống đây
if arduino_port: arduino_port.close()
cap.release()
cv2.destroyAllWindows()

# Báo hiệu cho file Master biết là Hand AI đã đóng
sys.exit(0)