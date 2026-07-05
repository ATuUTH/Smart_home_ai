import tkinter as tk
from tkinter import messagebox
import subprocess, threading, os, time, math, serial, csv
from datetime import datetime
from collections import deque
import queue
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

# --- THƯ VIỆN MACHINE LEARNING ---
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- SERIAL KẾT NỐI DUY NHẤT ---
try:
    _arduino = serial.Serial('COM3', 9600, timeout=1)
    HAS_SERIAL = True
except Exception:
    _arduino = None
    HAS_SERIAL = False


def uart_send(cmd: str):
    if HAS_SERIAL and _arduino:
        try:
            _arduino.write((cmd + '\n').encode())
        except Exception:
            pass


C = {
    "bg_app": "#0f1117",
    "bg_card": "#1a1d27",
    "bg_input": "#252836",
    "bg_hover": "#2e3245",
    "blue": "#4f8ef7",
    "green": "#3ecf8e",
    "amber": "#f5a623",
    "red": "#e05c5c",
    "purple": "#9b7ff4",
    "cyan": "#3ecfb8",
    "text_primary": "#eef0f8",
    "text_secondary": "#8890a8",
    "text_muted": "#555a72",
    "border": "#2a2d3e",
}
F_TITLE, F_LABEL, F_SMALL, F_SECTION, F_MONO, F_BIG = ("Segoe UI", 11, "bold"), ("Segoe UI", 10), ("Segoe UI", 9), (
    "Segoe UI", 9, "bold"), ("Consolas", 9), ("Segoe UI", 22, "bold")

LOG_FILE = "ac_habit_log.csv"
LOG_FIELDS = ["hour", "outdoor_temp", "indoor_temp", "owner", "target_temp"]


# =========================================================
# BỘ NÃO MACHINE LEARNING TỰ HỌC THÓI QUEN ĐIỀU HÒA
# =========================================================
class HabitEngine:
    def __init__(self, owner: str):
        self.owner = owner
        self.scaler = StandardScaler()
        self.knn = None
        self.lr = None
        self.trained = False
        self.n = 0
        self._train()

    def _load(self):
        X, y = [], []
        if not os.path.exists(LOG_FILE):
            return X, y
        with open(LOG_FILE, newline='') as f:
            for row in csv.DictReader(f):
                if row.get("owner") == self.owner:
                    try:
                        X.append([float(row["hour"]), float(row["outdoor_temp"]), float(row["indoor_temp"])])
                        y.append(float(row["target_temp"]))
                    except (ValueError, KeyError):
                        pass
        return X, y

    def _train(self):
        X, y = self._load()
        self.n = len(X)
        if self.n >= 3:  # Có từ 3 mẫu trở lên là AI bắt đầu học
            Xs = self.scaler.fit_transform(X)
            k = min(3, self.n)
            self.knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            self.knn.fit(Xs, y)
            self.lr = LinearRegression().fit(Xs, y)
            self.trained = True

    def predict(self, outdoor: float, indoor: float) -> tuple[float, str]:
        # ĐÃ SỬA: Luật fallback linh hoạt dựa vào nhiệt độ Proteus nếu chưa có data học
        if not self.trained:
            if indoor > 30.0:
                t = 22.0
            elif indoor > 27.0:
                t = 24.0
            else:
                t = 26.0
            return t, f"AI RULE (Cần thêm {3 - self.n} mẫu)"

        hour = datetime.now().hour
        feat = self.scaler.transform([[hour, outdoor, indoor]])
        knn_p = self.knn.predict(feat)[0]
        if self.n >= 15:
            lr_p = self.lr.predict(feat)[0]
            t = round(0.7 * knn_p + 0.3 * lr_p, 1)
            src = f"ENSEMBLE AI ({self.n} mẫu)"
        else:
            t = round(knn_p, 1)
            src = f"KNN AI ({self.n} mẫu)"
        return max(16.0, min(30.0, t)), src

    def record(self, outdoor: float, indoor: float, target: float):
        write_hdr = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_hdr: w.writeheader()
            w.writerow({"hour": datetime.now().hour,
                        "outdoor_temp": round(outdoor, 1),
                        "indoor_temp": round(indoor, 1),
                        "owner": self.owner,
                        "target_temp": round(target, 1)})
        self._train()  # Tiến hóa: Học lại ngay lập tức sau khi chủ sửa số!


# --- GESTURE PROCESSOR ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def _background_check(): pass


def _angle(p1, p2, p3):
    a, b, c = math.dist(p2, p3), math.dist(p1, p3), math.dist(p1, p2)
    try:
        return math.degrees(math.acos(max(-1.0, min(1.0, (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))))
    except ZeroDivisionError:
        return 0


def parse_gesture(lm_list, history: deque):
    if not lm_list: return None
    wrist, thumb_tip, index_tip, middle_mcp = lm_list[0][1:], lm_list[4][1:], lm_list[8][1:], lm_list[9][1:]
    hand_size = math.dist(wrist, middle_mcp) or 1
    history.append(middle_mcp)
    joints = {"Thumb": (2, 3, 4), "Index": (5, 6, 8), "Middle": (9, 10, 12), "Ring": (13, 14, 16),
              "Pinky": (17, 18, 20)}
    open_f, total = {}, 0
    for name, (p1, p2, p3) in joints.items():
        open_f[name] = _angle(lm_list[p1][1:], lm_list[p2][1:], lm_list[p3][1:]) > (155 if name == "Thumb" else 145)
        if open_f[name]: total += 1
    if math.dist(thumb_tip, index_tip) / hand_size < 0.20 and (
            open_f["Middle"] or open_f["Ring"] or open_f["Pinky"]): return "OK"
    if len(history) == history.maxlen:
        dx = history[-1][0] - history[0][0]
        if dx > hand_size * 0.45: history.clear(); return "W"
        if dx < -hand_size * 0.45: history.clear(); return "L"
    return str(total)


class card(tk.Frame):
    def __init__(self, parent, **kw): super().__init__(parent, bg=C["bg_card"], highlightbackground=C["border"],
                                                       highlightthickness=1, **kw)


def sec_label(parent, text, bg=None):
    f = tk.Frame(parent, bg=bg or C["bg_app"])
    tk.Label(f, text=text.upper(), font=F_SECTION, fg=C["text_muted"], bg=bg or C["bg_app"]).pack(side="left")
    return f


def hdiv(parent, bg=None): return tk.Frame(parent, height=1, bg=bg or C["border"])


class StepBtn(tk.Frame):
    def __init__(self, parent, num, text, color, cmd):
        super().__init__(parent, bg=C["bg_card"])
        tk.Label(self, text=str(num), font=("Segoe UI", 10, "bold"), fg=color, bg=C["bg_card"], width=2, padx=8,
                 pady=7).pack(side="left")
        b = tk.Button(self, text=text, font=F_LABEL, fg=C["text_primary"], bg=C["bg_card"],
                      activebackground=C["bg_hover"], activeforeground=C["text_primary"], relief="flat", cursor="hand2",
                      anchor="w", padx=10, pady=7, command=cmd)
        b.pack(side="left", fill="x", expand=True)
        b.bind("<Enter>", lambda e: b.config(bg=C["bg_hover"]));
        b.bind("<Leave>", lambda e: b.config(bg=C["bg_card"]))
        self._s = tk.Label(self, text="", font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"], padx=10);
        self._s.pack(side="right")

    def set_state(self, t, color=None): self._s.config(text=t, fg=color or C["text_muted"])


class LogBox(tk.Frame):
    def __init__(self, parent, height=7):
        super().__init__(parent, bg=C["bg_card"], highlightbackground=C["border"], highlightthickness=1)
        hdr = tk.Frame(self, bg=C["bg_card"], padx=12, pady=6);
        hdr.pack(fill="x")
        tk.Label(hdr, text="ACTIVITY LOG", font=F_SECTION, fg=C["text_muted"], bg=C["bg_card"]).pack(side="left")
        tk.Button(hdr, text="Clear", font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"], activebackground=C["bg_hover"],
                  relief="flat", cursor="hand2", command=self._clear).pack(side="right")
        hdiv(self).pack(fill="x")
        self.txt = tk.Text(self, height=height, bg=C["bg_input"], fg=C["text_secondary"], font=F_MONO, relief="flat",
                           padx=10, pady=8, state="disabled", cursor="arrow", selectbackground=C["bg_hover"],
                           wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        for tag, col in [("ok", C["green"]), ("warn", C["amber"]), ("err", C["red"]), ("info", C["blue"]),
                         ("ts", C["text_muted"])]: self.txt.tag_config(tag, foreground=col)

    def log(self, msg, level="info"):
        self.txt.config(state="normal");
        self.txt.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] ", "ts");
        self.txt.insert("end", msg + "\n", level if level in ("ok", "warn", "err", "info") else "info");
        self.txt.see("end");
        self.txt.config(state="disabled")

    def _clear(self): self.txt.config(state="normal"); self.txt.delete("1.0", "end"); self.txt.config(state="disabled")


# --- ĐÃ SỬA: THÊM COONG / TRỪ ĐIỀU CHỈNH NHIỆT ĐỘ AC TRÊN CARD TÍNH NĂNG ---
class DeviceCard(tk.Frame):
    def __init__(self, parent, icon, name, gesture_on, gesture_off, uart_on, uart_off, has_temp=False, on_up=None,
                 on_down=None):
        super().__init__(parent, bg=C["bg_card"], highlightbackground=C["border"], highlightthickness=1)
        self._on, self._uart_on, self._uart_off, self._name = False, uart_on, uart_off, name
        self._bar = tk.Frame(self, bg=C["text_muted"], width=3);
        self._bar.pack(side="left", fill="y")
        body = tk.Frame(self, bg=C["bg_card"], padx=12, pady=10);
        body.pack(side="left", fill="both", expand=True)
        top = tk.Frame(body, bg=C["bg_card"]);
        top.pack(fill="x")
        tk.Label(top, text=f"{icon}  {name}", font=F_TITLE, fg=C["text_primary"], bg=C["bg_card"]).pack(side="left")
        self._status = tk.Label(top, text="OFF", font=("Segoe UI", 9, "bold"), fg=C["text_muted"], bg=C["bg_card"]);
        self._status.pack(side="right")
        tk.Label(body, text=f"Bật: {gesture_on}   Tắt: {gesture_off}", font=F_SMALL, fg=C["text_muted"],
                 bg=C["bg_card"]).pack(anchor="w", pady=(3, 0))

        # Cụm nút thao tác điều khiển ngang hàng
        row_btn = tk.Frame(body, bg=C["bg_card"]);
        row_btn.pack(anchor="w", pady=(6, 0))
        self._btn = tk.Button(row_btn, text="Bật", font=F_SMALL, fg=C["text_primary"], bg=C["bg_input"],
                              activebackground=C["bg_hover"], relief="flat", cursor="hand2", padx=10, pady=3,
                              command=self.toggle)
        self._btn.pack(side="left")

        if has_temp:
            tk.Label(row_btn, text="   Cài đặt:", font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"]).pack(side="left")
            btn_sub = tk.Button(row_btn, text="—", font=("Segoe UI", 8, "bold"), fg=C["text_primary"], bg=C["bg_input"],
                                relief="flat", cursor="hand2", padx=6, command=on_down)
            btn_sub.pack(side="left", padx=4)
            self._temp_lbl = tk.Label(row_btn, text="25.0°C", font=F_LABEL, fg=C["cyan"], bg=C["bg_card"])
            self._temp_lbl.pack(side="left", padx=2)
            btn_add = tk.Button(row_btn, text="+", font=("Segoe UI", 8, "bold"), fg=C["text_primary"], bg=C["bg_input"],
                                relief="flat", cursor="hand2", padx=6, command=on_up)
            btn_add.pack(side="left", padx=4)

    def toggle(self, force=None):
        self._on = (not self._on) if force is None else force
        if self._on:
            self._bar.config(bg=C["green"]);
            self._status.config(text="ON", fg=C["green"]);
            self._btn.config(text="Tắt")
            uart_send(self._uart_on)
        else:
            self._bar.config(bg=C["text_muted"]);
            self._status.config(text="OFF", fg=C["text_muted"]);
            self._btn.config(text="Bật")
            uart_send(self._uart_off)


class GesturePanel(tk.Frame):
    CAM_W, CAM_H = 320, 240

    def __init__(self, parent, on_command, log_fn):
        super().__init__(parent, bg=C["bg_card"], highlightbackground=C["border"], highlightthickness=1)
        self._on_command, self._log = on_command, log_fn
        self._hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self._history, self._cap, self._running, self._img_tk, self._frame_q = deque(maxlen=15), None, False, None, None
        self._locked, self._unlock_buf, self._awake_timer, self._stable_cmd, self._stable_n, self._pending_cmd = True, 0, 0, None, 0, None
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=C["bg_card"], padx=12, pady=8);
        hdr.pack(fill="x")
        tk.Label(hdr, text="HAND GESTURE CONTROL", font=F_SECTION, fg=C["text_muted"], bg=C["bg_card"]).pack(
            side="left")
        self._cam_dot = tk.Label(hdr, text="● OFF", font=("Segoe UI", 8, "bold"), fg=C["text_muted"], bg=C["bg_card"]);
        self._cam_dot.pack(side="right")
        hdiv(self).pack(fill="x")
        body = tk.Frame(self, bg=C["bg_card"], padx=12, pady=10);
        body.pack(fill="both", expand=True)
        cam_wrap = tk.Frame(body, bg="#06080f", highlightbackground=C["border"], highlightthickness=1);
        cam_wrap.pack(fill="x")
        self._canvas = tk.Canvas(cam_wrap, width=self.CAM_W, height=self.CAM_H, bg="#06080f", highlightthickness=0);
        self._canvas.pack()
        self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw", image="")
        info = tk.Frame(body, bg=C["bg_card"]);
        info.pack(fill="x", pady=(10, 0))
        self._lock_lbl = tk.Label(info, text="🔒  Giơ OK để mở khoá", font=F_LABEL, fg=C["amber"], bg=C["bg_card"]);
        self._lock_lbl.pack(side="left")
        self._cmd_lbl = tk.Label(info, text="", font=("Segoe UI", 10, "bold"), fg=C["cyan"], bg=C["bg_card"]);
        self._cmd_lbl.pack(side="right")
        hint_card = tk.Frame(body, bg=C["bg_input"], highlightbackground=C["border"], highlightthickness=1);
        hint_card.pack(fill="x", pady=(10, 0))
        hints = [("☝ 1 ngón", "Đèn ON/OFF"), ("✌ 2 ngón", "Quạt ON/OFF"), ("👉 Vuốt Phải", "Mở Rèm"),
                 ("👈 Vuốt Trái", "Đóng Rèm"), ("✊ Nắm tay", "Tắt Sạch Nhà")]
        for i, (gest, act) in enumerate(hints):
            r, c2 = divmod(i, 2);
            rf = tk.Frame(hint_card, bg=C["bg_input"]);
            rf.grid(row=r, column=c2, sticky="w", padx=10, pady=3)
            tk.Label(rf, text=gest, font=F_SMALL, fg=C["text_secondary"], bg=C["bg_input"], width=14, anchor="w").pack(
                side="left")
            tk.Label(rf, text=act, font=F_SMALL, fg=C["text_muted"], bg=C["bg_input"]).pack(side="left")

    def start(self):
        if self._running: return
        self._frame_q, self._running = queue.Queue(maxsize=2), True
        self._cam_dot.config(text="● Đang kết nối...", fg=C["amber"])
        threading.Thread(target=self._cam_thread, daemon=True).start()
        self.winfo_toplevel().after(50, self._poll_frame)

    def stop(self):
        self._running = False;
        self._frame_q = None

        def _release():
            time.sleep(0.25)
            if self._cap and self._cap.isOpened(): self._cap.release()
            self._cap = None

        threading.Thread(target=_release, daemon=True).start()
        try:
            self._cam_dot.config(text="● OFF", fg=C["text_muted"]); self._canvas.itemconfig(self._canvas_img_id,
                                                                                            image=""); self._img_tk = None
        except Exception:
            pass
        self._log("Hand Gesture camera tắt", "warn")

    def _cam_thread(self):
        deadline, cap = time.time() + 15, None
        while self._running and time.time() < deadline:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW);
            time.sleep(0.5)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: break
            cap.release();
            cap = None;
            time.sleep(0.8)
        if cap is None or not cap.isOpened():
            self._running = False
            if self._frame_q:
                try:
                    self._frame_q.put_nowait("ERROR")
                except Exception:
                    pass
            return
        self._cap = cap
        if self._frame_q:
            try:
                self._frame_q.put_nowait("LIVE")
            except Exception:
                pass
        while self._running:
            ret, frame = self._cap.read()
            if not ret: time.sleep(0.05); continue
            frame = cv2.flip(frame, 1);
            h, w = frame.shape[:2];
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb);
            lm_list, raw_cmd = [], None
            if result.multi_hand_landmarks:
                for hlms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS)
                    for idx, lm in enumerate(hlms.landmark): lm_list.append([idx, int(lm.x * w), int(lm.y * h)])
                raw_cmd = parse_gesture(lm_list, self._history)
            else:
                if self._history: self._history.popleft()
            dispatch_cmd = self._update_state(raw_cmd, frame, w, h)
            if dispatch_cmd: self._pending_cmd = dispatch_cmd
            rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.CAM_W, self.CAM_H))
            if self._frame_q:
                try:
                    if self._frame_q.full():
                        try:
                            self._frame_q.get_nowait()
                        except Exception:
                            pass
                    self._frame_q.put_nowait(rgb_small.tobytes())
                except Exception:
                    pass
            time.sleep(0.030)
        if self._cap: self._cap.release()

    def _poll_frame(self):
        if not self._running: return
        try:
            if not self.winfo_exists(): return
        except Exception:
            return
        if self._frame_q:
            try:
                item = self._frame_q.get_nowait()
                if item == "LIVE":
                    self._cam_dot.config(text="● LIVE", fg=C["green"]); self._log("Hand Gesture camera bật", "ok")
                elif item == "ERROR":
                    self._cam_dot.config(text="● LỖI CAM", fg=C["red"]); self._log("Không mở được camera.",
                                                                                   "err"); return
                elif item == "__UNLOCK__":
                    self._lock_lbl.config(text="🔓  Đã mở — ra lệnh đi!", fg=C["green"])
                elif item == "__LOCK__":
                    self._lock_lbl.config(text="🔒  Giơ OK để mở khoá", fg=C["amber"])
                elif isinstance(item, bytes):
                    img = Image.frombytes("RGB", (self.CAM_W, self.CAM_H), item);
                    tk_img = ImageTk.PhotoImage(img)
                    self._img_tk = tk_img;
                    self._canvas.itemconfig(self._canvas_img_id, image=tk_img)
            except Exception:
                pass
        if self._pending_cmd:
            cmd = self._pending_cmd;
            self._pending_cmd = None
            if cmd not in ("__LOCK__", "__UNLOCK__"):
                self._on_command(cmd);
                self._cmd_lbl.config(text=f"▶ {cmd}");
                self._log(f"Gesture: {cmd}", "info")
                self.after(1500, self._cmd_lbl.config, {"text": ""})
        try:
            if self.winfo_exists(): self.winfo_toplevel().after(33, self._poll_frame)
        except Exception:
            pass

    def _update_state(self, raw_cmd, frame, w, h) -> str | None:
        h_px = frame.shape[0]
        if self._locked:
            cv2.putText(frame, "LOCKED - Giu OK de mo", (10, h_px - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 166, 35),
                        2)
            if raw_cmd == "OK":
                self._unlock_buf += 1
                cv2.putText(frame, f"Opening {self._unlock_buf}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (62, 207, 142), 2)
                if self._unlock_buf >= 10:
                    self._locked, self._unlock_buf, self._awake_timer, self._pending_cmd = False, 0, 150, "__UNLOCK__"
            else:
                self._unlock_buf = 0
            return None

        self._awake_timer -= 1
        cv2.putText(frame, f"Unlocked {self._awake_timer // 30}s", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (62, 207, 142), 2)
        if self._awake_timer <= 0:
            self._locked, self._stable_cmd, self._stable_n, self._pending_cmd = True, None, 0, "__LOCK__"
            return None

        if raw_cmd in ("W", "L"):
            self._awake_timer = 150;
            return raw_cmd
        if raw_cmd in ("0", "1", "2"):
            self._awake_timer = 150
            if raw_cmd == self._stable_cmd:
                self._stable_n += 1
            else:
                self._stable_cmd, self._stable_n = raw_cmd, 1
            cv2.putText(frame, f"Chot: {raw_cmd} ({self._stable_n}/10)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (62, 207, 142), 2)
            if self._stable_n >= 10: self._stable_n = 0; return raw_cmd
        else:
            self._stable_n = 0
        return None


class HubApp(tk.Toplevel):
    def __init__(self, owner: str = ""):
        super().__init__()
        self.title("Smart Home AI Hub")
        self.geometry("1100x700");
        self.configure(bg=C["bg_app"]);
        self.resizable(True, True);
        self.minsize(900, 600)
        self._owner, self._cam_active = owner, False

        # Bộ đệm để Feedback Loop lưu thói quen thực tế
        self.current_target_ac = 25.0
        self.last_known_indoor = 28.5

        self._build();
        self._tick();
        self.after(400, self._startup_check)
        if owner:
            self.after(2000, self._enable_gesture)
            self.after(2500, self._init_smart_ac)

    def _build(self):
        self._build_header()
        body = tk.Frame(self, bg=C["bg_app"]);
        body.pack(fill="both", expand=True, padx=14, pady=10)
        col_left, col_mid, col_right = tk.Frame(body, bg=C["bg_app"]), tk.Frame(body, bg=C["bg_app"]), tk.Frame(body,
                                                                                                                bg=C[
                                                                                                                    "bg_app"])
        col_left.pack(side="left", fill="both", padx=(0, 8))
        col_mid.pack(side="left", fill="both", expand=True, padx=(0, 8))
        col_right.pack(side="left", fill="both", padx=(0, 0))
        self._build_col_left(col_left);
        self._build_col_mid(col_mid);
        self._build_col_right(col_right)

    def _build_header(self):
        hdr = card(self);
        hdr.pack(fill="x")
        lft = tk.Frame(hdr, bg=C["bg_card"], padx=16, pady=10);
        lft.pack(side="left")
        tk.Label(lft, text="⌂  Smart Home AI Hub", font=("Segoe UI", 13, "bold"), fg=C["text_primary"],
                 bg=C["bg_card"]).pack(side="left")
        self._owner_lbl = tk.Label(lft, text="", font=F_SMALL, fg=C["cyan"], bg=C["bg_card"]);
        self._owner_lbl.pack(side="left", padx=(10, 0))
        rgt = tk.Frame(hdr, bg=C["bg_card"], padx=16, pady=8);
        rgt.pack(side="right")
        self._cv, self._dv = tk.StringVar(value="--:--:--"), tk.StringVar(value="")
        tk.Label(rgt, textvariable=self._cv, font=("Segoe UI", 11), fg=C["text_secondary"], bg=C["bg_card"]).pack()
        tk.Label(rgt, textvariable=self._dv, font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"]).pack()

    def _build_col_left(self, parent):
        sec_label(parent, "Thành viên").pack(fill="x", pady=(0, 6))
        mem_card = card(parent);
        mem_card.pack(fill="x", pady=(0, 12))
        mi = tk.Frame(mem_card, bg=C["bg_card"], padx=12, pady=10);
        mi.pack(fill="x")
        self._mem_frame = tk.Frame(mi, bg=C["bg_card"]);
        self._mem_frame.pack(fill="x")
        sec_label(parent, "Thiết lập AI").pack(fill="x", pady=(0, 6))
        sc = card(parent);
        sc.pack(fill="x")
        inner = tk.Frame(sc, bg=C["bg_card"], padx=12, pady=10);
        inner.pack(fill="x")
        nr = tk.Frame(inner, bg=C["bg_card"]);
        nr.pack(fill="x", pady=(0, 8))
        tk.Label(nr, text="Tên:", font=F_SMALL, fg=C["text_secondary"], bg=C["bg_card"]).pack(side="left")
        self._name = tk.Entry(nr, font=F_LABEL, bg=C["bg_input"], fg=C["text_primary"],
                              insertbackground=C["text_primary"], relief="flat", width=14);
        self._name.pack(side="left", padx=(8, 0), ipady=4)
        hdiv(inner).pack(fill="x", pady=(0, 8))
        steps = [(1, "Đăng ký khuôn mặt", C["blue"], self._register_face),
                 (2, "Lưu và cập nhật trạng thái", C["amber"], self._train_ai)]
        self._steps = {}
        for n, txt, col, fn in steps:
            b = StepBtn(inner, n, txt, col, fn);
            b.pack(fill="x", pady=2);
            self._steps[n] = b
        hdiv(inner).pack(fill="x", pady=(8, 0))
        tk.Frame(inner, bg=C["bg_card"], height=6).pack()
        self._door_btn = tk.Button(inner, text="🚪  Mở Door Panel", font=F_LABEL, fg=C["text_primary"], bg=C["bg_card"],
                                   activebackground=C["bg_hover"], relief="flat", cursor="hand2", padx=10, pady=7,
                                   highlightbackground=C["border"], highlightthickness=1, command=self._open_door);
        self._door_btn.pack(fill="x")

    def _build_col_mid(self, parent):
        sec_label(parent, "Điều khiển thiết bị").pack(fill="x", pady=(0, 6))

        # Khởi tạo 3 thiết bị đầu tiên bình thường
        self._devices = {}
        dc1 = DeviceCard(parent, "💡", "Đèn phòng", "☝ 1 ngón", "✊ Nắm tắt", "LAMP_ON", "LAMP_OFF")
        dc1.pack(fill="x", pady=(0, 6));
        self._devices["LAMP_ON"] = dc1

        dc2 = DeviceCard(parent, "🌀", "Quạt", "✌ 2 ngón", "✊ Nắm tắt", "FAN_ON", "FAN_OFF")
        dc2.pack(fill="x", pady=(0, 6));
        self._devices["FAN_ON"] = dc2

        dc3 = DeviceCard(parent, "🪟", "Rèm cửa", "👉 Vuốt phải", "👈 Vuốt trái", "W", "L")
        dc3.pack(fill="x", pady=(0, 6));
        self._devices["W"] = dc3

        # --- ĐÃ SỬA: CARD ĐIỀU HÒA ĐƯỢC TÍCH HỢP THÊM HAI NÚT TĂNG GIẢM NHIỆT ĐỘ ---
        dc_ac = DeviceCard(parent, "❄", "Điều hòa (AI Auto)", "(Tự động)", "(Tự động)", "AC_ON", "AC_OFF",
                           has_temp=True, on_up=self._manual_ac_up, on_down=self._manual_ac_down)
        dc_ac.pack(fill="x", pady=(0, 6));
        self._devices["AC_ON"] = dc_ac

        all_off = tk.Button(parent, text="⏹  Tắt tất cả thiết bị", font=F_LABEL, fg=C["text_secondary"],
                            bg=C["bg_card"], activebackground=C["bg_hover"], relief="flat", cursor="hand2", padx=10,
                            pady=7, highlightbackground=C["border"], highlightthickness=1, command=self._all_off);
        all_off.pack(fill="x", pady=(0, 10))
        self._log_box = LogBox(parent, height=6);
        self._log_box.pack(fill="both", expand=True)

    def _build_col_right(self, parent):
        sec_label(parent, "Hand Gesture Camera").pack(fill="x", pady=(0, 6))
        self._gesture_panel = GesturePanel(parent, on_command=self._on_gesture, log_fn=self._log_box.log);
        self._gesture_panel.pack(fill="both", expand=True)
        tk.Frame(parent, bg=C["bg_app"], height=8).pack()
        self._cam_toggle = tk.Button(parent, text="▶  Bật camera gesture", font=F_LABEL, fg=C["text_primary"],
                                     bg=C["blue"], activebackground=C["bg_hover"], relief="flat", cursor="hand2",
                                     padx=10, pady=8, command=self._toggle_cam);
        self._cam_toggle.pack(fill="x")
        tk.Frame(parent, bg=C["bg_app"], height=10).pack()
        ic = card(parent);
        ic.pack(fill="x")
        ii = tk.Frame(ic, bg=C["bg_card"], padx=12, pady=8);
        ii.pack(fill="x")
        self._irows = {}
        for key, val in [("Face model", "—"), ("Dataset", "—"), ("Smart AC", "KNN+Linear"), ("Arduino", "COM3")]:
            row = tk.Frame(ii, bg=C["bg_card"]);
            row.pack(fill="x", pady=2)
            tk.Label(row, text=key, font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"], width=12, anchor="w").pack(
                side="left")
            lbl = tk.Label(row, text=val, font=F_SMALL, fg=C["text_secondary"], bg=C["bg_card"], anchor="e");
            lbl.pack(side="right");
            self._irows[key] = lbl

    def _tick(self):
        now = datetime.now();
        self._cv.set(now.strftime("%H:%M:%S"));
        self._dv.set(now.strftime("%A, %d/%m/%Y"));
        self.after(1000, self._tick)

    def _startup_check(self):
        if self._owner: self._owner_lbl.config(text=f"  Xin chào, {self._owner}!")
        has_model = os.path.exists("face_ocsvm_model.pkl");
        self._irows["Face model"].config(text="✓ Đã có" if has_model else "Chưa train",
                                         fg=C["green"] if has_model else C["amber"])
        csvs = [f for f in os.listdir(".") if f.startswith("dataset_") and f.endswith(".csv")];
        names = [f.replace("dataset_", "").replace(".csv", "") for f in csvs];
        self._refresh_members(names)
        self._irows["Arduino"].config(text="✓ COM3" if HAS_SERIAL else "Không kết nối",
                                      fg=C["green"] if HAS_SERIAL else C["amber"])
        self._log_box.log("Hệ thống sẵn sàng", "ok" if has_model else "warn")

    # --- ĐÃ SỬA: THÊM NÚT XÓA THÀNH VIÊN TRỰC QUAN ---
    def _refresh_members(self, names):
        for w in self._mem_frame.winfo_children(): w.destroy()
        if not names: tk.Label(self._mem_frame, text="Chưa có thành viên.", font=F_SMALL, fg=C["text_muted"],
                               bg=C["bg_card"]).pack(anchor="w", pady=4); return
        for name in names:
            row = tk.Frame(self._mem_frame, bg=C["bg_card"]);
            row.pack(fill="x", pady=2)
            tk.Label(row, text="".join(p[0].upper() for p in name.split()[:2]), font=("Segoe UI", 8, "bold"),
                     fg=C["blue"], bg=C["bg_input"], width=3, padx=4, pady=3).pack(side="left", padx=(0, 8))
            tk.Label(row, text=name, font=F_LABEL, fg=C["text_primary"], bg=C["bg_card"]).pack(side="left")
            # Ghìm nút xóa dấu X vào góc phải
            tk.Button(row, text="✕", font=F_SMALL, fg=C["text_muted"], bg=C["bg_card"], activebackground=C["bg_hover"],
                      relief="flat", cursor="hand2", command=lambda n=name: self._delete_member(n)).pack(side="right",
                                                                                                         padx=4)

    def _delete_member(self, name):
        if not messagebox.askyesno("Xoá thành viên",
                                   f"Bạn chắc chắn muốn xoá dữ liệu của '{name}'?\nHệ thống sẽ buộc phải train lại mô hình AI."):
            return
        path = f"dataset_{name}.csv"
        if os.path.exists(path):
            os.remove(path)
            self._log_box.log(f"Đã gỡ bỏ dữ liệu: {name}", "warn")
            self.after(100, self._startup_check)

    def _on_gesture(self, cmd: str):
        if cmd == "0": self._all_off(); return
        ui_map = {"1": ("LAMP_ON", "đèn"), "2": ("FAN_ON", "quạt"), "W": ("W", "mở rèm"), "L": ("L", "đóng rèm")}
        if cmd in ui_map:
            uart_cmd, name = ui_map[cmd]
            uart_send(uart_cmd)
            if cmd == "1":
                self._devices["LAMP_ON"].toggle(force=True)
            elif cmd == "2":
                self._devices["FAN_ON"].toggle(force=True)
            elif cmd == "W":
                self._devices["W"].toggle(force=True)
            elif cmd == "L":
                self._devices["W"].toggle(force=False)
            self._log_box.log(f"Cử chỉ [{cmd}]: Kích hoạt {name}", "info")

    def _all_off(self):
        uart_send("0")
        for dc in self._devices.values(): dc.toggle(force=False)
        self._log_box.log("Tắt tất cả thiết bị", "warn")

    # =========================================================================
    # KHU VỰC ĐIỀU KHIỂN ĐIỀU HÒA BẰNG TAY (FEEDBACK LOOP - VÒNG LẶP TIẾN HÓA)
    # =========================================================================
    def _manual_ac_up(self):
        if not hasattr(self, "habit_ai"): return
        self.current_target_ac = min(30.0, self.current_target_ac + 0.5)
        self._devices["AC_ON"]._temp_lbl.config(text=f"{self.current_target_ac}°C")
        uart_send(f"SET:{self.current_target_ac}:COOL")
        self._log_box.log(f"Chỉnh tay AC → {self.current_target_ac}°C (AI đang ghi nhớ thói quen...)", "warn")

        # Lấy nhiệt độ ngoài trời tại đúng khoảnh khắc bấm nút
        outdoor = getattr(self, "current_outdoor_temp", 34.0)
        self.habit_ai.record(outdoor, self.last_known_indoor, self.current_target_ac)

    def _manual_ac_down(self):
        if not hasattr(self, "habit_ai"): return
        self.current_target_ac = max(16.0, self.current_target_ac - 0.5)
        self._devices["AC_ON"]._temp_lbl.config(text=f"{self.current_target_ac}°C")
        uart_send(f"SET:{self.current_target_ac}:COOL")
        self._log_box.log(f"Chỉnh tay AC → {self.current_target_ac}°C (AI đang ghi nhớ thói quen...)", "warn")

        # Lấy nhiệt độ ngoài trời tại đúng khoảnh khắc bấm nút
        outdoor = getattr(self, "current_outdoor_temp", 34.0)
        self.habit_ai.record(outdoor, self.last_known_indoor, self.current_target_ac)

    # =========================================================================
    # KHU VỰC CHẠY AI TỰ ĐỘNG (DÙNG CHUNG CỔNG COM - KHÔNG LO XUNG ĐỘT)
    # =========================================================================
    def _init_smart_ac(self):
        self._log_box.log(f"Đang nạp thuật toán KNN + Linear Regression cho: {self._owner}", "info")
        self.habit_ai = HabitEngine(self._owner)  # Khởi tạo lõi học máy thực thụ
        uart_send("AC_ON")
        self._devices["AC_ON"].toggle(force=True)
        self._auto_ac_loop()

    def _auto_ac_loop(self):
        if not self.winfo_exists(): return

        # THÀNH PHẦN 1: Tự động nạp model Demo nếu ông mở trực tiếp file Hub để chống sập vòng lặp
        if not hasattr(self, "habit_ai"):
            self.habit_ai = HabitEngine(self._owner or "Demo")

        # Lấy nhiệt độ nền ban đầu
        indoor_temp = getattr(self, "last_known_indoor", 27.0)

        if HAS_SERIAL and _arduino:
            try:
                _arduino.write(b"R\n")
                line = _arduino.readline().decode(errors="ignore").strip()
                if line.startswith("TEMP_IN:"):
                    indoor_temp = float(line.split(":")[1])
            except Exception as e:
                # THÀNH PHẦN 2: Nếu có lỗi kết nối cổng COM, in thẳng lên log để dễ dò bug
                self._log_box.log(f"Kết nối Serial lỗi: {e}", "err")

        self.current_outdoor_temp = round(np.random.uniform(32.5, 35.5), 1)
        self.last_known_indoor = indoor_temp

        # Gọi bộ não AI dự đoán nhiệt độ thích hợp
        target_ac, ai_model_src = self.habit_ai.predict(self.current_outdoor_temp, indoor_temp)

        # THÀNH PHẦN 3: Ép hiển thị Log liên tục ra màn hình cho đẹp giao diện
        if self._devices["AC_ON"]._on:
            # Nếu Điều hòa đang BẬT -> Hiện dòng log màu xanh lá [ok]
            self.current_target_ac = target_ac
            self._devices["AC_ON"]._temp_lbl.config(text=f"{target_ac}°C")
            uart_send(f"SET:{target_ac}:COOL")
            self._log_box.log(
                f"[{ai_model_src}] Phòng {indoor_temp}C | Ngoài {self.current_outdoor_temp}C → Đặt {target_ac}C", "ok")
        else:
            # Nếu Điều hòa đang TẮT -> Vẫn hiện log giám sát màu xanh dương [info] để màn hình có chữ chạy
            self._log_box.log(
                f"Giám sát phòng: {indoor_temp}C | Ngoài trời: {self.current_outdoor_temp}C [AC: ĐANG TẮT]", "info")

        # Đều đặn quét cập nhật sau mỗi 8 giây
        self.after(8000, self._auto_ac_loop)

    def _toggle_cam(self):
        if not self._cam_active:
            self._enable_gesture()
        else:
            self._disable_gesture()

    def _enable_gesture(self):
        self._cam_active = True;
        self._cam_toggle.config(text="⏳ Đang bật cam...", bg=C["bg_hover"], fg=C["amber"])
        self._gesture_panel.start();
        self.after(3000, self._check_cam_ok)

    def _check_cam_ok(self):
        if not self._cam_active: return
        if self._gesture_panel._cap and self._gesture_panel._cap.isOpened():
            self._cam_toggle.config(text="⏹  Tắt camera gesture", bg=C["bg_card"], fg=C["text_secondary"])
        else:
            self._cam_toggle.config(text="🔄  Thử lại camera", bg=C["amber"], fg="#000"); self._cam_active = False

    def _disable_gesture(self):
        self._cam_active = False;
        self._gesture_panel.stop();
        self._cam_toggle.config(text="▶  Bật camera gesture", bg=C["blue"], fg=C["text_primary"])

    def _run_bg(self, fn, btn, s, d):
        def worker():
            self.after(0, btn.set_state, "Đang chạy...", C["amber"]);
            self._log_box.log(s, "info")
            try:
                fn(); self.after(0, btn.set_state, "✓ Xong", C["green"]); self._log_box.log(d, "ok"); self.after(200,
                                                                                                                 self._startup_check)
            except Exception as e:
                self.after(0, btn.set_state, "✗ Lỗi", C["red"]); self._log_box.log(f"Lỗi: {e}", "err")

        threading.Thread(target=worker, daemon=True).start()

    def _register_face(self):
        n = self._name.get().strip()
        if not n:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập Tên thành viên trước!")
            return

        self._disable_gesture()  # Tắt cam tay

        def _do():
            # 1. PHÉP THUẬT NẰM Ở ĐÂY: Chờ 1.5s để Windows nhả hẳn Camera
            time.sleep(1.5)

            import importlib.util
            # 2. Tự động quét tìm đúng tên file AI của ông
            file_name = "ai_mat" if os.path.exists("ai_mat.py") else "face_ai_pro"
            spec = importlib.util.find_spec(file_name)

            if spec is None:
                raise Exception(f"Không tìm thấy file {file_name}.py!")

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.auto_collect_owner_data(n)

        self._run_bg(_do, self._steps[1], f"Quét mặt: {n}", f"Lưu xong dataset_{n}.csv")


    def _train_ai(self):
        def _do():
            import importlib.util
            file_name = "ai_mat" if os.path.exists("ai_mat.py") else "face_ai_pro"
            spec = importlib.util.find_spec(file_name)
            if spec is None: raise Exception("Thiếu file AI Khuôn mặt!")

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.train_svc_multi_class()

        self._run_bg(_do, self._steps[3], "Đang học dữ liệu AI (SVC)...", "Đã cập nhật face_ocsvm_model.pkl")
    def _open_door(self):
        self._disable_gesture()
        threading.Thread(target=lambda: subprocess.run(["python", "door_gui.py"]), daemon=True).start()

    def _on_close(self):
        self._disable_gesture(); self.destroy()


_hub_instance = None


def show_hub(owner_name: str = ""):
    global _hub_instance
    if _hub_instance and _hub_instance.winfo_exists():
        _hub_instance.deiconify();
        _hub_instance._owner = owner_name
        _hub_instance.after(0, _hub_instance._owner_lbl.config, {"text": f"  Xin chào, {owner_name}!"})
        return _hub_instance
    else:
        _hub_instance = HubApp(owner_name)
        _hub_instance.protocol("WM_DELETE_WINDOW", _hub_instance._on_close)
        return _hub_instance


if __name__ == "__main__":
    root = tk.Tk();
    root.withdraw()
    app = HubApp()
    app.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
    root.mainloop()