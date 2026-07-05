"""
door_gui.py — Giao diện Cửa Bảo Mật Smart Home
================================================
Tích hợp hoàn toàn: camera nhúng trong GUI, face verification
chạy ngầm, giả lập cửa mở/đóng, không cần terminal.

Luồng:
  Mở app → Camera tự bật → Bấm "Kích hoạt" → Face AI quét
  → Nhận ra chủ nhà → Hiện anti-spoofing challenge
  → Vượt qua → Cửa mở (animation) + bật Hand/AC module
  → Timeout hoặc người lạ → Cảnh báo + khoá cửa

Chạy: python door_gui.py
"""

import tkinter as tk
import threading
import time
import os
import random
import pickle
import subprocess
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

try:
    import pygame
    pygame.mixer.init()
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

# ─────────────────────── TOKENS ───────────────────────
DARK   = "#0a0c12"
PANEL  = "#12151f"
CARD   = "#181c28"
LINE   = "#252a3a"
LINE2  = "#2e3448"
BLUE   = "#4a8af4"
CYAN   = "#3ecfb8"
GREEN  = "#3ecf8e"
AMBER  = "#f5a623"
RED    = "#e05c5c"
WHITE  = "#eef0f8"
MUTED  = "#8890a8"
DIM    = "#454d6a"

FONT_HUD   = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_MONO  = ("Consolas",  9)
FONT_MED   = ("Segoe UI", 14, "bold")
MODEL_FILE = "face_ocsvm_model.pkl"

# ─────────────────────── FACE ENGINE ──────────────────
class FaceEngine:
    LEFT_EYE  = [362, 385, 386, 263, 374, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(self):
        mp_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4)
        self.model = None
        self.classes = []
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                d = pickle.load(f)
            self.model   = d["model"]
            self.classes = d["classes"]
            return True
        return False

    def process(self, rgb):
        r = self.face_mesh.process(rgb)
        if not r.multi_face_landmarks:
            return None, None
        lms    = r.multi_face_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lms])
        core   = coords[[33, 263, 1, 61, 291, 199, 4, 152, 10, 109]]
        base   = np.linalg.norm(coords[33] - coords[263]) or 1
        feats  = [np.linalg.norm(core[i]-core[j]) / base
                  for i in range(len(core))
                  for j in range(i+1, len(core))]
        return np.array(feats), lms

    def predict(self, feats):
        if self.model is None or feats is None:
            return None, 0.0
        probs = self.model.predict_proba([feats])[0]
        idx   = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def ear(self, lms, idx):
        p = lambda i: np.array([lms[i].x, lms[i].y])
        v = (np.linalg.norm(p(idx[1])-p(idx[5])) +
             np.linalg.norm(p(idx[2])-p(idx[4])))
        h = np.linalg.norm(p(idx[0])-p(idx[3]))
        return v / (2*h) if h else 0

    def turn_ratio(self, lms):
        L = abs(lms[4].x - lms[234].x)
        R = abs(lms[4].x - lms[454].x)
        return L / (R + 1e-6)

    def mouth_open(self, lms):
        eye_base = abs(lms[33].x - lms[263].x) or 1
        d = np.linalg.norm(
            np.array([lms[0].x, lms[0].y]) -
            np.array([lms[17].x, lms[17].y]))
        return d / eye_base


# ─────────────────────── DOOR GUI ─────────────────────
class DoorGUI(tk.Tk):
    STAGE_SLEEP      = "sleep"
    STAGE_SCAN       = "scan"
    STAGE_CHALLENGE  = "challenge"
    STAGE_GRANTED    = "granted"
    STAGE_DENIED     = "denied"

    CHALLENGES = {
        "BLINK":      "Nháy mắt liên tục",
        "TURN_LEFT":  "Quay mặt sang trái",
        "TURN_RIGHT": "Quay mặt sang phải",
        "OPEN_MOUTH": "Há miệng to",
    }
    DW, DH = 180, 280

    def __init__(self):
        super().__init__()
        self.title("Smart Door — Security Panel")
        self.configure(bg=DARK)
        self.resizable(False, False)

        self._engine        = FaceEngine()
        self._cap           = None
        self._running       = True
        self._stage         = self.STAGE_SLEEP
        self._owner         = "—"
        self._unk_count     = 0
        self._challenge     = None
        self._ch_ok         = 0
        self._ch_t0         = 0
        self._door_angle    = 0.0
        self._door_target   = 0.0
        self._img_tk        = None
        self._scan_frames   = 0

        self._build_ui()
        self._start_camera()
        self._animate_door()
        self._tick_clock()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(1200, self._wake_up)   # Tự kích hoạt khi mở app

    # ═══════════════ BUILD UI ═══════════════
    def _build_ui(self):
        # ── TOP BAR ──────────────────────────
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x")
        tk.Label(bar, text="  ⌂  SMART DOOR  —  Security Panel",
                 font=FONT_TITLE, fg=WHITE, bg=PANEL,
                 pady=10).pack(side="left")
        self._clock_var = tk.StringVar(value="--:--:--")
        tk.Label(bar, textvariable=self._clock_var,
                 font=("Segoe UI", 11), fg=MUTED, bg=PANEL,
                 padx=16).pack(side="right")

        # ── BODY ─────────────────────────────
        body = tk.Frame(self, bg=DARK, padx=14, pady=12)
        body.pack(fill="both", expand=True)

        self._build_camera_col(body)
        self._build_door_col(body)

        # ── STATUS BAR ───────────────────────
        sbar = tk.Frame(self, bg=PANEL,
                        highlightbackground=LINE, highlightthickness=1)
        sbar.pack(fill="x", side="bottom")
        inner = tk.Frame(sbar, bg=PANEL, padx=14, pady=7)
        inner.pack(fill="x")
        self._log_var = tk.StringVar(value="Hệ thống sẵn sàng.")
        tk.Label(inner, textvariable=self._log_var,
                 font=FONT_MONO, fg=MUTED, bg=PANEL).pack(side="left")
        self._match_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._match_var,
                 font=FONT_MONO, fg=DIM, bg=PANEL).pack(side="right")

    # ─── CAMERA COLUMN ───────────────────────
    def _build_camera_col(self, parent):
        col = tk.Frame(parent, bg=DARK)
        col.pack(side="left", fill="both", expand=True)

        # Header
        hdr = tk.Frame(col, bg=DARK)
        hdr.pack(fill="x", pady=(0, 6))
        tk.Label(hdr, text="CAMERA LIVE",
                 font=("Segoe UI", 8, "bold"), fg=DIM, bg=DARK).pack(side="left")
        self._cam_dot = tk.Label(hdr, text="● CONNECTING",
                                 font=("Segoe UI", 8, "bold"), fg=AMBER, bg=DARK)
        self._cam_dot.pack(side="right")

        # Camera frame
        cam_wrap = tk.Frame(col, bg=LINE,
                            highlightbackground=LINE2, highlightthickness=1)
        cam_wrap.pack()
        self._cam_lbl = tk.Label(cam_wrap, bg="#06080f", width=480, height=360)
        self._cam_lbl.pack()

        # Stage indicator (dưới camera)
        self._stage_bar = tk.Frame(col, bg=DIM, height=3)
        self._stage_bar.pack(fill="x", pady=(0, 10))

        # Identity card
        id_wrap = tk.Frame(col, bg=CARD,
                           highlightbackground=LINE2, highlightthickness=1)
        id_wrap.pack(fill="x")
        id_inner = tk.Frame(id_wrap, bg=CARD, padx=14, pady=10)
        id_inner.pack(fill="x")

        row1 = tk.Frame(id_inner, bg=CARD)
        row1.pack(fill="x")
        tk.Label(row1, text="NHẬN DIỆN",
                 font=("Segoe UI", 8, "bold"), fg=DIM, bg=CARD).pack(side="left")
        self._mode_lbl = tk.Label(row1, text="SLEEP",
                                  font=("Segoe UI", 8, "bold"), fg=DIM, bg=CARD)
        self._mode_lbl.pack(side="right")

        self._id_name = tk.Label(id_inner, text="—",
                                 font=FONT_MED, fg=WHITE, bg=CARD, anchor="w")
        self._id_name.pack(fill="x", pady=(4, 0))
        self._id_sub = tk.Label(id_inner, text="Đang chờ...",
                                font=("Segoe UI", 10), fg=MUTED, bg=CARD, anchor="w")
        self._id_sub.pack(fill="x")

        # Confidence bar
        bar_bg = tk.Frame(id_inner, bg=LINE, height=4)
        bar_bg.pack(fill="x", pady=(8, 0))
        bar_bg.pack_propagate(False)
        self._conf_bar = tk.Frame(bar_bg, bg=DIM, height=4)
        self._conf_bar.place(x=0, y=0, relheight=1, width=0)

        # Challenge card
        ch_wrap = tk.Frame(col, bg=CARD,
                           highlightbackground=LINE2, highlightthickness=1)
        ch_wrap.pack(fill="x", pady=(8, 0))
        ch_inner = tk.Frame(ch_wrap, bg=CARD, padx=14, pady=10)
        ch_inner.pack(fill="x")
        tk.Label(ch_inner, text="ANTI-SPOOFING",
                 font=("Segoe UI", 8, "bold"), fg=DIM, bg=CARD).pack(anchor="w")
        self._ch_lbl = tk.Label(ch_inner, text="—",
                                font=FONT_BOLD, fg=MUTED, bg=CARD, anchor="w")
        self._ch_lbl.pack(fill="x", pady=(4, 0))
        self._ch_canvas = tk.Canvas(ch_inner, height=6, bg=LINE,
                                    highlightthickness=0)
        self._ch_canvas.pack(fill="x", pady=(8, 0))

    # ─── DOOR COLUMN ─────────────────────────
    def _build_door_col(self, parent):
        col = tk.Frame(parent, bg=DARK, padx=14)
        col.pack(side="right", fill="y")

        tk.Label(col, text="TRẠNG THÁI CỬA",
                 font=("Segoe UI", 8, "bold"), fg=DIM, bg=DARK,
                 pady=(0)).pack(anchor="w", pady=(0, 6))

        # Canvas cửa
        door_bg = tk.Frame(col, bg=CARD,
                           highlightbackground=LINE2, highlightthickness=1)
        door_bg.pack()
        self._door_canvas = tk.Canvas(door_bg,
                                      width=self.DW+80,
                                      height=self.DH+80,
                                      bg=CARD, highlightthickness=0)
        self._door_canvas.pack(padx=16, pady=16)
        self._draw_door(0)

        # Buttons
        tk.Frame(col, bg=DARK, height=16).pack()

        self._wake_btn = tk.Button(
            col, text="◉  Kích hoạt quét mặt",
            font=FONT_BOLD, fg=WHITE,
            bg=BLUE, activebackground="#3070d0",
            relief="flat", cursor="hand2",
            padx=12, pady=10,
            command=self._wake_up)
        self._wake_btn.pack(fill="x")

        tk.Frame(col, bg=DARK, height=6).pack()

        self._manual_btn = tk.Button(
            col, text="🔓  Mở cửa thủ công",
            font=FONT_HUD, fg=MUTED,
            bg=CARD, activebackground=LINE2,
            relief="flat", cursor="hand2",
            padx=12, pady=7,
            highlightbackground=LINE2, highlightthickness=1,
            command=self._manual_open)
        self._manual_btn.pack(fill="x")

        tk.Frame(col, bg=DARK, height=6).pack()

        self._lock_btn = tk.Button(
            col, text="🔒  Khoá cửa",
            font=FONT_HUD, fg=MUTED,
            bg=CARD, activebackground=LINE2,
            relief="flat", cursor="hand2",
            padx=12, pady=7,
            highlightbackground=LINE2, highlightthickness=1,
            command=self._lock_door)
        self._lock_btn.pack(fill="x")

        # Info nhỏ
        tk.Frame(col, bg=DARK, height=16).pack()
        info = tk.Frame(col, bg=CARD,
                        highlightbackground=LINE2, highlightthickness=1)
        info.pack(fill="x")
        info_inner = tk.Frame(info, bg=CARD, padx=12, pady=8)
        info_inner.pack(fill="x")

        self._info_model = self._info_row(info_inner, "Model")
        self._info_ac    = self._info_row(info_inner, "Smart AC")
        self._info_hand  = self._info_row(info_inner, "Hand module")

        has_model = os.path.exists(MODEL_FILE)
        self._info_model.config(
            text="✓ Sẵn sàng" if has_model else "✗ Chưa train",
            fg=GREEN if has_model else AMBER)
        self._info_ac.config(text="✓ Tích hợp Smart Hub", fg=GREEN)
        self._info_hand.config(
            text="✓ test_hand.py" if os.path.exists("test_hand.py") else "Không tìm thấy",
            fg=GREEN if os.path.exists("test_hand.py") else MUTED)

    def _info_row(self, parent, key):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=key, font=("Segoe UI", 9), fg=DIM,
                 bg=CARD, width=14, anchor="w").pack(side="left")
        val = tk.Label(row, text="—", font=("Segoe UI", 9),
                       fg=MUTED, bg=CARD, anchor="e")
        val.pack(side="right")
        return val

    # ═══════════════ DOOR CANVAS ════════════
    def _draw_door(self, ratio):
        c  = self._door_canvas
        cw = self.DW + 80
        ch = self.DH + 80
        ox, oy = 40, 30
        dw, dh = self.DW, self.DH
        c.delete("all")

        # Tường
        c.create_rectangle(0, 0, cw, ch, fill="#14161e", outline="")

        # Khung cửa
        fw = 8
        c.create_rectangle(ox-fw, oy-fw, ox+dw+fw, oy+dh+fw,
                            fill="#1e2130", outline=LINE2, width=1)

        # Tấm cửa (co lại khi mở)
        pw = int(dw * (1 - ratio * 0.88))
        if pw > 6:
            shade = max(0, int(40 - ratio * 30))
            door_col = f"#{shade+30:02x}{shade+35:02x}{shade+55:02x}"
            c.create_rectangle(ox, oy, ox+pw, oy+dh,
                                fill=door_col, outline=LINE2, width=1)
            # Ô kính
            if pw > 45:
                mg = 14
                gw = pw - mg*2
                gh = int(dh * 0.32)
                c.create_rectangle(ox+mg, oy+mg, ox+mg+gw, oy+mg+gh,
                                   fill="#090d18", outline=LINE2, width=1)
                # Phản sáng kính
                if gw > 20:
                    c.create_line(ox+mg+4, oy+mg+4,
                                  ox+mg+gw//4, oy+mg+4,
                                  fill="#1a2233", width=2)
            # Tay nắm
            if pw > 32:
                hx = ox + pw - 16
                hy = oy + dh // 2
                c.create_oval(hx-5, hy-5, hx+5, hy+5,
                              fill="#3a4060", outline=LINE2)

        # Khe tối khi cửa mở
        if ratio > 0.04:
            gap_x1 = ox + pw if pw > 6 else ox
            gap_x2 = ox + int(dw * ratio * 0.88)
            if gap_x2 > gap_x1:
                c.create_rectangle(gap_x1, oy, gap_x2+2, oy+dh,
                                   fill="#04050a", outline="")

        # Label trạng thái
        if   ratio < 0.08:  txt, col = "ĐÓNG", RED
        elif ratio > 0.85:  txt, col = "MỞ",   GREEN
        else:               txt, col = "...",   AMBER
        c.create_text(ox + dw//2, oy + dh + 24,
                      text=txt, fill=col,
                      font=("Segoe UI", 9, "bold"))

    # ═══════════════ ANIMATE ════════════════
    def _animate_door(self):
        if not self._running:
            return
        step = 0.055
        if   self._door_angle < self._door_target:
            self._door_angle = min(self._door_angle + step, self._door_target)
        elif self._door_angle > self._door_target:
            self._door_angle = max(self._door_angle - step, self._door_target)
        self._draw_door(self._door_angle)
        self.after(28, self._animate_door)

    def _tick_clock(self):
        self._clock_var.set(datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ═══════════════ CAMERA LOOP ════════════
    def _start_camera(self):
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        while self._running:
            if not (self._cap and self._cap.isOpened()):
                time.sleep(0.1); continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05); continue

            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=20)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats, lms = self._engine.process(rgb)

            # Vẽ overlay
            self._overlay(frame, feats, lms)

            # Logic nhận dạng
            if self._stage not in (self.STAGE_GRANTED, self.STAGE_SLEEP):
                self._recognize(feats, lms)

            # Đếm scan_frames
            if self._stage == self.STAGE_SCAN:
                self._scan_frames -= 1
                if self._scan_frames <= 0:
                    self.after(0, self._go_sleep)

            # Hiển thị
            rgb_resized = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (480, 360))
            try:
                if self.winfo_exists():
                    self.after(0, self._show_frame,
                               rgb_resized.tobytes(), 480, 360)
            except Exception:
                pass
            time.sleep(0.033)

    def _show_frame(self, raw_bytes, w, h):
        """Main thread only — tạo PhotoImage ở đây."""
        if not self._running:
            return
        try:
            img    = Image.frombytes("RGB", (w, h), raw_bytes)
            tk_img = ImageTk.PhotoImage(image=img)
            self._img_tk = tk_img          # PHẢI giữ reference trước khi config
            self._cam_lbl.config(image=self._img_tk)
            self._cam_dot.config(text="● LIVE", fg=GREEN)
        except Exception as e:
            pass

    def _overlay(self, frame, feats, lms):
        h, w = frame.shape[:2]
        col_bgr = {
            self.STAGE_SCAN:      (244, 138, 74),
            self.STAGE_CHALLENGE: (35, 166, 245),
            self.STAGE_GRANTED:   (142, 207, 62),
            self.STAGE_DENIED:    (92,  92, 224),
            self.STAGE_SLEEP:     (106, 77,  69),
        }.get(self._stage, (106, 77, 69))

        t, L = 3, 30
        for (cx, cy), (dx, dy) in zip(
            [(0,0),(w-L,0),(0,h-L),(w-L,h-L)],
            [(1,1),(-1,1),(1,-1),(-1,-1)]
        ):
            cv2.line(frame, (cx,cy), (cx+dx*L, cy), col_bgr, t)
            cv2.line(frame, (cx,cy), (cx, cy+dy*L), col_bgr, t)

        if lms:
            # Vẽ contour mặt nhẹ
            ih, iw = frame.shape[:2]
            pts = [(int(lms[i].x*iw), int(lms[i].y*ih))
                   for i in [10,338,297,332,284,251,389,356,454,323,361,
                              288,397,365,379,378,400,377,152,148,176,149,
                              150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]]
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], col_bgr, 1)

    # ═══════════════ RECOGNITION ════════════
    def _recognize(self, feats, lms):
        if self._stage == self.STAGE_SCAN:
            if feats is None:
                self._unk_count += 1
                self.after(0, self._id_sub.config, {"text": "Không thấy mặt...", "fg": MUTED})
                return

            if not self._engine.model:
                self.after(0, self._id_sub.config,
                           {"text": "Chưa có model — cần train AI", "fg": AMBER})
                return

            name, prob = self._engine.predict(feats)
            self.after(0, self._match_var.set, f"match {prob*100:.1f}%")
            self.after(0, self._update_conf_bar, prob)

            if prob > 0.70 and name != "Unknown":
                self._owner     = name
                self._unk_count = 0
                self._stage     = self.STAGE_CHALLENGE
                self._challenge = random.choice(list(self.CHALLENGES))
                self._ch_ok     = 0
                self._ch_t0     = time.time()
                self.after(0, self._on_matched, name, prob)

            elif name == "Unknown" or prob < 0.55:
                self._unk_count += 1
                self.after(0, self._id_sub.config,
                           {"text": f"Không nhận ra  ({prob*100:.0f}%)", "fg": MUTED})
                if self._unk_count > 90:
                    self.after(0, self._on_stranger)
            else:
                self.after(0, self._id_sub.config,
                           {"text": "Đang xác định...", "fg": AMBER})

        elif self._stage == self.STAGE_CHALLENGE:
            if lms is None:
                return
            elapsed = time.time() - self._ch_t0
            if elapsed > 12:
                self.after(0, self._on_timeout); return

            avg_ear = (self._engine.ear(lms, self._engine.LEFT_EYE) +
                       self._engine.ear(lms, self._engine.RIGHT_EYE)) / 2
            turn  = self._engine.turn_ratio(lms)
            mouth = self._engine.mouth_open(lms)

            ok = (self._challenge == "BLINK"      and avg_ear < 0.35  or
                  self._challenge == "TURN_LEFT"  and turn    < 0.45  or
                  self._challenge == "TURN_RIGHT" and turn    > 2.2   or
                  self._challenge == "OPEN_MOUTH" and mouth   > 0.55)

            self._ch_ok = max(0, self._ch_ok + (1 if ok else -1))
            self.after(0, self._update_ch_bar, self._ch_ok/6, elapsed/12)

            if self._ch_ok >= 6:
                self.after(0, self._on_granted)

    # ═══════════════ TRANSITIONS ════════════
    def _on_matched(self, name, prob):
        self._id_name.config(text=name, fg=CYAN)
        self._id_sub.config(text=f"Xác nhận {prob*100:.0f}% — kiểm tra liveness", fg=CYAN)
        self._ch_lbl.config(
            text=self.CHALLENGES[self._challenge], fg=AMBER)
        self._set_mode("CHALLENGE", AMBER)
        self._stage_bar.config(bg=AMBER)
        self._log(f"Nhận ra {name} ({prob*100:.0f}%) — anti-spoofing bắt đầu")

    def _on_granted(self):
        self._stage = self.STAGE_GRANTED
        self._door_target = 1.0
        self._id_name.config(fg=GREEN)
        self._id_sub.config(text=f"Xin chào, {self._owner}! Đang mở cửa...", fg=GREEN)
        self._ch_lbl.config(text="✓ Liveness xác nhận", fg=GREEN)
        self._set_mode("GRANTED", GREEN)
        self._stage_bar.config(bg=GREEN)
        self._log(f"✓ XÁC THỰC THÀNH CÔNG — {self._owner}")
        self._play("chao_sep.mp3")
        with open("current_owner.txt", "w", encoding="utf-8") as f:
            f.write(self._owner)
        # Sau 2.5s: nhả camera → tắt cửa sổ → bật module khác
        self.after(2500, self._handoff)

    def _on_stranger(self):
        self._stage = self.STAGE_DENIED
        self._unk_count = 0
        self._id_name.config(text="NGƯỜI LẠ", fg=RED)
        self._id_sub.config(text="Truy cập bị từ chối — cửa khoá!", fg=RED)
        self._set_mode("DENIED", RED)
        self._stage_bar.config(bg=RED)
        self._log("⚠ Phát hiện người lạ — cảnh báo!")
        self._play("bao_dong.mp3")
        self.after(4000, self._go_sleep)

    def _on_timeout(self):
        self._stage = self.STAGE_SCAN
        self._ch_ok = 0
        self._ch_lbl.config(text="Hết thời gian — thử lại", fg=RED)
        self._log("Anti-spoofing: timeout, quay lại scan")
        self.after(1500, lambda: self._ch_lbl.config(text="—", fg=MUTED))

    def _go_sleep(self):
        self._stage   = self.STAGE_SLEEP
        self._owner   = "—"
        self._unk_count = 0
        self._ch_ok   = 0
        self._id_name.config(text="—", fg=WHITE)
        self._id_sub.config(text="Đang chờ...", fg=MUTED)
        self._ch_lbl.config(text="—", fg=MUTED)
        self._set_mode("SLEEP", DIM)
        self._stage_bar.config(bg=DIM)
        self._match_var.set("")
        self._update_conf_bar(0)
        self._update_ch_bar(0, 0)
        self._log("Hệ thống về chế độ chờ")

    # ═══════════════ MANUAL BUTTONS ═════════
    def _wake_up(self):
        if self._stage == self.STAGE_GRANTED:
            return
        self._stage = self.STAGE_SCAN
        self._unk_count   = 0
        self._scan_frames = 30 * 30   # 30 giây
        self._id_name.config(text="Đang quét...", fg=AMBER)
        self._id_sub.config(text="Camera đang nhận diện khuôn mặt", fg=AMBER)
        self._set_mode("SCANNING", AMBER)
        self._stage_bar.config(bg=AMBER)
        self._log("Kích hoạt quét mặt")
        self._play("dang_quet.mp3")

    def _manual_open(self):
        self._door_target = 1.0
        self._stage = self.STAGE_GRANTED
        self._id_sub.config(text="Mở thủ công", fg=AMBER)
        self._stage_bar.config(bg=AMBER)
        self._log("Mở cửa thủ công")

    def _lock_door(self):
        self._door_target = 0.0
        if self._stage == self.STAGE_GRANTED:
            self._go_sleep()
        self._log("Cửa đã khoá")

    def _handoff(self):
        """
        1. Dừng camera & Ẩn Door GUI
        2. Dùng after() chờ 1.5s để cam tắt hẳn rồi gọi Hub
        """
        owner = self._owner
        self._log(f"Nhả camera — chuyển sang Smart Hub cho {owner}...")

        # ── 1. Dừng camera (Không dùng sleep để tránh đơ app) ──
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None

        # ── 2. Ẩn Door GUI ──
        self.withdraw()

        # ── 3. Bật Smart AC ngầm (vẫn xài Thread vì không đụng UI) ──
        threading.Thread(target=self._run_ac, args=(owner,), daemon=True).start()

        # ── 4. Chờ 1.5s rồi gọi Hub (đảm bảo Camera đã được giải phóng) ──
        self.after(1500, self._launch_hub_safe, owner)

    def _launch_hub_safe(self, owner):
        """Mở Hub trên Main Thread và bắt Cửa phải chờ"""
        try:
            from smart_hub_gui import show_hub
            hub_win = show_hub(owner)

            # ĐÂY LÀ PHÉP THUẬT: Lệnh này bắt DoorGUI "đóng băng"
            # chờ đến khi nào cái hub_win bị bấm nút X tắt đi thì mới chạy tiếp!
            if hub_win:
                self.wait_window(hub_win)

        except Exception as e:
            self._log(f"Hub lỗi: {e}")

        # ── 5. Hub đã đóng → Phục hồi Door GUI ──
        self._restore_door()


    def _run_ac(self, owner):
        """Chạy Smart AC hoàn toàn ngầm."""
        return
        try:
            from ac_ai import run_smart_ac
            run_smart_ac(owner)
        except Exception as e:
            self.after(0, self._log, f"AC module lỗi: {e}")

    def _restore_door(self):
        """Hiện lại Door GUI và khởi động lại camera để quét lần sau."""
        self._log("Hand đã đóng — Door GUI trở lại hoạt động")
        self.deiconify()               # hiện lại cửa sổ

        # Khởi động lại camera
        self._running = True
        self._cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        threading.Thread(target=self._cam_loop, daemon=True).start()

        # Reset về trạng thái chờ và tự kích hoạt quét
        self._go_sleep()
        self.after(1000, self._wake_up)

    # ═══════════════ UI HELPERS ═════════════
    def _set_mode(self, text, color):
        self._mode_lbl.config(text=text, fg=color)

    def _update_conf_bar(self, ratio):
        ratio = max(0.0, min(1.0, ratio))
        w = max(self._conf_bar.master.winfo_width(), 200)
        bw = int(w * ratio)
        col = GREEN if ratio > 0.7 else (AMBER if ratio > 0.45 else RED)
        self._conf_bar.place(x=0, y=0, width=bw, relheight=1)
        self._conf_bar.config(bg=col)

    def _update_ch_bar(self, prog, time_r):
        c = self._ch_canvas
        cw = max(c.winfo_width(), 200)
        ch = 6
        c.delete("all")
        c.create_rectangle(0, 0, cw, ch, fill=LINE, outline="")
        if prog > 0:
            c.create_rectangle(0, 0, int(cw * prog), ch, fill=GREEN, outline="")
        if time_r > 0.7:
            c.create_rectangle(int(cw*time_r), 0, cw, ch, fill=RED, outline="")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}]  {msg}"
        self._log_var.set(full)
        print(f"[DOOR] {full}")

    def _play(self, fn):
        if HAS_AUDIO and os.path.exists(fn):
            try:
                pygame.mixer.music.load(fn)
                pygame.mixer.music.play()
            except Exception:
                pass

    def _on_close(self):
        self._running = False
        time.sleep(0.1)   # chờ cam_loop thoát
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self.destroy()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    DoorGUI().mainloop()