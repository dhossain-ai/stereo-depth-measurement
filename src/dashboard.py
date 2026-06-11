"""
Traffic Vehicle Counter — Interactive Dashboard
================================================
Entry point:  py src/dashboard.py
Two-phase workflow:
  Phase 1 — Setup : browse frames, draw ROI, place counting line, tune params
  Phase 2 — Run   : process full video with configured settings

Features:
  • Interactive ROI polygon drawing  (left-click = add point, right-click = close)
  • Draggable counting line          (click near line + drag vertically)
  • Parameter tuning sliders         (MOG2, morphology, contour, tracker)
  • HOG + LinearSVC pipeline         (two-pass: collect samples → train → detect)
  • Centroid tracking + line crossing
  • Real-time preview during processing
  • Annotated video + count log export
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.feature import hog as compute_hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance as dist
from collections import OrderedDict
import threading
import os


# ================================================================
#  CENTROID TRACKER
# ================================================================
class CentroidTracker:
    """Simple centroid-based multi-object tracker."""

    def __init__(self, max_disappeared=15, max_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, rects):
        """Update with list of (x, y, w, h) bounding boxes."""
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        centroids = np.array([
            (int(x + w / 2), int(y + h / 2)) for x, y, w, h in rects
        ])

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return self.objects

        ids = list(self.objects.keys())
        prev = np.array(list(self.objects.values()))
        D = dist.cdist(prev, centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = ids[r]
            self.objects[oid] = centroids[c]
            self.disappeared[oid] = 0
            used_r.add(r)
            used_c.add(c)

        for r in set(range(D.shape[0])) - used_r:
            oid = ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(D.shape[1])) - used_c:
            self.register(centroids[c])

        return self.objects


# ================================================================
#  DASHBOARD
# ================================================================
class Dashboard:
    DISPLAY_WIDTH = 960
    LINE_GRAB_PX = 12

    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Vehicle Counter — Dashboard")

        # ── video state ─────────────────────────────────────────
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.vid_w = self.vid_h = 0
        self.current_idx = 0
        self.frame_raw = None
        self.display_scale = 1.0
        self.display_w = self.DISPLAY_WIDTH
        self.display_h = 0
        self._photo = None
        self._slider_busy = False

        # ── ROI state ───────────────────────────────────────────
        self.roi_points = []
        self.roi_closed = False
        self.mode = "idle"

        # ── counting line state ─────────────────────────────────
        self.count_line_y = None
        self._dragging_line = False

        # ── parameters ──────────────────────────────────────────
        self.params = dict(
            mog2_history=500,
            mog2_threshold=16,
            morph_kernel=5,
            min_contour_area=1500,
            large_vehicle_area=5000,
            tracker_max_disappeared=15,
            tracker_max_distance=150,
        )
        self._param_sliders = {}

        # ── pipeline state ──────────────────────────────────────
        self.running = False
        self._cancel = False
        self._progress = 0.0
        self._status_msg = ""
        self._preview = None
        self._done_flag = False
        self.results = dict(large=0, small=0, total=0)
        self.output_video_path = ""

        # ── build UI → load video → show first frame ───────────
        self._build_layout()
        self._load_video()
        if self.cap:
            self.count_line_y = int(self.display_h * 0.55)
        self._show_frame(0)

    # ============================================================
    #  LAYOUT
    # ============================================================
    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── LEFT PANEL ──────────────────────────────────────────
        left = ttk.Frame(self.root, width=280, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)

        ttk.Label(left, text="Traffic Vehicle Counter",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.phase_var = tk.StringVar(value="Phase 1 — Setup")
        ttk.Label(left, textvariable=self.phase_var,
                  font=("Segoe UI", 10, "italic")).pack(anchor="w", pady=(0, 8))

        # navigation
        nav = ttk.LabelFrame(left, text="Frame Navigation", padding=6)
        nav.pack(fill="x", pady=(0, 6))
        self.frame_slider = ttk.Scale(nav, from_=0, to=1,
                                      orient="horizontal",
                                      command=self._on_slider)
        self.frame_slider.pack(fill="x", pady=(0, 2))
        self.frame_lbl = ttk.Label(nav, text="Frame 0 / 0")
        self.frame_lbl.pack(anchor="w")
        self.time_lbl = ttk.Label(nav, text="Time 0.00 / 0.00 s")
        self.time_lbl.pack(anchor="w", pady=(0, 4))
        btn_row = ttk.Frame(nav)
        btn_row.pack(fill="x")
        for txt, d in [("⏮-10", -10), ("◀-1", -1), ("+1▶", 1), ("+10⏭", 10)]:
            ttk.Button(btn_row, text=txt, width=6,
                       command=lambda d=d: self._step(d)).pack(
                           side="left", padx=1, expand=True)

        # ROI controls
        roi_sec = ttk.LabelFrame(left, text="ROI Polygon", padding=6)
        roi_sec.pack(fill="x", pady=(0, 6))
        roi_btns = ttk.Frame(roi_sec)
        roi_btns.pack(fill="x")
        self.roi_draw_btn = ttk.Button(roi_btns, text="✏ Draw ROI",
                                       command=self._toggle_roi_mode)
        self.roi_draw_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.roi_clear_btn = ttk.Button(roi_btns, text="✖ Clear",
                                        command=self._clear_roi)
        self.roi_clear_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.roi_info = ttk.Label(roi_sec, text="No ROI drawn (full frame)",
                                  foreground="gray")
        self.roi_info.pack(anchor="w", pady=(4, 0))

        # counting line
        line_sec = ttk.LabelFrame(left, text="Counting Line", padding=6)
        line_sec.pack(fill="x", pady=(0, 6))
        self.line_info = ttk.Label(line_sec,
                                   text="Drag the red line on the video")
        self.line_info.pack(anchor="w")

        # parameters
        param_sec = ttk.LabelFrame(left, text="Parameters", padding=6)
        param_sec.pack(fill="x", pady=(0, 6))
        for label, key, lo, hi, default in [
            ("MOG2 history",       "mog2_history",             100, 1000, 500),
            ("MOG2 threshold",     "mog2_threshold",            8,   50,  16),
            ("Morph kernel",       "morph_kernel",               3,   15,   5),
            ("Min contour area",   "min_contour_area",         500, 15000, 1500),
            ("Large vehicle area", "large_vehicle_area",       1000, 20000, 5000),
            ("Track max lost",     "tracker_max_disappeared",    3,   40,  15),
            ("Track max dist",     "tracker_max_distance",      30,  400, 150),
        ]:
            self._make_slider(param_sec, label, key, lo, hi, default)

        # results
        res_sec = ttk.LabelFrame(left, text="Results", padding=6)
        res_sec.pack(fill="x", pady=(0, 6))
        self.result_lbl = ttk.Label(res_sec, text="—",
                                    font=("Consolas", 10))
        self.result_lbl.pack(anchor="w")

        # action buttons
        self.run_btn = ttk.Button(left, text="▶  Run Pipeline",
                                  command=self._on_run)
        self.run_btn.pack(fill="x", pady=(2, 2))
        self.export_btn = ttk.Button(left, text="💾  Save Count Log",
                                     command=self._export_log,
                                     state="disabled")
        self.export_btn.pack(fill="x", pady=(0, 2))

        # ── RIGHT PANEL ─────────────────────────────────────────
        right = ttk.Frame(self.root, padding=4)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        # ── BOTTOM BAR ──────────────────────────────────────────
        bottom = ttk.Frame(self.root)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(bottom, mode="determinate",
                                        maximum=100)
        self.progress.pack(fill="x", side="top")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var,
                  relief="sunken", padding=(6, 3)).pack(fill="x")

    def _make_slider(self, parent, label, key, lo, hi, default):
        frm = ttk.Frame(parent)
        frm.pack(fill="x", pady=1)
        lbl = ttk.Label(frm, text=f"{label}: {default}", width=30,
                        anchor="w", font=("Segoe UI", 8))
        lbl.pack(anchor="w")

        def cb(v, _l=lbl, _la=label, _k=key):
            val = int(float(v))
            _l.configure(text=f"{_la}: {val}")
            self.params[_k] = val

        s = ttk.Scale(frm, from_=lo, to=hi, orient="horizontal",
                      command=cb, value=default)
        s.pack(fill="x")
        self._param_sliders[key] = (s, lbl)

    # ============================================================
    #  VIDEO LOADING
    # ============================================================
    def _load_video(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.video_path = os.path.join(base, "data", "moving_traffic.mp4")
        if not os.path.exists(self.video_path):
            self.status_var.set(f"ERROR — video not found: {self.video_path}")
            return
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.status_var.set("ERROR — could not open video")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_scale = self.DISPLAY_WIDTH / self.vid_w
        self.display_w = self.DISPLAY_WIDTH
        self.display_h = int(self.vid_h * self.display_scale)
        self.frame_slider.configure(to=max(self.total_frames - 1, 1))
        self.status_var.set(
            f"Video: {self.vid_w}×{self.vid_h} | "
            f"{self.total_frames} frames @ {self.fps:.0f} fps | "
            f"Display: {self.display_w}×{self.display_h}")

    # ============================================================
    #  FRAME DISPLAY + OVERLAY
    # ============================================================
    def _show_frame(self, idx):
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_raw = frame
        self._render_display(frame)
        self.frame_lbl.configure(text=f"Frame {idx} / {self.total_frames - 1}")
        t = idx / self.fps
        self.time_lbl.configure(
            text=f"Time {t:.2f} / {self.total_frames / self.fps:.2f} s")
        self._slider_busy = True
        self.frame_slider.set(idx)
        self._slider_busy = False

    def _render_display(self, frame_bgr):
        disp = cv2.resize(frame_bgr, (self.display_w, self.display_h),
                          interpolation=cv2.INTER_AREA)
        if self.roi_points:
            pts = np.array(self.roi_points, np.int32)
            if self.roi_closed:
                overlay = disp.copy()
                cv2.fillPoly(overlay, [pts], (255, 255, 0))
                cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)
                cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(disp, tuple(pts[i]), tuple(pts[i + 1]),
                             (0, 255, 255), 2)
            for p in self.roi_points:
                cv2.circle(disp, p, 5, (0, 255, 255), -1)

        if self.count_line_y is not None:
            y = self.count_line_y
            cv2.line(disp, (0, y), (self.display_w, y), (0, 0, 255), 2)
            cv2.putText(disp, "COUNTING LINE", (10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.configure(width=self.display_w, height=self.display_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _refresh_overlay(self):
        if self.frame_raw is not None:
            self._render_display(self.frame_raw)

    # ============================================================
    #  NAVIGATION
    # ============================================================
    def _step(self, delta):
        if not self.running:
            self._show_frame(self.current_idx + delta)

    def _on_slider(self, value):
        if self._slider_busy or self.running:
            return
        idx = int(float(value))
        if idx != self.current_idx:
            self._show_frame(idx)

    # ============================================================
    #  ROI DRAWING
    # ============================================================
    def _toggle_roi_mode(self):
        if self.mode == "roi_draw":
            self.mode = "idle"
            self.roi_draw_btn.configure(text="✏ Draw ROI")
            self.status_var.set("ROI drawing off")
        else:
            self.roi_points.clear()
            self.roi_closed = False
            self.mode = "roi_draw"
            self.roi_draw_btn.configure(text="⏹ Stop Drawing")
            self.status_var.set(
                "Left-click to add points · Right-click to close polygon")
        self._update_roi_info()
        self._refresh_overlay()

    def _clear_roi(self):
        self.roi_points.clear()
        self.roi_closed = False
        self.mode = "idle"
        self.roi_draw_btn.configure(text="✏ Draw ROI")
        self._update_roi_info()
        self._refresh_overlay()
        self.status_var.set("ROI cleared")

    def _update_roi_info(self):
        n = len(self.roi_points)
        if n == 0:
            self.roi_info.configure(text="No ROI drawn (full frame)",
                                    foreground="gray")
        elif self.roi_closed:
            self.roi_info.configure(text=f"ROI closed — {n} vertices ✓",
                                    foreground="green")
        else:
            self.roi_info.configure(text=f"Drawing… {n} point(s)",
                                    foreground="orange")

    # ============================================================
    #  CANVAS MOUSE EVENTS
    # ============================================================
    def _on_left_click(self, event):
        if self.running:
            return
        x, y = event.x, event.y
        if self.mode == "roi_draw" and not self.roi_closed:
            self.roi_points.append((x, y))
            self._update_roi_info()
            self._refresh_overlay()
            return
        if (self.count_line_y is not None
                and abs(y - self.count_line_y) <= self.LINE_GRAB_PX):
            self._dragging_line = True
            self.canvas.configure(cursor="sb_v_double_arrow")

    def _on_right_click(self, event):
        if self.running:
            return
        if self.mode == "roi_draw" and len(self.roi_points) >= 3:
            self.roi_closed = True
            self.mode = "idle"
            self.roi_draw_btn.configure(text="✏ Draw ROI")
            self._update_roi_info()
            self._refresh_overlay()
            self.status_var.set("ROI polygon closed ✓")

    def _on_mouse_drag(self, event):
        if self._dragging_line:
            self.count_line_y = max(10, min(event.y, self.display_h - 10))
            pct = self.count_line_y / self.display_h * 100
            self.line_info.configure(text=f"Line at {pct:.0f}% height")
            self._refresh_overlay()

    def _on_mouse_release(self, event):
        if self._dragging_line:
            self._dragging_line = False
            self.canvas.configure(cursor="")

    # ============================================================
    #  PIPELINE — RUN
    # ============================================================
    def _on_run(self):
        if self.running:
            self._cancel = True
            self.status_var.set("Cancelling…")
            return

        roi_msg = (f"{len(self.roi_points)}-vertex polygon"
                   if self.roi_closed else "none (full frame)")
        line_pct = (f"{self.count_line_y / self.display_h * 100:.0f}%"
                    if self.count_line_y else "55% default")
        if not messagebox.askyesno(
                "Run Pipeline",
                f"ROI: {roi_msg}\n"
                f"Counting line: {line_pct}\n\nStart processing?"):
            return

        self.running = True
        self._cancel = False
        self._done_flag = False
        self._progress = 0.0
        self.results = dict(large=0, small=0, total=0)
        self.run_btn.configure(text="⏹  Cancel")
        self.export_btn.configure(state="disabled")
        self.phase_var.set("Phase 2 — Running")
        self._toggle_controls(False)

        threading.Thread(target=self._run_pipeline, daemon=True).start()
        self.root.after(60, self._poll_pipeline)

    def _toggle_controls(self, enabled):
        st = "normal" if enabled else "disabled"
        self.roi_draw_btn.configure(state=st)
        self.roi_clear_btn.configure(state=st)
        for s, _ in self._param_sliders.values():
            s.configure(state=st)

    def _poll_pipeline(self):
        preview = self._preview
        if preview is not None:
            self._preview = None
            disp = cv2.resize(preview, (self.display_w, self.display_h),
                              interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self.progress["value"] = self._progress * 100
        self.status_var.set(self._status_msg)
        r = self.results
        self.result_lbl.configure(
            text=f"Large: {r['large']}  |  Cars: {r['small']}"
                 f"  |  Total: {r['total']}")

        if self._done_flag:
            self._pipeline_finished()
        else:
            self.root.after(60, self._poll_pipeline)

    def _pipeline_finished(self):
        self.running = False
        self.run_btn.configure(text="▶  Run Pipeline")
        self.phase_var.set("Phase 1 — Setup")
        self._toggle_controls(True)
        if self._cancel:
            self.status_var.set("Pipeline cancelled")
        else:
            self.export_btn.configure(state="normal")
            r = self.results
            messagebox.showinfo(
                "Pipeline Complete",
                f"Large vehicles: {r['large']}\n"
                f"Cars / small:   {r['small']}\n"
                f"Total counted:  {r['total']}\n\n"
                f"Annotated video saved to:\n{self.output_video_path}")
        self._show_frame(0)

    # ============================================================
    #  PIPELINE — WORKER THREAD
    # ============================================================
    def _run_pipeline(self):
        try:
            self._pipeline_core()
        except Exception as e:
            self._status_msg = f"Error: {e}"
        finally:
            self._done_flag = True

    def _pipeline_core(self):
        scale = 1.0 / self.display_scale
        p = self.params

        # ROI mask at original resolution
        mask = np.zeros((self.vid_h, self.vid_w), dtype=np.uint8)
        if self.roi_closed and self.roi_points:
            roi_orig = [(int(x * scale), int(y * scale))
                        for x, y in self.roi_points]
            cv2.fillPoly(mask, [np.array(roi_orig, np.int32)], 255)
        else:
            roi_orig = []
            mask[:] = 255

        line_y = int((self.count_line_y
                      or int(self.display_h * 0.55)) * scale)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p["morph_kernel"], p["morph_kernel"]))
        total = self.total_frames

        # ========================================================
        #  PASS 1 — collect HOG training samples
        # ========================================================
        self._status_msg = "Pass 1 — collecting HOG training samples…"
        bg1 = cv2.createBackgroundSubtractorMOG2(
            history=p["mog2_history"],
            varThreshold=p["mog2_threshold"],
            detectShadows=True)

        samples, labels = [], []
        cap = cv2.VideoCapture(self.video_path)

        for i in range(total):
            if self._cancel:
                cap.release(); return
            ret, frame = cap.read()
            if not ret:
                break

            fg = bg1.apply(cv2.bitwise_and(frame, frame, mask=mask))
            fg = cv2.threshold(fg, 244, 255, cv2.THRESH_BINARY)[1]
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

            for cnt in cv2.findContours(
                    fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if area < p["min_contour_area"]:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                roi_img = frame[by:by+bh, bx:bx+bw]
                if roi_img.size == 0:
                    continue
                gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (128, 128))
                feat = compute_hog(resized, orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   feature_vector=True)
                samples.append(feat)
                labels.append(
                    1 if area >= p["large_vehicle_area"] else 0)

            self._progress = 0.40 * (i + 1) / total
        cap.release()

        # ========================================================
        #  TRAIN SVM
        # ========================================================
        self._status_msg = "Training HOG + LinearSVC…"
        self._progress = 0.42
        use_svm = True
        scaler = StandardScaler()
        clf = LinearSVC(max_iter=10000, dual="auto")

        X = np.array(samples) if samples else np.empty((0, 0))
        y_arr = np.array(labels)

        if len(X) < 20 or len(set(labels)) < 2:
            use_svm = False
            self._status_msg = ("Too few samples — "
                                "falling back to size-based classification")
        else:
            pos = np.where(y_arr == 1)[0]
            neg = np.where(y_arr == 0)[0]
            n = min(len(pos), len(neg), 500)
            if n < 10:
                use_svm = False
            else:
                rng = np.random.default_rng(42)
                sel = np.concatenate([
                    rng.choice(pos, n, replace=len(pos) < n),
                    rng.choice(neg, n, replace=len(neg) < n)])
                rng.shuffle(sel)
                X_t = scaler.fit_transform(X[sel])
                clf.fit(X_t, y_arr[sel])
                self._status_msg = f"SVM trained on {len(sel)} samples"

        self._progress = 0.45

        # ========================================================
        #  PASS 2 — detect + classify + track + count
        # ========================================================
        self._status_msg = "Pass 2 — detecting and counting…"
        bg2 = cv2.createBackgroundSubtractorMOG2(
            history=p["mog2_history"],
            varThreshold=p["mog2_threshold"],
            detectShadows=True)

        tracker = CentroidTracker(
            max_disappeared=p["tracker_max_disappeared"],
            max_distance=p["tracker_max_distance"])

        prev_cy = {}
        counted = set()
        obj_cls = {}
        large_cnt = small_cnt = 0

        # video writer
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base, "output")
        os.makedirs(out_dir, exist_ok=True)
        self.output_video_path = os.path.join(out_dir,
                                              "dashboard_result.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(self.output_video_path, fourcc,
                                 self.fps, (self.vid_w, self.vid_h))

        cap = cv2.VideoCapture(self.video_path)

        for i in range(total):
            if self._cancel:
                writer.release(); cap.release(); return
            ret, frame = cap.read()
            if not ret:
                break

            ann = frame.copy()
            fg = bg2.apply(cv2.bitwise_and(frame, frame, mask=mask))
            fg = cv2.threshold(fg, 244, 255, cv2.THRESH_BINARY)[1]
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

            rects, r_cls = [], []
            for cnt in cv2.findContours(
                    fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if area < p["min_contour_area"]:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                rects.append((bx, by, bw, bh))

                if use_svm:
                    roi_img = frame[by:by+bh, bx:bx+bw]
                    if roi_img.size == 0:
                        r_cls.append("small"); continue
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (128, 128))
                    feat = compute_hog(
                        resized, orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        feature_vector=True).reshape(1, -1)
                    pred = clf.predict(scaler.transform(feat))[0]
                    r_cls.append("large" if pred == 1 else "small")
                else:
                    r_cls.append(
                        "large" if area >= p["large_vehicle_area"]
                        else "small")

            objects = tracker.update(rects)

            # assign class to newly tracked objects
            for oid, ctr in objects.items():
                if oid not in obj_cls and rects:
                    ds = [abs(ctr[0] - (r[0]+r[2]//2))
                          + abs(ctr[1] - (r[1]+r[3]//2))
                          for r in rects]
                    ci = int(np.argmin(ds))
                    if ci < len(r_cls):
                        obj_cls[oid] = r_cls[ci]

            # counting line crossing (downward = toward camera)
            for oid, ctr in objects.items():
                cy = ctr[1]
                if oid in prev_cy:
                    if prev_cy[oid] <= line_y < cy and oid not in counted:
                        counted.add(oid)
                        c = obj_cls.get(oid, "small")
                        if c == "large":
                            large_cnt += 1
                        else:
                            small_cnt += 1
                prev_cy[oid] = cy

            # ── draw annotations ────────────────────────────────
            if roi_orig:
                cv2.polylines(ann, [np.array(roi_orig)], True,
                              (255, 255, 0), 2)
            cv2.line(ann, (0, line_y), (self.vid_w, line_y),
                     (0, 0, 255), 3)

            for (bx, by, bw, bh), c in zip(rects, r_cls):
                col = (0, 255, 0) if c == "large" else (200, 140, 0)
                cv2.rectangle(ann, (bx, by), (bx+bw, by+bh), col, 2)
                cv2.putText(ann, c.upper(), (bx, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

            for oid, ctr in objects.items():
                cv2.circle(ann, tuple(ctr), 6, (0, 255, 255), -1)
                cv2.putText(ann, f"ID{oid}", (ctr[0]-15, ctr[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)

            info = (f"Large: {large_cnt}   Cars: {small_cnt}   "
                    f"Total: {large_cnt + small_cnt}")
            cv2.putText(ann, info, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            writer.write(ann)

            # send preview to UI every 3 frames
            if i % 3 == 0:
                self._preview = ann

            self._progress = 0.45 + 0.55 * (i + 1) / total
            self.results = dict(large=large_cnt, small=small_cnt,
                                total=large_cnt + small_cnt)

        writer.release()
        cap.release()

        tag = "HOG+SVM" if use_svm else "size-based"
        self._status_msg = (
            f"Done ({tag}) — Large: {large_cnt}  "
            f"Cars: {small_cnt}  Total: {large_cnt + small_cnt}")
        self._progress = 1.0

    # ============================================================
    #  EXPORT LOG
    # ============================================================
    def _export_log(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "output", "dashboard_count_log.txt")
        r = self.results
        with open(path, "w") as f:
            f.write("Traffic Vehicle Counter — Dashboard Results\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Video : {self.video_path}\n")
            f.write(f"Frames: {self.total_frames}  "
                    f"FPS: {self.fps:.0f}\n\n")
            f.write("Parameters:\n")
            for k, v in self.params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nROI vertices (display): {self.roi_points}\n")
            f.write(f"Counting line y (display): {self.count_line_y}\n\n")
            f.write("Results:\n")
            f.write(f"  Large vehicles : {r['large']}\n")
            f.write(f"  Cars / small   : {r['small']}\n")
            f.write(f"  Total counted  : {r['total']}\n\n")
            f.write(f"Annotated video  : {self.output_video_path}\n")
        self.status_var.set(f"Log saved → {path}")
        messagebox.showinfo("Saved", f"Count log saved to:\n{path}")

    # ============================================================
    #  CLEANUP
    # ============================================================
    def on_close(self):
        self._cancel = True
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ================================================================
#  ENTRY POINT
# ================================================================
def main():
    root = tk.Tk()
    root.geometry("1300x780")
    root.minsize(1000, 620)
    app = Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()