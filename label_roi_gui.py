# roi_tool_gui.py — RAW .hdr + images only; no .npy open; calibrate only for .hdr
import os
import csv
import time
import traceback
import numpy as np
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Optional: OpenCV for saving/normalizing
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Optional: PlantCV for ENVI RAW reading + calibration
try:
    from plantcv import plantcv as pcv
    HAS_PCV = True
except Exception:
    HAS_PCV = False


# ---------------------------
# Utilities
# ---------------------------
def _norm_uint8_cv2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    if HAS_CV2:
        return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    z = (x - x_min) / (x_max - x_min + 1e-12)
    return (z * 255).clip(0, 255).astype(np.uint8)


def save_rgb_image(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_CV2 and rgb.ndim == 3 and rgb.shape[2] == 3:
        cv2.imwrite(str(path), rgb[..., ::-1])  # RGB->BGR
    else:
        Image.fromarray(rgb).save(str(path))


def rgb_from_cube_bands(cube: np.ndarray, b_idx=24, g_idx=70, r_idx=95) -> np.ndarray:
    H, W, C = cube.shape
    b_idx = max(0, min(b_idx, C - 1))
    g_idx = max(0, min(g_idx, C - 1))
    r_idx = max(0, min(r_idx, C - 1))
    blue = _norm_uint8_cv2(cube[..., b_idx])
    green = _norm_uint8_cv2(cube[..., g_idx])
    red = _norm_uint8_cv2(cube[..., r_idx])
    return np.dstack([red, green, blue])  # true RGB


# ---------------------------
# ENVI helpers
# ---------------------------
def _derive_paths_from_hdr(hdr_path: str):
    """
    From .../<image_dir>/capture/*.hdr:
      - try going two dirs up if it still has 'capture'
      - infer <folder> from hdr filename (strip DARKREF_/WHITEREF_ if present)
      - build RAW paths:
          capture/<folder>.raw
          capture/DARKREF_<folder>.raw
          capture/WHITEREF_<folder>.raw
    """
    hdr_p = Path(hdr_path)
    capture_dir = hdr_p.parent
    image_dir_path = capture_dir.parent  # one up from 'capture'
    maybe_two_up = image_dir_path.parent
    if (maybe_two_up / "capture").exists():
        image_dir_path = maybe_two_up

    name = hdr_p.stem
    if name.startswith("DARKREF_"):
        folder = name[len("DARKREF_"):]
    elif name.startswith("WHITEREF_"):
        folder = name[len("WHITEREF_"):]
    else:
        folder = name

    cap = image_dir_path / "capture"
    data_raw_path = cap / f"{folder}.raw"
    dark_ref_raw_path = cap / f"DARKREF_{folder}.raw"
    white_ref_raw_path = cap / f"WHITEREF_{folder}.raw"
    return str(image_dir_path), folder, str(data_raw_path), str(dark_ref_raw_path), str(white_ref_raw_path)


def _read_envi(path: str):
    if not HAS_PCV:
        raise RuntimeError("PlantCV not available. Install 'plantcv'.")
    return pcv.readimage(path, mode='envi')


def _to_np_array(obj):
    if hasattr(obj, "array_data"):
        return obj.array_data
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "mat_data"):
        return obj.mat_data
    raise TypeError("Unsupported ENVI object type; cannot extract array data.")


# ---------------------------
# Loaders (RAW and calibrated)
# ---------------------------
def load_raw_cube_from_hdr(hdr_path: str):
    base_dir, folder, data_raw_path, _, _ = _derive_paths_from_hdr(hdr_path)
    if not os.path.exists(data_raw_path):
        raise FileNotFoundError(f"Missing RAW data: {data_raw_path}")

    raw_obj = _read_envi(data_raw_path)
    cube = _to_np_array(raw_obj)    # RAW (uncalibrated)
    rgb = rgb_from_cube_bands(cube)
    return cube, rgb, base_dir


def calibrate_cube_from_hdr(hdr_path: str):
    base_dir, folder, data_raw_path, dark_ref_raw_path, white_ref_raw_path = _derive_paths_from_hdr(hdr_path)
    for p in (data_raw_path, dark_ref_raw_path, white_ref_raw_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required RAW file: {p}")

    raw_img = _read_envi(data_raw_path)
    white_ref = _read_envi(white_ref_raw_path)
    dark_ref = _read_envi(dark_ref_raw_path)

    calibrated = pcv.hyperspectral.calibrate(
        raw_data=raw_img, white_reference=white_ref, dark_reference=dark_ref
    )
    cube = _to_np_array(calibrated)
    rgb = rgb_from_cube_bands(cube)
    return cube, rgb, base_dir


# ---------------------------
# Fallback loader (images only)
# ---------------------------
def load_image_only(path: str):
    """Load a standard RGB image; produce (cube, rgb) with cube == rgb."""
    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    cube = rgb
    return cube, rgb


# ---------------------------
# Colors
# ---------------------------
DEFAULT_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9A6324",
    "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
    "#808080", "#ffffff", "#000000",
]


# ---------------------------
# GUI App
# ---------------------------
class ROIToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HSI ROI Creator (RAW .hdr + images; no .npy open)")
        self.geometry("1320x880")

        # Image state
        self.cube = None
        self.rgb = None
        self.display_img = None
        self.source = None           # current file path (.hdr / image)
        self.base_image_dir = None
        self.output_dir = None

        # Drawing state
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_bbox = None
        self.scale = 1.0
        self.shapes = []

        # Class state
        self.class_to_color = {}
        self.class_order = []
        self.active_class = tk.StringVar(value="")

        self._build_ui()
        self.canvas.bind("<Configure>", lambda e: self.render_image())

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

    # ---------- UI ----------
    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open HSI .hdr (RAW)", command=self.on_open_hsi_hdr_raw).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open Image", command=self.on_open_image).pack(side=tk.LEFT, padx=4)

        # Calibrate: enabled only when current source is .hdr
        self.btn_cal = ttk.Button(top, text="Calibrate (PlantCV)", command=self.on_calibrate_current_hdr)
        self.btn_cal.pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="Choose Output Folder", command=self.on_choose_output).pack(side=tk.LEFT, padx=12)
        self.output_label = ttk.Label(top, text="Output: (not set)")
        self.output_label.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Save ROI", command=self.on_save_roi).pack(side=tk.RIGHT, padx=6)
        ttk.Button(top, text="Clear Last Box", command=self.on_clear_last_box).pack(side=tk.RIGHT, padx=6)
        ttk.Button(top, text="Clear All Boxes", command=self.on_clear_all_boxes).pack(side=tk.RIGHT, padx=6)

        # Class bar
        clsbar = ttk.Frame(self, padding=(6, 0, 6, 6))
        clsbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(clsbar, text="Active class:").pack(side=tk.LEFT, padx=(0, 4))
        self.class_combo = ttk.Combobox(clsbar, textvariable=self.active_class, values=[], state="readonly", width=22)
        self.class_combo.pack(side=tk.LEFT, padx=(0, 12))

        self.new_class_entry = ttk.Entry(clsbar, width=18)
        self.new_class_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(clsbar, text="Add Class", command=self.on_add_class).pack(side=tk.LEFT, padx=(0, 12))

        # Main area
        main = tk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: fixed-width, scrollable labels
        self.left_panel = tk.Frame(main, bd=1, relief=tk.GROOVE, width=260)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)

        ttk.Label(self.left_panel, text="Labels (click to set active):").pack(anchor="w", padx=8, pady=(8, 4))

        self.labels_canvas = tk.Canvas(self.left_panel, borderwidth=0, highlightthickness=0)
        self.labels_scroll = ttk.Scrollbar(self.left_panel, orient="vertical", command=self.labels_canvas.yview)
        self.labels_container = ttk.Frame(self.labels_canvas)
        self.labels_container.bind("<Configure>", lambda e: self.labels_canvas.configure(scrollregion=self.labels_canvas.bbox("all")))
        self.labels_canvas.create_window((0, 0), window=self.labels_container, anchor="nw")
        self.labels_canvas.configure(yscrollcommand=self.labels_scroll.set)
        self.labels_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))
        self.labels_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 8))

        # Right: canvas
        self.canvas = tk.Canvas(main, bg="#202020", cursor="tcross")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Status bar
        self.status = tk.StringVar(value="Open an HSI .hdr to view RAW (calibration available), or open an image.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def _refresh_class_ui(self):
        for w in self.labels_container.winfo_children():
            w.destroy()
        for name in self.class_order:
            color = self.class_to_color[name]
            row = tk.Frame(self.labels_container)
            row.pack(fill=tk.X, padx=6, pady=2)
            sw = tk.Canvas(row, width=18, height=18, highlightthickness=0, bg=color)
            sw.pack(side=tk.LEFT)
            btn = ttk.Button(row, text=name, command=lambda n=name: self._set_active_class(n))
            btn.pack(side=tk.LEFT, padx=6)
        self.labels_container.update_idletasks()
        self.labels_canvas.configure(scrollregion=self.labels_canvas.bbox("all"))

    # ---------- Class management ----------
    def on_add_class(self):
        name = self.new_class_entry.get().strip()
        if not name:
            messagebox.showwarning("Class name", "Please type a class name.")
            return
        if name in self.class_to_color:
            messagebox.showinfo("Exists", f"Class '{name}' already exists.")
            self.active_class.set(name)
            self.class_combo.set(name)
            return
        color = self._next_color()
        self.class_to_color[name] = color
        self.class_order.append(name)
        self._refresh_class_ui()
        self.active_class.set(name)
        self.class_combo["values"] = self.class_order
        self.class_combo.set(name)
        self.new_class_entry.delete(0, tk.END)
        self.status.set(f"Added class '{name}' with color {color}")

    def _set_active_class(self, name: str):
        if name in self.class_to_color:
            self.active_class.set(name)
            self.class_combo.set(name)
            self.status.set(f"Active class: {name}")

    def _next_color(self) -> str:
        used = set(self.class_to_color.values())
        for hx in DEFAULT_PALETTE:
            if hx not in used:
                return hx
        from random import randint
        return "#%02x%02x%02x" % (randint(64, 255), randint(64, 255), randint(64, 255))

    # ---------- Open / Calibrate ----------
    def on_open_hsi_hdr_raw(self):
        path = filedialog.askopenfilename(
            title="Select HSI .hdr (under .../capture/*.hdr)",
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.status.set("Loading RAW cube...")
            self.update_idletasks()
            cube, rgb, base_dir = load_raw_cube_from_hdr(path)
        except Exception as e:
            messagebox.showerror("Load RAW error", f"{e}")
            traceback.print_exc()
            self.status.set("RAW load failed.")
            return
        self._set_image(cube, rgb, path, base_dir)
        self.status.set("RAW cube loaded. Click Calibrate if you want calibrated data.")
        self._update_calibrate_state()

    def on_open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            cube, rgb = load_image_only(path)
        except Exception as e:
            messagebox.showerror("Error loading image", f"{e}")
            traceback.print_exc()
            self.status.set("Load failed.")
            return
        self._set_image(cube, rgb, path, base_dir=None)
        self.status.set("Image loaded.")
        self._update_calibrate_state()

    def on_calibrate_current_hdr(self):
        """Run PlantCV calibration only if current source is an .hdr."""
        if not (isinstance(self.source, str) and self.source.lower().endswith(".hdr")):
            messagebox.showinfo(
                "Calibration not available",
                "Calibration is only available when a RAW .hdr is open."
            )
            return
        try:
            if not HAS_PCV:
                raise RuntimeError("PlantCV not available. Install 'plantcv'.")
            self.status.set("Calibrating (PlantCV)...")
            self.update_idletasks()
            cube, rgb, base_dir = calibrate_cube_from_hdr(self.source)
        except Exception as e:
            messagebox.showerror("Calibration error", f"{e}")
            traceback.print_exc()
            self.status.set("Calibration failed.")
            return
        self._set_image(cube, rgb, self.source, base_dir)
        self.status.set("Calibration complete.")
        self._update_calibrate_state()

    # ---------- Common setters & UI state ----------
    def _set_image(self, cube, rgb, source, base_dir):
        self.cube = cube
        self.rgb = rgb
        self.source = source
        self.base_image_dir = base_dir

        self.shapes.clear()
        self.rect_id = None
        self.current_bbox = None

        H, W = rgb.shape[:2]
        src_name = Path(source).name if isinstance(source, str) else "image"
        extra = f" | base_dir: {base_dir}" if base_dir else ""
        self.status.set(f"Loaded: {src_name} | RGB: {H}x{W} | Cube: {cube.shape}{extra}")
        self.render_image()

    def _update_calibrate_state(self):
        """Enable Calibrate only when current source is .hdr."""
        is_hdr = isinstance(self.source, str) and str(self.source).lower().endswith(".hdr")
        self.btn_cal.configure(state=(tk.NORMAL if is_hdr else tk.DISABLED))

    # ---------- Output ----------
    def on_choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if not d:
            return
        self.output_dir = d
        self.output_label.config(text=f"Output: {d}")

    # ---------- Canvas & drawing ----------
    def _image_draw_region(self):
        if self.rgb is None:
            return 0, 0, 1, 1, 1, 1
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        H, W, _ = self.rgb.shape
        disp_w = int(W * self.scale)
        disp_h = int(H * self.scale)
        img_left = (cw - disp_w) // 2
        img_top = (ch - disp_h) // 2
        return img_left, img_top, disp_w, disp_h, W, H

    def render_image(self):
        if self.rgb is None:
            return
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        H, W, _ = self.rgb.shape
        scale = min(cw / W, ch / H)
        scale = min(scale, 1.0)
        self.scale = scale

        disp_w = max(int(W * scale), 1)
        disp_h = max(int(H * scale), 1)
        pil = Image.fromarray(self.rgb).resize((disp_w, disp_h), Image.BILINEAR)
        self.display_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image((cw // 2, ch // 2), image=self.display_img, anchor=tk.CENTER)

        img_left, img_top, _, _, _, _ = self._image_draw_region()
        for shp in self.shapes:
            (x1, y1, x2, y2) = shp["bbox_img"]
            x1d = int(x1 * self.scale) + img_left
            y1d = int(y1 * self.scale) + img_top
            x2d = int(x2 * self.scale) + img_left
            y2d = int(y2 * self.scale) + img_top
            color = shp["color"]
            rect_id = self.canvas.create_rectangle(x1d, y1d, x2d, y2d, outline=color, width=2)
            text_id = self.canvas.create_text(x1d + 4, y1d + 10, text=shp["class"], anchor="w", fill=color)
            shp["canvas_rect_id"] = rect_id
            shp["canvas_text_id"] = text_id

    def on_mouse_down(self, event):
        if self.rgb is None:
            return
        if not self.active_class.get():
            messagebox.showinfo("No class", "Add a class and select it as active.")
            return
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.current_bbox = None

    def on_mouse_drag(self, event):
        if self.start_x is None or self.start_y is None:
            return
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        cls = self.active_class.get()
        color = self.class_to_color.get(cls, "#00FF00")
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline=color, width=2
        )

    def on_mouse_up(self, event):
        if self.start_x is None or self.start_y is None:
            return
        x1, x2 = sorted([self.start_x, event.x])
        y1, y2 = sorted([self.start_y, event.y])

        img_left, img_top, disp_w, disp_h, W, H = self._image_draw_region()
        x1 = int(np.clip(x1, img_left, img_left + disp_w))
        x2 = int(np.clip(x2, img_left, img_left + disp_w))
        y1 = int(np.clip(y1, img_top, img_top + disp_h))
        y2 = int(np.clip(y2, img_top, img_top + disp_h))

        if x2 <= x1 or y2 <= y1:
            self.status.set("Invalid ROI: zero area.")
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            return

        x1_img = int((x1 - img_left) / self.scale)
        y1_img = int((y1 - img_top) / self.scale)
        x2_img = int((x2 - img_left) / self.scale)
        y2_img = int((y2 - img_top) / self.scale)

        x1_img = int(np.clip(x1_img, 0, W - 1))
        x2_img = int(np.clip(x2_img, 0, W))
        y1_img = int(np.clip(y1_img, 0, H - 1))
        y2_img = int(np.clip(y2_img, 0, H))

        if x2_img <= x1_img or y2_img <= y1_img:
            self.status.set("Invalid ROI after clipping.")
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            return

        cls = self.active_class.get()
        color = self.class_to_color.get(cls, "#00FF00")
        shp = {
            "bbox_img": (x1_img, y1_img, x2_img, y2_img),
            "class": cls,
            "color": color,
            "canvas_rect_id": None,
            "canvas_text_id": None,
        }
        self.shapes.append(shp)

        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.render_image()
        self.status.set(f"ROI added for class '{cls}': ({x1_img},{y1_img})-({x2_img},{y2_img})")

    # ---------- Clear ----------
    def on_clear_last_box(self):
        if self.shapes:
            last = self.shapes.pop()
            if last.get("canvas_rect_id"):
                self.canvas.delete(last["canvas_rect_id"])
            if last.get("canvas_text_id"):
                self.canvas.delete(last["canvas_text_id"])
            self.render_image()
            self.status.set("Removed last ROI.")
        else:
            self.status.set("No ROIs to remove.")

    def on_clear_all_boxes(self):
        self.shapes.clear()
        self.render_image()
        self.status.set("Cleared all ROIs.")

    # ---------- Save ----------
    def on_save_roi(self):
        if self.rgb is None or self.cube is None:
            messagebox.showwarning("No image", "Open an HSI .hdr (RAW) or an image first.")
            return
        if self.output_dir is None:
            messagebox.showwarning("No output folder", "Choose an output folder.")
            return
        if not self.shapes:
            messagebox.showwarning("No ROIs", "Draw at least one ROI.")
            return

        base_out = Path(self.output_dir)
        csv_path = base_out / "roi_log.csv"
        header = ["rgb_file", "cube_file", "class", "x1", "y1", "x2", "y2", "source", "base_image_dir"]

        saved = 0
        for shp in self.shapes:
            (x1, y1, x2, y2) = shp["bbox_img"]
            cls = shp["class"]

            roi_rgb = self.rgb[y1:y2, x1:x2]
            roi_cube = self.cube[y1:y2, x1:x2, :] if self.cube.ndim == 3 else self.cube[y1:y2, x1:x2]

            cls_dir = base_out / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            src_name = Path(self.source).stem if isinstance(self.source, str) else "image"

            jpg_path = cls_dir / f"{src_name}_{cls}_{timestamp}_{saved+1}.jpg"
            npy_path = cls_dir / f"{src_name}_{cls}_{timestamp}_{saved+1}.npy"

            try:
                save_rgb_image(jpg_path, roi_rgb)
                np.save(npy_path, roi_cube)
                saved += 1
            except Exception as e:
                messagebox.showerror("Save error", f"Failed to save ROI files:\n{e}")
                traceback.print_exc()
                continue

            try:
                write_header = not csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    w.writerow([
                        str(jpg_path.relative_to(base_out)),
                        str(npy_path.relative_to(base_out)),
                        cls, x1, y1, x2, y2, str(self.source), str(self.base_image_dir or "")
                    ])
            except Exception as e:
                messagebox.showwarning("CSV warning", f"ROI saved, but CSV logging failed:\n{e}")

        self.status.set(f"Saved {saved} ROI(s).")
        self.shapes.clear()
        self.render_image()


if __name__ == "__main__":
    app = ROIToolApp()
    app.mainloop()
