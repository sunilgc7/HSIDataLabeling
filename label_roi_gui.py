# roi_tool_gui.py
# -------------------------------------------------------------
# GUI for HSI ROI creation with dynamic class labels and per-class colors.
# - Open HSI .hdr (under .../capture/*.hdr). The app goes two dirs up to find the base image dir
#   and reconstructs RAW paths:
#     capture/{FOLDER}.raw
#     capture/DARKREF_{FOLDER}.raw
#     capture/WHITEREF_{FOLDER}.raw
# - Calibrate using PlantCV (raw + white/dark refs).
# - Visualize RGB composite with bands: B=24, G=70, R=95 (per-band normalization).
# - Add custom classes at runtime (no hard-coded list).
# - Draw ROIs with the mouse in the current class color; save RGB + HSI crops per class.
# - Append roi_log.csv: rgb_file, cube_file, class, bbox, source, base_image_dir.
# -------------------------------------------------------------

import os
import csv
import sys
import time
import random
import traceback
import numpy as np
from pathlib import Path

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Imaging
from PIL import Image, ImageTk

# Optional: OpenCV for saving/normalizing
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# PlantCV for ENVI RAW reading + calibration
try:
    from plantcv import plantcv as pcv
    HAS_PCV = True
except Exception:
    HAS_PCV = False


# ---------------------------
# Utilities
# ---------------------------

def _norm_uint8_cv2(x):
    """Normalize a 2D array to uint8 using cv2.normalize-like behavior."""
    x = x.astype(np.float32)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    if HAS_CV2:
        out = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return out
    else:
        z = (x - x_min) / (x_max - x_min + 1e-12)
        return (z * 255).clip(0, 255).astype(np.uint8)


def save_rgb_image(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_CV2 and rgb.ndim == 3 and rgb.shape[2] == 3:
        cv2.imwrite(str(path), rgb[..., ::-1])  # RGB->BGR
    else:
        Image.fromarray(rgb).save(str(path))


def rgb_from_cube_bands_like_user(cube: np.ndarray, b_idx=24, g_idx=70, r_idx=95) -> np.ndarray:
    """Create RGB as in your original snippet: normalize each band then merge to RGB."""
    H, W, C = cube.shape
    b_idx = min(max(0, b_idx), C - 1)
    g_idx = min(max(0, g_idx), C - 1)
    r_idx = min(max(0, r_idx), C - 1)

    blue_band = cube[..., b_idx]
    green_band = cube[..., g_idx]
    red_band = cube[..., r_idx]

    blue_norm = _norm_uint8_cv2(blue_band)
    green_norm = _norm_uint8_cv2(green_band)
    red_norm = _norm_uint8_cv2(red_band)

    # Return true RGB for display
    rgb = np.dstack([red_norm, green_norm, blue_norm])
    return rgb


# ---------------------------
# HSI Loader from .hdr (two dirs up)
# ---------------------------

def _derive_paths_from_hdr(hdr_path: str):
    """
    Given any .hdr path under .../<image_dir>/capture/*.hdr,
    - go two directories up to get image_dir_path (as requested),
    - infer 'folder' from the .hdr filename (strip DARKREF_ / WHITEREF_ if present),
    - build RAW file paths per your pattern.
    """
    hdr_p = Path(hdr_path)
    capture_dir = hdr_p.parent
    image_dir_path = capture_dir.parent  # 1 up from 'capture' = image_dir
    # Honor "go two directories back" if feasible
    maybe_two_up = image_dir_path.parent
    if (maybe_two_up / "capture").exists():
        image_dir_path = maybe_two_up

    name = hdr_p.stem  # "FOLDER" or "DARKREF_FOLDER" or "WHITEREF_FOLDER"
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

    return image_dir_path, folder, str(data_raw_path), str(dark_ref_raw_path), str(white_ref_raw_path)


def load_hsi_from_hdr(hdr_path: str):
    """
    PlantCV-based loader mirroring your pipeline, driven from a .hdr selection.
    Returns:
      cube: calibrated hyperspectral cube (H, W, Bands)
      rgb:  RGB composite using (B=24, G=70, R=95)
      base_image_dir: the directory two levels up from the hdr
    """
    if not HAS_PCV:
        raise RuntimeError("PlantCV not available. Install 'plantcv' to load ENVI data.")

    image_dir_path, folder, data_raw_path, dark_ref_raw_path, white_ref_raw_path = _derive_paths_from_hdr(hdr_path)

    for p in (data_raw_path, dark_ref_raw_path, white_ref_raw_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required RAW file: {p}")

    raw_img = pcv.readimage(data_raw_path, mode='envi')
    white_ref = pcv.readimage(white_ref_raw_path, mode='envi')
    dark_ref = pcv.readimage(dark_ref_raw_path, mode='envi')

    calibrated = pcv.hyperspectral.calibrate(
        raw_data=raw_img, white_reference=white_ref, dark_reference=dark_ref
    )
    cube = calibrated.array_data  # (H, W, Bands)
    rgb = rgb_from_cube_bands_like_user(cube, b_idx=24, g_idx=70, r_idx=95)
    return cube, rgb, str(image_dir_path)


# ---------------------------
# Fallback loaders
# ---------------------------

def load_image_or_npy(path: str):
    """Load a standard RGB image or a .npy cube; produce (cube, rgb)."""
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        cube = np.load(path)
        rgb = quick_rgb(cube)
        return cube, rgb
    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    cube = rgb
    return cube, rgb


def quick_rgb(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError("Expected (H, W, C) array.")
    H, W, C = cube.shape
    if C == 3:
        img = cube.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        return (img * 255).clip(0, 255).astype(np.uint8)

    # Heuristic: pick bands [60, 30, 10] as [R,G,B] with fallbacks
    def pick(idx, fallback):
        return idx if idx < C else fallback

    r_idx = pick(60, C - 1)
    g_idx = pick(30, C // 2)
    b_idx = pick(10, 0)

    def norm(x):
        x = x.astype(np.float32)
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        if hi <= lo:
            lo, hi = x.min(), x.max()
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
        x = np.clip((x - lo) / (hi - lo), 0, 1)
        return (x * 255).astype(np.uint8)

    r = norm(cube[..., r_idx])
    g = norm(cube[..., g_idx])
    b = norm(cube[..., b_idx])
    return np.stack([r, g, b], axis=-1)


# ---------------------------
# Colors
# ---------------------------

DEFAULT_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9A6324",
    "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
    "#808080", "#ffffff", "#000000",
]

def _hex_to_rgb_tuple(hx: str):
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join([c*2 for c in hx])
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return (r, g, b)


# ---------------------------
# GUI App
# ---------------------------

class ROIToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HSI ROI Creator (dynamic classes, per-class colors)")
        self.geometry("1320x860")

        # Image state
        self.cube = None       # (H,W,B) or (H,W,3)
        self.rgb = None        # (H,W,3) uint8
        self.display_img = None
        self.source = None     # hdr path or image path
        self.base_image_dir = None  # derived from hdr
        self.output_dir = None

        # Canvas drawing state
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_bbox = None
        self.scale = 1.0
        self.shapes = []  # list of dicts: {bbox(img coords), class, color, canvas_rect_id, canvas_text_id}

        # Class management (dynamic)
        self.class_to_color = {}   # name -> hex color
        self.class_order = []      # to preserve order for UI
        self.active_class = tk.StringVar(value="")  # empty until user adds

        # Build UI
        self._build_ui()

        # Redraw on resize
        self.canvas.bind("<Configure>", lambda e: self.render_image())

    # ---------- UI ----------
    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open HSI .hdr", command=self.on_open_hsi_hdr).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open File (image/NPY)", command=self.on_open_file).pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text="Choose Output Folder", command=self.on_choose_output).pack(side=tk.LEFT, padx=12)
        self.output_label = ttk.Label(top, text="Output: (not set)")
        self.output_label.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Save ROI", command=self.on_save_roi).pack(side=tk.RIGHT, padx=6)
        ttk.Button(top, text="Clear Last Box", command=self.on_clear_last_box).pack(side=tk.RIGHT, padx=6)
        ttk.Button(top, text="Clear All Boxes", command=self.on_clear_all_boxes).pack(side=tk.RIGHT, padx=6)

        # Second row: class management
        clsbar = ttk.Frame(self, padding=(6, 0, 6, 6))
        clsbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(clsbar, text="Active class:").pack(side=tk.LEFT, padx=(0, 4))
        self.class_combo = ttk.Combobox(clsbar, textvariable=self.active_class, values=[], state="readonly", width=22)
        self.class_combo.pack(side=tk.LEFT, padx=(0, 12))

        self.new_class_entry = ttk.Entry(clsbar, width=18)
        self.new_class_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(clsbar, text="Add Class", command=self.on_add_class).pack(side=tk.LEFT, padx=(0, 12))

        # Classes panel (left)
        left_panel = ttk.Frame(self, padding=6)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_panel, text="Labels (click to set active):").pack(anchor="w")

        self.labels_frame = tk.Frame(left_panel)
        self.labels_frame.pack(fill=tk.Y, expand=False, pady=(4, 8))

        # Canvas (center/right)
        self.canvas = tk.Canvas(self, bg="#202020", cursor="tcross")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Bindings for drawing on canvas
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Status bar
        self.status = tk.StringVar(value="Add at least one class to start.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

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

        # Assign a color
        color = self._next_color()
        self.class_to_color[name] = color
        self.class_order.append(name)
        self._refresh_class_ui()

        # Make it active
        self.active_class.set(name)
        self.class_combo["values"] = self.class_order
        self.class_combo.set(name)

        self.new_class_entry.delete(0, tk.END)
        self.status.set(f"Added class '{name}' with color {color}")

    def _refresh_class_ui(self):
        # Clear current label widgets
        for w in self.labels_frame.winfo_children():
            w.destroy()

        # Rebuild list with colored swatches
        for name in self.class_order:
            color = self.class_to_color[name]

            row = tk.Frame(self.labels_frame)
            row.pack(fill=tk.X, padx=0, pady=2)

            sw = tk.Canvas(row, width=18, height=18, highlightthickness=0, bg=color)
            sw.pack(side=tk.LEFT)
            btn = ttk.Button(row, text=name, command=lambda n=name: self._set_active_class(n))
            btn.pack(side=tk.LEFT, padx=6)

        # Update combobox values
        self.class_combo["values"] = self.class_order

    def _set_active_class(self, name: str):
        if name in self.class_to_color:
            self.active_class.set(name)
            self.class_combo.set(name)
            self.status.set(f"Active class: {name}")

    def _next_color(self) -> str:
        # Cycle through palette; if exhausted, pick a random bright color
        used = set(self.class_to_color.values())
        for hx in DEFAULT_PALETTE:
            if hx not in used:
                return hx
        # Fallback random bright color
        def rnd():
            return random.randint(64, 255)
        return "#%02x%02x%02x" % (rnd(), rnd(), rnd())

    # ---------- Open ----------
    def on_open_hsi_hdr(self):
        path = filedialog.askopenfilename(
            title="Select HSI .hdr (under .../capture/*.hdr)",
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            cube, rgb, base_dir = load_hsi_from_hdr(path)
        except Exception as e:
            messagebox.showerror("Error loading HSI", f"{e}")
            traceback.print_exc()
            return
        self._set_image(cube, rgb, path, base_dir)

    def on_open_file(self):
        path = filedialog.askopenfilename(
            title="Select image or .npy",
            filetypes=[
                ("All supported", "*.npy;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                ("NumPy array", "*.npy"),
                ("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            cube, rgb = load_image_or_npy(path)
        except Exception as e:
            messagebox.showerror("Error loading file", f"{e}")
            traceback.print_exc()
            return
        self._set_image(cube, rgb, path, base_dir=None)

    def _set_image(self, cube, rgb, source, base_dir):
        self.cube = cube
        self.rgb = rgb
        self.source = source
        self.base_image_dir = base_dir

        # Clear shapes when loading new image
        self.shapes.clear()
        self.rect_id = None
        self.current_bbox = None

        H, W = rgb.shape[:2]
        src_name = Path(source).name if isinstance(source, str) else "image"
        extra = f" | base_dir: {base_dir}" if base_dir else ""
        self.status.set(f"Loaded: {src_name} | RGB: {H}x{W} | Cube: {cube.shape}{extra}")
        self.render_image()

    # ---------- Output ----------
    def on_choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if not d:
            return
        self.output_dir = d
        self.output_label.config(text=f"Output: {d}")

    # ---------- Canvas helpers ----------
    def _image_draw_region(self):
        """Return (img_left, img_top, disp_w, disp_h, W, H) for display coords."""
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
        """Draw background image and all persistent rectangles."""
        if self.rgb is None:
            return
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)

        H, W, _ = self.rgb.shape
        scale = min(cw / W, ch / H)
        scale = min(scale, 1.0)  # don't upscale
        self.scale = scale

        disp_w = max(int(W * scale), 1)
        disp_h = max(int(H * scale), 1)
        pil = Image.fromarray(self.rgb).resize((disp_w, disp_h), Image.BILINEAR)
        self.display_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image((cw // 2, ch // 2), image=self.display_img, anchor=tk.CENTER)

        # Redraw persistent shapes
        img_left, img_top, _, _, _, _ = self._image_draw_region()
        for shp in self.shapes:
            (x1, y1, x2, y2) = shp["bbox_img"]
            # Map to display coords
            x1d = int(x1 * self.scale) + img_left
            y1d = int(y1 * self.scale) + img_top
            x2d = int(x2 * self.scale) + img_left
            y2d = int(y2 * self.scale) + img_top
            color = shp["color"]
            rect_id = self.canvas.create_rectangle(x1d, y1d, x2d, y2d, outline=color, width=2)
            # Label text (small)
            text_id = self.canvas.create_text(x1d + 4, y1d + 10, text=shp["class"], anchor="w", fill=color)
            shp["canvas_rect_id"] = rect_id
            shp["canvas_text_id"] = text_id

        # If a temp rect is in progress, it will be handled by drag events.

    # ---------- Mouse events ----------
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
        # Use current class color
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

        # Clip to displayed image region
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

        # Convert to image coordinates
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

        # Persist shape
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

        # Reset temp rect; force re-render to draw persistent rect+label
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
            messagebox.showwarning("No image", "Open an HSI .hdr or an image/NPY first.")
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

        # Save each ROI
        saved = 0
        for shp in self.shapes:
            (x1, y1, x2, y2) = shp["bbox_img"]
            cls = shp["class"]

            roi_rgb = self.rgb[y1:y2, x1:x2]
            # Hyperspectral crop (or RGB if fallback)
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

            # Append to CSV
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

        # If you want to keep boxes after saving, comment out the next two lines:
        self.shapes.clear()
        self.render_image()


if __name__ == "__main__":
    app = ROIToolApp()
    app.mainloop()
