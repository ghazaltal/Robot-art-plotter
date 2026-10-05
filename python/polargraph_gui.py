"""
POLARGRAPH PLOTTER GUI Dissertation Project
A gui to:
1. Load and preview images
2. Choose from 5 image processing methods
3. Preview processed coordinates
4. Connect to Arduino via serial
5. Send coordinates and monitor drawing progress

DEPENDENCIES:
  - tkinter (built into Python)
  - opencv-python
  - numpy
  - PIL/Pillow
  - serial (pyserial)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import glob


try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial not installed. Install with: pip3 install pyserial")
    print("Running in preview-only mode.\n")



# IMAGE PROCESSING METHODS


def method_edge_detection(img_gray, params):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, params.get('low_thresh', 50), params.get('high_thresh', 150))
    points = np.column_stack(np.where(edges > 0))
    return points[:, 1], points[:, 0], edges

def method_adaptive_threshold(img_gray, params):
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        blockSize=params.get('block_size', 11),
        C=params.get('C', 2)
    )
    points = np.column_stack(np.where(binary > 0))
    return points[:, 1], points[:, 0], binary

def method_dithering(img_gray, params):
    img = img_gray.astype(np.float64)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 255.0 if old > 127 else 0.0
            img[y, x] = new
            error = old - new
            if x + 1 < w:
                img[y, x + 1] += error * 7.0 / 16.0
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3.0 / 16.0
                img[y + 1, x] += error * 5.0 / 16.0
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1.0 / 16.0
    dithered = np.clip(img, 0, 255).astype(np.uint8)
    inv = 255 - dithered
    points = np.column_stack(np.where(inv > 0))
    return points[:, 1], points[:, 0], dithered

def method_contour_tracing(img_gray, params):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_x, all_y = [], []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if cv2.arcLength(approx, True) > 10:
            for point in approx:
                all_x.append(point[0][0])
                all_y.append(point[0][1])
            all_x.append(approx[0][0][0])
            all_y.append(approx[0][0][1])
    contour_img = np.zeros_like(img_gray)
    for c in contours:
        eps = 0.005 * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps, True)
        if cv2.arcLength(a, True) > 10:
            cv2.drawContours(contour_img, [a], -1, 255, 1)
    return np.array(all_x), np.array(all_y), contour_img

def method_stippling(img_gray, params):
    inverted = 255.0 - img_gray.astype(np.float64)
    inverted = cv2.GaussianBlur(inverted, (5, 5), 2.0)
    prob_map = inverted / inverted.sum()
    h, w = img_gray.shape
    num_points = min(params.get('max_points', 2500), 2500)
    flat_prob = prob_map.flatten()
    indices = np.random.choice(len(flat_prob), size=num_points, replace=False, p=flat_prob)
    rows = indices // w
    cols = indices % w
    for iteration in range(5):
        new_rows = np.zeros_like(rows, dtype=np.float64)
        new_cols = np.zeros_like(cols, dtype=np.float64)
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            r_min, r_max = max(0, r - 5), min(h, r + 6)
            c_min, c_max = max(0, c - 5), min(w, c + 6)
            patch = inverted[r_min:r_max, c_min:c_max]
            if patch.sum() > 0:
                rr, cc = np.meshgrid(np.arange(r_min, r_max), np.arange(c_min, c_max), indexing='ij')
                total = patch.sum()
                new_rows[i] = (rr * patch).sum() / total
                new_cols[i] = (cc * patch).sum() / total
            else:
                new_rows[i], new_cols[i] = r, c
        rows = np.clip(new_rows.astype(int), 0, h - 1)
        cols = np.clip(new_cols.astype(int), 0, w - 1)
    stipple_img = np.ones_like(img_gray) * 255
    for r, c in zip(rows, cols):
        cv2.circle(stipple_img, (int(c), int(r)), 1, 0, -1)
    return cols.astype(float), rows.astype(float), stipple_img

def optimise_path(coords_x, coords_y):
    n = len(coords_x)
    if n <= 1:
        return coords_x, coords_y
    coords = np.column_stack([coords_x, coords_y]).astype(np.float64)
    visited = np.zeros(n, dtype=bool)
    path = np.zeros(n, dtype=int)
    path[0] = 0
    visited[0] = True
    for i in range(1, n):
        current = path[i - 1]
        dists = np.sum((coords - coords[current]) ** 2, axis=1)
        dists[visited] = 1e18
        path[i] = int(np.argmin(dists))
        visited[path[i]] = True
    return coords[path, 0], coords[path, 1]



# MAIN GUI# 

class PolargraphGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Polargraph Plotter — Robot Art")
        self.root.configure(bg='#0f1117')
        self.root.geometry("1100x750")
        self.image_path = None
        self.img_gray = None
        self.coords_x = None
        self.coords_y = None
        self.coords_mm_x = None
        self.coords_mm_y = None
        self.serial_conn = None
        self.is_drawing = False
        self.stop_drawing = False
        self.D = 400.0
        self.draw_width = 250.0
        self.draw_height = 100.0
        self.offset_x = 75.0
        self.offset_y = 310.0
        self.max_points = 2000
        self.home_y = 400.0
        self.setup_styles()
        self.build_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure('Dark.TFrame', background='#0f1117')
        style.configure('Card.TFrame', background='#1a1e2e')
        style.configure('Dark.TLabel', background='#1a1e2e', foreground='#e2e8f0',
                        font=('Helvetica', 11))
        style.configure('Title.TLabel', background='#0f1117', foreground='#e2e8f0',
                        font=('Helvetica', 18, 'bold'))
        style.configure('Header.TLabel', background='#1a1e2e', foreground='#00b4d8',
                        font=('Helvetica', 12, 'bold'))
        style.configure('Status.TLabel', background='#1a1e2e', foreground='#06d6a0',
                        font=('Helvetica', 10))
        style.configure('Dark.TButton', font=('Helvetica', 11))
        style.configure('Action.TButton', font=('Helvetica', 11, 'bold'))

    def build_ui(self):
        
        main = ttk.Frame(self.root, style='Dark.TFrame')
        main.pack(fill='both', expand=True, padx=12, pady=12)
        ttk.Label(main, text="POLARGRAPH PLOTTER", style='Title.TLabel').pack(anchor='w')
        ttk.Label(main, text="Robot Art Dissertation Project", 
                  background='#0f1117', foreground='#718096',
                  font=('Helvetica', 10)).pack(anchor='w', pady=(0, 10))

        
        content = ttk.Frame(main, style='Dark.TFrame')
        content.pack(fill='both', expand=True)

        
        left = ttk.Frame(content, style='Dark.TFrame', width=340)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        self.build_image_section(left)
        self.build_method_section(left)
        self.build_serial_section(left)
        self.build_drawing_section(left)

        
        right = ttk.Frame(content, style='Card.TFrame')
        right.pack(side='right', fill='both', expand=True)
        self.build_preview_section(right)

    def build_image_section(self, parent):
        frame = ttk.Frame(parent, style='Card.TFrame')
        frame.pack(fill='x', pady=(0, 8))
        inner = ttk.Frame(frame, style='Card.TFrame')
        inner.pack(fill='x', padx=10, pady=8)

        ttk.Label(inner, text="IMAGE", style='Header.TLabel').pack(anchor='w')

        btn_frame = ttk.Frame(inner, style='Card.TFrame')
        btn_frame.pack(fill='x', pady=(6, 0))

        tk.Button(btn_frame, text="Load Image", command=self.load_image,
                  bg='#2563eb', fg='white', font=('Helvetica', 11, 'bold'),
                  relief='flat', padx=12, pady=4).pack(side='left')

        self.img_label = ttk.Label(inner, text="No image loaded", style='Dark.TLabel')
        self.img_label.pack(anchor='w', pady=(4, 0))

    def build_method_section(self, parent):
        frame = ttk.Frame(parent, style='Card.TFrame')
        frame.pack(fill='x', pady=(0, 8))
        inner = ttk.Frame(frame, style='Card.TFrame')
        inner.pack(fill='x', padx=10, pady=8)

        ttk.Label(inner, text="METHOD", style='Header.TLabel').pack(anchor='w')

        self.method_var = tk.StringVar(value='contour')
        methods = [
            ('Contour Tracing (fastest)', 'contour'),
            ('Edge Detection (lines)', 'edge'),
            ('Adaptive Threshold (bold)', 'threshold'),
            ('Floyd-Steinberg Dither (tone)', 'dither'),
            ('Weighted Stippling (art)', 'stipple'),
        ]
        for text, val in methods:
            rb = tk.Radiobutton(inner, text=text, variable=self.method_var, value=val,
                               bg='#1a1e2e', fg='#e2e8f0', selectcolor='#2d3448',
                               activebackground='#1a1e2e', activeforeground='#00b4d8',
                               font=('Helvetica', 10), anchor='w')
            rb.pack(fill='x', pady=1)

        # Max points slider
        pts_frame = ttk.Frame(inner, style='Card.TFrame')
        pts_frame.pack(fill='x', pady=(6, 0))
        ttk.Label(pts_frame, text="Max points:", style='Dark.TLabel').pack(side='left')
        self.points_var = tk.IntVar(value=2000)
        self.points_slider = tk.Scale(pts_frame, from_=200, to=5000, orient='horizontal',
                                       variable=self.points_var, bg='#1a1e2e', fg='#e2e8f0',
                                       troughcolor='#2d3448', highlightthickness=0,
                                       font=('Helvetica', 9), length=160)
        self.points_slider.pack(side='right')

        tk.Button(inner, text="Process Image", command=self.process_image,
                  bg='#7209b7', fg='white', font=('Helvetica', 11, 'bold'),
                  relief='flat', padx=12, pady=4).pack(fill='x', pady=(8, 0))

    def build_serial_section(self, parent):
        frame = ttk.Frame(parent, style='Card.TFrame')
        frame.pack(fill='x', pady=(0, 8))
        inner = ttk.Frame(frame, style='Card.TFrame')
        inner.pack(fill='x', padx=10, pady=8)

        ttk.Label(inner, text="CONNECTION", style='Header.TLabel').pack(anchor='w')

        port_frame = ttk.Frame(inner, style='Card.TFrame')
        port_frame.pack(fill='x', pady=(4, 0))

        ttk.Label(port_frame, text="Port:", style='Dark.TLabel').pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=22)
        self.port_combo.pack(side='left', padx=(4, 4))

        tk.Button(port_frame, text="↻", command=self.refresh_ports,
                  bg='#2d3448', fg='white', font=('Helvetica', 10),
                  relief='flat', width=3).pack(side='left')

        self.refresh_ports()

        btn_frame = ttk.Frame(inner, style='Card.TFrame')
        btn_frame.pack(fill='x', pady=(6, 0))

        tk.Button(btn_frame, text="Connect", command=self.connect_serial,
                  bg='#06d6a0', fg='#0f1117', font=('Helvetica', 11, 'bold'),
                  relief='flat', padx=12, pady=4).pack(side='left')
        tk.Button(btn_frame, text="Disconnect", command=self.disconnect_serial,
                  bg='#ef4444', fg='white', font=('Helvetica', 11, 'bold'),
                  relief='flat', padx=12, pady=4).pack(side='left', padx=(6, 0))

        self.serial_status = ttk.Label(inner, text="Not connected", style='Status.TLabel')
        self.serial_status.configure(foreground='#ef4444')
        self.serial_status.pack(anchor='w', pady=(4, 0))

    def build_drawing_section(self, parent):
        frame = ttk.Frame(parent, style='Card.TFrame')
        frame.pack(fill='x', pady=(0, 8))
        inner = ttk.Frame(frame, style='Card.TFrame')
        inner.pack(fill='x', padx=10, pady=8)

        ttk.Label(inner, text="DRAWING", style='Header.TLabel').pack(anchor='w')

        btn_frame = ttk.Frame(inner, style='Card.TFrame')
        btn_frame.pack(fill='x', pady=(6, 0))

        tk.Button(btn_frame, text="▶ START DRAWING", command=self.start_drawing,
                  bg='#ff6b35', fg='white', font=('Helvetica', 12, 'bold'),
                  relief='flat', padx=12, pady=6).pack(fill='x')

        tk.Button(btn_frame, text="⏹ STOP", command=self.stop_drawing_cmd,
                  bg='#ef4444', fg='white', font=('Helvetica', 10),
                  relief='flat', padx=12, pady=4).pack(fill='x', pady=(4, 0))

        tk.Button(btn_frame, text="🏠 HOME", command=self.go_home,
                  bg='#2d3448', fg='white', font=('Helvetica', 10),
                  relief='flat', padx=12, pady=4).pack(fill='x', pady=(4, 0))

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(inner, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=(8, 0))

        self.progress_label = ttk.Label(inner, text="Ready", style='Status.TLabel')
        self.progress_label.pack(anchor='w', pady=(4, 0))

    def build_preview_section(self, parent):
        inner = ttk.Frame(parent, style='Card.TFrame')
        inner.pack(fill='both', expand=True, padx=10, pady=8)

        ttk.Label(inner, text="PREVIEW", style='Header.TLabel').pack(anchor='w')
        self.canvas = tk.Canvas(inner, bg='#13151d', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, pady=(6, 0))
        self.info_label = ttk.Label(inner, text="Load an image to begin", style='Dark.TLabel')
        self.info_label.pack(anchor='w', pady=(4, 0))


    # FUNCTIONS
   

    def refresh_ports(self):
        if SERIAL_AVAILABLE:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        else:
            ports = ['pyserial not installed']

        
        if sys.platform == 'darwin':
            mac_ports = glob.glob('/dev/cu.usbmodem*')
            for p in mac_ports:
                if p not in ports:
                    ports.append(p)

        self.port_combo['values'] = ports
        if ports:
            self.port_var.set(ports[0])

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        self.image_path = path
        img = cv2.imread(path)
        img = cv2.resize(img, (400, 300))
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.img_label.configure(text=os.path.basename(path))
        self.show_preview_image(self.img_gray, "Original image loaded")

    def show_preview_image(self, img, info_text=""):
        h, w = img.shape[:2]
        canvas_w = self.canvas.winfo_width() or 600
        canvas_h = self.canvas.winfo_height() or 500
        scale = min(canvas_w / w, canvas_h / h, 1.5)
        new_w, new_h = int(w * scale), int(h * scale)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        x = canvas_w // 2
        y = canvas_h // 2
        self.canvas.create_image(x, y, image=self.photo, anchor='center')
        if info_text:
            self.info_label.configure(text=info_text)

    def show_preview_coords(self, coords_x, coords_y, img_shape, info_text=""):
        canvas_w = self.canvas.winfo_width() or 600
        canvas_h = self.canvas.winfo_height() or 500
        h, w = img_shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.5)
        offset_x = (canvas_w - w * scale) / 2
        offset_y = (canvas_h - h * scale) / 2
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            offset_x, offset_y,
            offset_x + w * scale, offset_y + h * scale,
            outline='#2d3448', width=1
        )
        #  drawing
        for i in range(len(coords_x)):
            px = offset_x + coords_x[i] * scale
            py = offset_y + coords_y[i] * scale
            self.canvas.create_oval(px - 1, py - 1, px + 1, py + 1, fill='#00b4d8', outline='')

        if info_text:
            self.info_label.configure(text=info_text)

    def process_image(self):
        if self.img_gray is None:
            messagebox.showwarning("No Image", "Load an image first")
            return
        method = self.method_var.get()
        self.max_points = self.points_var.get()
        self.progress_label.configure(text=f"Processing: {method}...")
        self.root.update()
        params = {
            'max_points': self.max_points,
            'low_thresh': 50,
            'high_thresh': 150,
            'block_size': 11,
            'C': 2,
        }

        methods = {
            'edge': ('Edge Detection', method_edge_detection),
            'threshold': ('Adaptive Threshold', method_adaptive_threshold),
            'dither': ('Floyd-Steinberg Dither', method_dithering),
            'contour': ('Contour Tracing', method_contour_tracing),
            'stipple': ('Weighted Stippling', method_stippling),
        }

        name, func = methods[method]
        px_x, px_y, preview = func(self.img_gray.copy(), params)

        if len(px_x) == 0:
            messagebox.showwarning("No Points", "Processing generated no points")
            return

        
        if len(px_x) > self.max_points:
            idx = np.random.choice(len(px_x), self.max_points, replace=False)
            px_x, px_y = px_x[idx], px_y[idx]

        
        if method != 'contour':
            px_x, px_y = optimise_path(px_x, px_y)

        self.coords_x = px_x
        self.coords_y = px_y

        #convert to coordinates - maintain aspect ratio
        h, w = self.img_gray.shape
        img_ratio = w / h
        area_ratio = self.draw_width / self.draw_height
        
        if img_ratio > area_ratio:
            #fit to width
            actual_width = self.draw_width
            actual_height = self.draw_width / img_ratio
        else:
            #fit to height
            actual_height = self.draw_height
            actual_width = self.draw_height * img_ratio
        
        # centre the image in the drawing area
        x_offset = self.offset_x + (self.draw_width - actual_width) / 2
        y_offset = self.offset_y + (self.draw_height - actual_height) / 2
        
        scale_x = actual_width / w
        scale_y = actual_height / h
        scale_y = self.draw_height / h
        self.coords_mm_x = px_x * scale_x + x_offset
        self.coords_mm_y = px_y * scale_y + y_offset

        # Show preview
        self.show_preview_coords(
            px_x, px_y, self.img_gray.shape,
            f"{name}: {len(px_x)} points — Ready to draw"
        )

        self.progress_label.configure(text=f"{name}: {len(px_x)} points ready")

    def connect_serial(self):
        if not SERIAL_AVAILABLE:
            messagebox.showerror("Error", "pyserial not installed.\nRun: pip3 install pyserial")
            return

        port = self.port_var.get()
        if not port:
            messagebox.showwarning("No Port", "Select a serial port")
            return

        try:
            self.serial_conn = serial.Serial(port, 115200, timeout=5)
            time.sleep(2)

            # Read startup message
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode().strip()
                print(f"Arduino: {line}")

            self.serial_status.configure(text=f"Connected: {port}", foreground='#06d6a0')
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.serial_status.configure(text=f"Error: {str(e)[:30]}", foreground='#ef4444')

    def disconnect_serial(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.serial_conn = None
        self.serial_status.configure(text="Disconnected", foreground='#ef4444')

    def send_command(self, cmd):
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        self.serial_conn.write(f"{cmd}\n".encode())
        response = self.serial_conn.readline().decode().strip()
        return response

    def go_home(self):
        if not self.serial_conn:
            messagebox.showwarning("Not Connected", "Connect to Arduino first")
            return
        response = self.send_command("H")
        self.progress_label.configure(text=f"Home: {response}")

    def start_drawing(self):
        if self.coords_mm_x is None:
            messagebox.showwarning("No Data", "Process an image first")
            return
        if not self.serial_conn:
            messagebox.showwarning("Not Connected", "Connect to Arduino first")
            return
        if self.is_drawing:
            return

        self.is_drawing = True
        self.stop_drawing = False

        # Run drawing in a separate thread so GUI stays responsive
        thread = threading.Thread(target=self.drawing_thread, daemon=True)
        thread.start()

    def drawing_thread(self):
        n = len(self.coords_mm_x)
        start_time = time.time()

        # Home first
        self.send_command("H")

        for i in range(n):
            if self.stop_drawing:
                break

            x = self.coords_mm_x[i]
            y = self.coords_mm_y[i]

            cmd = f"G {x:.1f} {y:.1f}"
            response = self.send_command(cmd)

            # Update progress
            progress = (i + 1) / n * 100
            elapsed = time.time() - start_time
            if i > 0:
                eta = elapsed / (i + 1) * (n - i - 1)
            else:
                eta = 0

            self.progress_var.set(progress)
            self.root.after(0, lambda p=progress, e=eta, idx=i: 
                self.progress_label.configure(
                    text=f"Drawing: {idx+1}/{n} ({p:.0f}%) — ETA: {e:.0f}s"
                ))

        # Home when done
        self.send_command("H")
        
        total = time.time() - start_time
        self.is_drawing = False
        self.root.after(0, lambda: self.progress_label.configure(
            text=f"Complete! {n} points in {total:.0f}s"
        ))

    def stop_drawing_cmd(self):
        self.stop_drawing = True
        self.is_drawing = False
        self.progress_label.configure(text="Drawing stopped")
# RUN

if __name__ == '__main__':
    root = tk.Tk()
    app = PolargraphGUI(root)
    root.mainloop()
