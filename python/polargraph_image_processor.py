"""
============================================
POLARGRAPH PLOTTER - Image Processing Pipeline
Robot Art Dissertation Project
============================================

This Python script processes images into drawing coordinates
for the polargraph plotter. It implements 5 different methods
and outputs coordinate files that MATLAB reads and sends to Arduino.

HYBRID PIPELINE:
  Python (this script) → processes image → saves coordinates as .csv
  MATLAB → reads .csv → sends coordinates to Arduino via serial

METHODS:
  1. Canny Edge Detection    - Extracts outlines/boundaries
  2. Adaptive Thresholding   - Binary conversion preserving local detail
  3. Floyd-Steinberg Dither  - Halftone dots for shading/tone
  4. Contour Tracing         - Connected continuous lines
  5. Weighted Voronoi Stippling - Artistic dot placement

Each method produces different artistic results.
Compare them in the dissertation!

USAGE:
  python polargraph_image_processor.py input_image.jpg --method all
  python polargraph_image_processor.py input_image.jpg --method edge
  python polargraph_image_processor.py input_image.jpg --method threshold
  python polargraph_image_processor.py input_image.jpg --method dither
  python polargraph_image_processor.py input_image.jpg --method contour
  python polargraph_image_processor.py input_image.jpg --method stipple

OUTPUT:
  coords_edge.csv, coords_threshold.csv, etc.
  Each CSV has columns: x, y (in mm)
  Also saves preview images of each method.

DEPENDENCIES:
  pip install opencv-python numpy pillow scipy matplotlib
============================================
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import argparse
import os
import sys


# ============================================
# CONFIGURATION
# ============================================

class PlotterConfig:
    """Machine parameters - CHANGE THESE TO MATCH YOUR SETUP"""
    D = 600              # Distance between motors (mm)
    DRAW_WIDTH = 400     # Drawing area width (mm)
    DRAW_HEIGHT = 300    # Drawing area height (mm)
    OFFSET_X = 100       # Drawing area left offset from left motor (mm)
    OFFSET_Y = 120       # Drawing area top offset below motor line (mm)
    MAX_POINTS = 3000    # Maximum points to send (more = slower but more detail)
    IMAGE_SIZE = (400, 300)  # Processing resolution (width, height) in pixels


# ============================================
# METHOD 1: CANNY EDGE DETECTION
# ============================================
# 
# HOW IT WORKS:
# 1. Convert image to grayscale
# 2. Apply Gaussian blur to reduce noise
# 3. Canny detector finds intensity gradients (where brightness changes sharply)
# 4. Non-maximum suppression thins edges to 1-pixel wide
# 5. Hysteresis thresholding connects strong and weak edges
#
# RESULT: Clean outlines of shapes and boundaries
# BEST FOR: Images with clear shapes, portraits, objects with defined edges
# ARTISTIC STYLE: Line drawing / sketch look
#
# PARAMETERS:
#   - blur_kernel: Size of Gaussian blur (larger = fewer noisy edges)
#   - low_threshold: Weak edge threshold (lower = more edges detected)
#   - high_threshold: Strong edge threshold (higher = only strongest edges)
# ============================================

def method_edge_detection(img_gray, config):
    """
    Canny Edge Detection
    Finds boundaries where pixel intensity changes sharply.
    """
    print("  [1/4] Applying Gaussian blur to reduce noise...")
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    
    print("  [2/4] Running Canny edge detector...")
    # Low and high thresholds control sensitivity
    # Low threshold: edges below this are discarded
    # High threshold: edges above this are kept
    # Edges between are kept only if connected to strong edges
    edges = cv2.Canny(blurred, 50, 150)
    
    print("  [3/4] Extracting edge coordinates...")
    # Find all white pixels (edge points)
    points = np.column_stack(np.where(edges > 0))
    # points[:, 0] = row (Y), points[:, 1] = col (X)
    
    # Convert to (x, y) in mm
    coords = pixels_to_mm(points[:, 1], points[:, 0], img_gray.shape, config)
    
    print(f"  [4/4] Found {len(coords)} edge points")
    
    return coords, edges


# ============================================
# METHOD 2: ADAPTIVE THRESHOLDING
# ============================================
#
# HOW IT WORKS:
# 1. Convert to grayscale
# 2. For each pixel, calculate threshold based on LOCAL neighbourhood
#    (unlike global thresholding which uses one value for entire image)
# 3. Pixel becomes white if brighter than local average, black otherwise
# 4. This preserves detail in both bright and dark areas
#
# RESULT: High-contrast black and white image preserving local detail
# BEST FOR: Images with uneven lighting, text, detailed textures
# ARTISTIC STYLE: Bold graphic / woodcut / stencil look
#
# PARAMETERS:
#   - block_size: Size of neighbourhood for local threshold (must be odd)
#   - C: Constant subtracted from mean (higher = less black area)
# ============================================

def method_adaptive_threshold(img_gray, config):
    """
    Adaptive Thresholding
    Converts to binary using locally-computed thresholds.
    Preserves detail better than a single global threshold.
    """
    print("  [1/4] Applying light blur...")
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    print("  [2/4] Computing adaptive threshold...")
    # ADAPTIVE_THRESH_GAUSSIAN_C: threshold = weighted sum of neighbourhood - C
    # block_size=11: uses 11x11 pixel neighbourhood
    # C=2: subtracts 2 from the mean (fine-tunes black/white balance)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert so dark areas become drawing points
        blockSize=11,
        C=2
    )
    
    print("  [3/4] Extracting coordinates from black regions...")
    points = np.column_stack(np.where(binary > 0))
    coords = pixels_to_mm(points[:, 1], points[:, 0], img_gray.shape, config)
    
    print(f"  [4/4] Found {len(coords)} points")
    
    return coords, binary


# ============================================
# METHOD 3: FLOYD-STEINBERG DITHERING
# ============================================
#
# HOW IT WORKS:
# 1. Process image pixel by pixel, left to right, top to bottom
# 2. For each pixel, decide: is it closer to black or white?
# 3. Set it to black or white accordingly
# 4. Calculate the ERROR (difference between original and chosen value)
# 5. DISTRIBUTE that error to neighbouring pixels:
#        [ current ]  7/16 →
#    3/16 ↙  5/16 ↓  1/16 ↘
# 6. This error diffusion creates the illusion of grey tones
#    using only black and white dots
#
# RESULT: Halftone-style image where dot density represents shade
# BEST FOR: Photographs, images with gradients/shading
# ARTISTIC STYLE: Newspaper print / halftone / stippled shading
#
# The plotter draws dots at each black pixel position,
# so denser areas appear darker — creating the illusion of tone.
# ============================================

def method_dithering(img_gray, config):
    """
    Floyd-Steinberg Dithering
    Converts grayscale to binary using error diffusion.
    Creates halftone effect where dot density = darkness.
    """
    print("  [1/4] Preparing image for dithering...")
    # Work with float copy to handle error diffusion
    img = img_gray.astype(np.float64)
    h, w = img.shape
    
    print("  [2/4] Running Floyd-Steinberg error diffusion...")
    # Process each pixel
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            # Quantize to black (0) or white (255)
            new_pixel = 255.0 if old_pixel > 127 else 0.0
            img[y, x] = new_pixel
            
            # Calculate quantization error
            error = old_pixel - new_pixel
            
            # Distribute error to neighbours (Floyd-Steinberg kernel)
            if x + 1 < w:
                img[y, x + 1] += error * 7.0 / 16.0       # Right
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3.0 / 16.0  # Bottom-left
                img[y + 1, x] += error * 5.0 / 16.0        # Bottom
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1.0 / 16.0  # Bottom-right
    
    # Convert back to uint8
    dithered = np.clip(img, 0, 255).astype(np.uint8)
    # Invert so black dots are the drawing points
    dithered_inv = 255 - dithered
    
    print("  [3/4] Extracting dot coordinates...")
    points = np.column_stack(np.where(dithered_inv > 0))
    coords = pixels_to_mm(points[:, 1], points[:, 0], img_gray.shape, config)
    
    print(f"  [4/4] Found {len(coords)} dither points")
    
    return coords, dithered


# ============================================
# METHOD 4: CONTOUR TRACING
# ============================================
#
# HOW IT WORKS:
# 1. Convert to binary using Otsu's method (automatic threshold selection)
# 2. Find contours — continuous chains of connected edge pixels
#    (unlike Canny which gives scattered edge points, contours are ORDERED)
# 3. Each contour is a sequence of (x,y) points forming a closed or open curve
# 4. Approximate contours to reduce point count (Douglas-Peucker algorithm)
# 5. Output as ordered drawing paths
#
# RESULT: Connected continuous lines that the pen follows without lifting
# BEST FOR: Images with clear shapes, logos, silhouettes
# ARTISTIC STYLE: Clean vector-like line art
#
# KEY ADVANTAGE: Because contours are ORDERED sequences, the plotter
# draws continuous lines instead of jumping between scattered dots.
# This is faster and produces cleaner results.
# ============================================

def method_contour_tracing(img_gray, config):
    """
    Contour Tracing
    Finds connected boundaries and traces them as continuous paths.
    Produces ordered coordinates for smooth line drawing.
    """
    print("  [1/5] Applying blur and binary threshold...")
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    # Otsu's method automatically finds optimal threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print("  [2/5] Finding contours...")
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"        Found {len(contours)} raw contours")
    
    print("  [3/5] Simplifying contours (Douglas-Peucker)...")
    # Approximate contours to reduce points while keeping shape
    simplified = []
    for contour in contours:
        # epsilon controls simplification: larger = fewer points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Filter out very small contours (noise)
        if cv2.arcLength(approx, True) > 10:
            simplified.append(approx)
    
    print(f"        Kept {len(simplified)} significant contours")
    
    print("  [4/5] Converting to drawing coordinates...")
    # Flatten all contour points into ordered path
    all_coords = []
    for contour in simplified:
        for point in contour:
            px, py = point[0]
            all_coords.append([px, py])
        # Add first point again to close the contour
        if len(contour) > 0:
            all_coords.append([contour[0][0][0], contour[0][0][1]])
    
    all_coords = np.array(all_coords)
    if len(all_coords) == 0:
        print("  WARNING: No contours found!")
        return np.array([]), binary
    
    coords = pixels_to_mm(all_coords[:, 0], all_coords[:, 1], img_gray.shape, config)
    
    # Draw contours on preview image
    contour_img = np.zeros_like(img_gray)
    cv2.drawContours(contour_img, simplified, -1, 255, 1)
    
    print(f"  [5/5] Generated {len(coords)} contour points")
    
    return coords, contour_img


# ============================================
# METHOD 5: WEIGHTED VORONOI STIPPLING
# ============================================
#
# HOW IT WORKS:
# 1. Generate random seed points across the image
# 2. Weight the density of points by image darkness
#    (darker areas get more points)
# 3. Use Lloyd's relaxation algorithm:
#    a) Compute Voronoi diagram of current points
#    b) Move each point to the weighted centroid of its Voronoi cell
#    c) Repeat — points gradually settle into optimal positions
# 4. After several iterations, points are evenly distributed
#    but denser where the image is darker
#
# RESULT: Artistic dot distribution that recreates the image
# BEST FOR: Portraits, photographs, artistic rendering
# ARTISTIC STYLE: Pointillism / stipple art / ink dots
#
# This is the most computationally expensive method but produces
# the most visually striking and "artistic" results.
#
# SIMPLIFIED VERSION: Instead of full Voronoi, we use a
# probability-based sampling approach which is faster and
# produces similar results for the plotter's resolution.
# ============================================

def method_stippling(img_gray, config):
    """
    Weighted Stippling (simplified Voronoi approach)
    Places dots with density proportional to image darkness.
    Produces pointillist art effect.
    """
    print("  [1/4] Computing darkness-weighted probability map...")
    # Invert so dark areas have high values
    inverted = 255.0 - img_gray.astype(np.float64)
    
    # Blur slightly for smoother distribution
    inverted = cv2.GaussianBlur(inverted, (5, 5), 2.0)
    
    # Normalize to probability distribution
    prob_map = inverted / inverted.sum()
    
    print("  [2/4] Generating stipple points...")
    h, w = img_gray.shape
    num_points = min(config.MAX_POINTS, 2500)
    
    # Sample pixel positions based on darkness probability
    # Flatten probability map and sample indices
    flat_prob = prob_map.flatten()
    indices = np.random.choice(
        len(flat_prob),
        size=num_points,
        replace=False,
        p=flat_prob
    )
    
    # Convert flat indices back to (row, col)
    rows = indices // w
    cols = indices % w
    
    print("  [3/4] Running Lloyd's relaxation (5 iterations)...")
    # Simple relaxation: move each point toward local density centroid
    for iteration in range(5):
        new_rows = np.zeros_like(rows, dtype=np.float64)
        new_cols = np.zeros_like(cols, dtype=np.float64)
        
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            # Define local neighbourhood
            r_min = max(0, r - 5)
            r_max = min(h, r + 6)
            c_min = max(0, c - 5)
            c_max = min(w, c + 6)
            
            # Get local probability patch
            patch = inverted[r_min:r_max, c_min:c_max]
            if patch.sum() > 0:
                # Compute weighted centroid
                rr, cc = np.meshgrid(
                    np.arange(r_min, r_max),
                    np.arange(c_min, c_max),
                    indexing='ij'
                )
                total = patch.sum()
                new_rows[i] = (rr * patch).sum() / total
                new_cols[i] = (cc * patch).sum() / total
            else:
                new_rows[i] = r
                new_cols[i] = c
        
        rows = np.clip(new_rows.astype(int), 0, h - 1)
        cols = np.clip(new_cols.astype(int), 0, w - 1)
    
    # Convert to mm coordinates
    coords = pixels_to_mm(cols, rows, img_gray.shape, config)
    
    # Create preview image
    stipple_img = np.ones_like(img_gray) * 255
    for r, c in zip(rows, cols):
        cv2.circle(stipple_img, (int(c), int(r)), 1, 0, -1)
    
    print(f"  [4/4] Generated {len(coords)} stipple points")
    
    return coords, stipple_img


# ============================================
# UTILITY FUNCTIONS
# ============================================

def pixels_to_mm(px_x, px_y, img_shape, config):
    """
    Convert pixel coordinates to mm coordinates on the plotter.
    
    Pixel (0,0) = top-left of image
    Plotter (0,0) = left motor position
    
    The drawing area is offset from the motors by OFFSET_X and OFFSET_Y.
    """
    h, w = img_shape[:2]
    
    # Scale pixels to drawing area mm
    scale_x = config.DRAW_WIDTH / w
    scale_y = config.DRAW_HEIGHT / h
    
    mm_x = px_x * scale_x + config.OFFSET_X
    mm_y = px_y * scale_y + config.OFFSET_Y
    
    coords = np.column_stack([mm_x, mm_y])
    return coords


def subsample_points(coords, max_points):
    """Randomly subsample if too many points."""
    if len(coords) > max_points:
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
    return coords


def optimise_path_nearest_neighbour(coords):
    """
    Nearest-Neighbour Path Optimisation
    
    Problem: Edge detection gives points in raster order (top to bottom,
    left to right). If we draw them in this order, the pen jumps across
    the board constantly — slow and messy.
    
    Solution: Starting from the first point, always move to the nearest
    unvisited point next. This greedy algorithm doesn't find the optimal
    path (that's the Travelling Salesman Problem — NP-hard) but it gives
    a much better path than raster order.
    
    This is like a simplified TSP (Travelling Salesman Problem) solver.
    The pen "visits" each coordinate once, minimising total travel distance.
    
    Time complexity: O(n²) — fine for up to ~5000 points
    """
    n = len(coords)
    if n <= 1:
        return coords
    
    print(f"  Optimising path for {n} points (nearest-neighbour)...")
    
    visited = np.zeros(n, dtype=bool)
    path = np.zeros(n, dtype=int)
    path[0] = 0
    visited[0] = True
    
    for i in range(1, n):
        current = path[i - 1]
        current_pt = coords[current]
        
        # Calculate distances to all unvisited points
        dists = np.sum((coords - current_pt) ** 2, axis=1)
        dists[visited] = np.inf  # Exclude visited
        
        nearest = np.argmin(dists)
        path[i] = nearest
        visited[nearest] = True
        
        if i % 500 == 0:
            print(f"    Progress: {i}/{n}")
    
    return coords[path]


def save_coordinates(coords, filename, config):
    """Save coordinates as CSV file for MATLAB to read."""
    np.savetxt(filename, coords, delimiter=',', header='x_mm,y_mm',
               comments='', fmt='%.2f')
    print(f"  Saved {len(coords)} coordinates to {filename}")


def save_preview(img, title, filename):
    """Save a preview image showing the processing result."""
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved preview: {filename}")


def save_comparison(results, original, output_dir):
    """Save a side-by-side comparison of all methods."""
    n = len(results) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Each method
    for i, (name, preview_img, num_points) in enumerate(results):
        axes[i + 1].imshow(preview_img, cmap='gray')
        axes[i + 1].set_title(f'{name}\n({num_points} points)', fontsize=11)
        axes[i + 1].axis('off')
    
    # Hide unused subplots
    for j in range(len(results) + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Image Processing Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'comparison_all_methods.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison saved: {filepath}")


# ============================================
# MAIN PIPELINE
# ============================================

def process_image(image_path, method='all', output_dir='output'):
    """
    Main processing pipeline.
    
    Flow:
      1. Load image
      2. Resize to processing resolution
      3. Convert to grayscale
      4. Apply selected method(s)
      5. Subsample if too many points
      6. Optimise drawing path
      7. Save coordinates as CSV
      8. Save preview images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    config = PlotterConfig()
    
    # Load and prepare image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image '{image_path}'")
        sys.exit(1)
    
    # Resize
    img = cv2.resize(img, config.IMAGE_SIZE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Image size: {img_gray.shape[1]}x{img_gray.shape[0]} pixels")
    print(f"Drawing area: {config.DRAW_WIDTH}x{config.DRAW_HEIGHT} mm")
    print(f"Max points: {config.MAX_POINTS}")
    print()
    
    # Define methods
    methods = {
        'edge':      ('1. Canny Edge Detection',    method_edge_detection),
        'threshold': ('2. Adaptive Thresholding',    method_adaptive_threshold),
        'dither':    ('3. Floyd-Steinberg Dithering', method_dithering),
        'contour':   ('4. Contour Tracing',          method_contour_tracing),
        'stipple':   ('5. Weighted Stippling',       method_stippling),
    }
    
    # Select methods to run
    if method == 'all':
        selected = methods.keys()
    elif method in methods:
        selected = [method]
    else:
        print(f"ERROR: Unknown method '{method}'")
        print(f"Available: {', '.join(methods.keys())}, all")
        sys.exit(1)
    
    results = []
    
    for key in selected:
        name, func = methods[key]
        print(f"{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")
        
        # Run method
        coords, preview_img = func(img_gray.copy(), config)
        
        if len(coords) == 0:
            print(f"  WARNING: No points generated, skipping.\n")
            continue
        
        # Subsample if needed
        coords = subsample_points(coords, config.MAX_POINTS)
        
        # Optimise path (skip for contour — already ordered)
        if key != 'contour':
            coords = optimise_path_nearest_neighbour(coords)
        
        # Save outputs
        csv_path = os.path.join(output_dir, f'coords_{key}.csv')
        save_coordinates(coords, csv_path, config)
        
        preview_path = os.path.join(output_dir, f'preview_{key}.png')
        save_preview(preview_img, name, preview_path)
        
        results.append((name, preview_img, len(coords)))
        print()
    
    # Save comparison image
    if len(results) > 1:
        save_comparison(results, img_gray, output_dir)
    
    print("="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"\nOutput files in: {output_dir}/")
    print("CSV files ready for MATLAB to send to Arduino.")
    print("\nIn MATLAB, load with:")
    print("  coords = readmatrix('coords_edge.csv');")
    print("  x = coords(:,1); y = coords(:,2);")


# ============================================
# COMMAND LINE INTERFACE
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Polargraph Plotter Image Processor'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--method', default='all',
                        choices=['all', 'edge', 'threshold', 'dither',
                                 'contour', 'stipple'],
                        help='Processing method (default: all)')
    parser.add_argument('--output', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--max-points', type=int, default=3000,
                        help='Maximum drawing points (default: 3000)')
    parser.add_argument('--width', type=float, default=400,
                        help='Drawing width in mm (default: 400)')
    parser.add_argument('--height', type=float, default=300,
                        help='Drawing height in mm (default: 300)')
    parser.add_argument('-D', type=float, default=600,
                        help='Motor distance in mm (default: 600)')
    
    args = parser.parse_args()
    
    # Update config
    PlotterConfig.D = args.D
    PlotterConfig.DRAW_WIDTH = args.width
    PlotterConfig.DRAW_HEIGHT = args.height
    PlotterConfig.OFFSET_X = (args.D - args.width) / 2
    PlotterConfig.MAX_POINTS = args.max_points
    
    process_image(args.image, args.method, args.output)
