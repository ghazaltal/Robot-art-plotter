# Robot Art: Polargraph Plotter for Algorithmic Art Generation

A wall-hanging drawing robot that turns any image into a pen drawing on a whiteboard. Built as my final-year BEng Computer Systems Engineering project at the University of Sheffield (2026), supervised by Prof. Andrew Maiden.



## Overview

A polargraph plotter hangs a pen holder (the "gondola") from two belts driven by stepper motors mounted at the top corners of a board. By changing the length of each belt, the gondola can reach any point in the drawing area.

The project combines hardware design, embedded firmware and an image processing pipeline:

```
Image ──► Python (image processing + path optimisation) ──► CSV coordinates
      ──► MATLAB / Python GUI (serial) ──► Arduino (inverse kinematics + stepping) ──► Motors
```

## Hardware

| Component | Details |
|---|---|
| Microcontroller | Arduino UNO R4 WiFi |
| Motors | 2 × NEMA 17 stepper motors |
| Drivers | 2 × A4988 stepper drivers |
| Gondola | Custom 3D-printed PLA pen holder (rounded design for whiteboard contact) |
| Drawing surface | Whiteboard with marker pens, mounted on a plywood frame |

## Image processing methods

Five methods convert an image into drawable coordinates, each with different artistic style:

| Method | Style | How it works |
|---|---|---|
| **Canny edge detection** | Line drawing | Gaussian blur, gradient detection to extract outlines |
| **Adaptive thresholding** | Bold graphic | Converts to black and white using local neighbourhood thresholds, preserving detail under uneven lighting |
| **Floyd–Steinberg dithering** | Halftone | Error diffusion turns greyscale tones into dot density |
| **Contour tracing** | Continuous lines | Traces connected boundaries into long continuous strokes |
| **Weighted Voronoi stippling** | Dot art | Places dots weighted by image darkness, refined with a simplified Lloyd relaxation |

### Path optimisation

Extracted points come out in no useful order, so the pen would jump randomly across the board. A nearest-neighbour TSP heuristic orders the points so each move goes to the closest unvisited point, greatly reducing total travel distance and drawing time.

## Firmware

The Arduino firmware:
- Receives target coordinates over serial (115200 baud)
- Converts each `(x, y)` point into belt lengths using the polargraph inverse kinematics:

  ```
  L1 = sqrt(x² + y²)
  L2 = sqrt((D − x)² + y²)
  ```
  where `D` is the distance between the motor shafts.
- Steps both motors together using Bresenham-style dual-motor stepping, so the gondola moves in a straight line

**Serial commands**

| Command | Action |
|---|---|
| `H` | Move to home position |
| `G x y` | Move to point (x, y) in mm |
| `P` | Report current position |

The board prints `POLARGRAPH READY` on startup.

## Repository structure

```
robot-art-plotter/
├── arduino/      # polargraph_arduino.ino (firmware)
├── python/       # polargraph_image_processor.py, polargraph_gui.py
├── matlab/       # serial sender script
└── pictures/     # input images and finished drawings
```

## Getting started

**1. Install Python dependencies**
```bash
pip install opencv-python numpy pillow scipy matplotlib pyserial
```

**2. Set your machine parameters.** Measure the distance between the motor shafts (`D`, in mm) and set it in the firmware, the Python config (`PlotterConfig`) and the MATLAB script. They must match.

**3. Upload the firmware.** Open `arduino/polargraph_arduino.ino` in the Arduino IDE and upload it to the UNO R4 WiFi.

**4. Process an image**
```bash
python python/polargraph_image_processor.py input.jpg --method all
```
Methods: `edge`, `threshold`, `dither`, `contour`, `stipple` or `all`. Outputs `coords_<method>.csv` (x, y in mm) plus preview images.

**5. Draw.** Run the GUI to preview and send coordinates:
```bash
python python/polargraph_gui.py
```
or run the MATLAB sender script and select a CSV file.

## Results

<!-- Add side-by-side outputs of the five methods from the same source image -->

| Original | Edge | Threshold | Dither | Contour | Stipple |
|---|---|---|---|---|---|
| | | | | | |

## Future work

- Servo pen lift, so the pen doesn't draw while travelling between strokes
- Microstepping for smoother, higher-resolution lines
- Spring-loaded pen holder for consistent pen pressure
- Neural style transfer as a sixth image processing method

## Author

**Gazl Talib**, BEng Computer Systems Engineering, University of Sheffield
<gazl.talib178@gmail.com>
