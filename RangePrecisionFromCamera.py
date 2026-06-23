"""
RangePrecisionFromCamera.py

Estimates range precision achievable by counting pixels subtended by a target
of known width, given a camera with a fixed angular resolution (FoV per pixel).

Key idea: a target at range R subtends an angular width of atan(W/R). Dividing
by the FoV-per-pixel gives a pixel count. Because the count is discrete, a blur
circle of radius Blur_Circle_Radius_px adds uncertainty of +/-1 pixel, which maps
back to an uncertainty in inferred range. The gradient of range w.r.t. pixel count
gives the sensitivity: how many metres of range error correspond to one pixel.
"""

import numpy as np
import matplotlib.pyplot as plt

TargetWidth = 1       # known target width (m)
FoVpp       = 0.00049  # camera angular resolution (degrees per pixel)
Range       = 7500    # reference range for the single-point calculation (m)

# ── Single-point calculation ───────────────────────────────────────────
# Angular width in degrees, then convert to pixel count
Sub_Tgt_Ang   = np.atan(TargetWidth / Range) * 360 / np.pi
True_Pxl_Width = Sub_Tgt_Ang / FoVpp
print("True Pixel Width:", True_Pxl_Width)

Blur_Circle_Radius_px = 1  # optical blur adds +/-1 pixel of uncertainty

# Pixel count bounds due to blur, then back-project to metres
Px_W_min = np.floor(True_Pxl_Width - Blur_Circle_Radius_px)
Px_W_max = np.ceil (True_Pxl_Width + Blur_Circle_Radius_px)

Calc_W_min = Range * np.tan(FoVpp * Px_W_min * np.pi / 180)
Calc_W_max = Range * np.tan(FoVpp * Px_W_max * np.pi / 180)

W_uncertainty_from_blur = Calc_W_max - Calc_W_min
print("Width range:", Calc_W_min, "to", Calc_W_max, ". Uncertainty:", W_uncertainty_from_blur)

# ── Range vs pixel count sensitivity ──────────────────────────────────
# Build a pixel count array and compute the corresponding range for each count
pxl_count_array = np.arange(5, 120, 1)
# Range at which the target fills exactly pxl_count pixels
pxl_count_range = TargetWidth / np.tan(pxl_count_array * FoVpp * np.pi / 180)

# Gradient: metres of range change per additional pixel — steeper = coarser range resolution
pxlrange_grad = np.gradient(pxl_count_range)
print(pxlrange_grad)

# ── Pixel width vs range over a continuous range array ────────────────
Range_step_array = np.arange(min(pxl_count_range), max(pxl_count_range), 100)
Sub_Tgt_Ang      = np.atan(TargetWidth / Range_step_array) * 180 / np.pi
True_Pxl_Width   = Sub_Tgt_Ang / FoVpp

fig, ax1 = plt.subplots()

# Left axis: range sensitivity (m per pixel) — tells you how coarse range estimation is
color = 'tab:red'
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Px Grad / m ', color=color)
ax1.plot(pxl_count_range[1:len(pxl_count_range)-1], -1 * pxlrange_grad[1:len(pxl_count_range)-1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Right axis: pixel width vs range — tells you how many pixels the target fills
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Px Width', color=color)
ax2.plot(Range_step_array, True_Pxl_Width, color=color)
ax2.tick_params(axis='y', labelcolor=color)

title = "Target Width " + str(TargetWidth) + " m"
plt.title(title)
fig.tight_layout()
plt.show()
