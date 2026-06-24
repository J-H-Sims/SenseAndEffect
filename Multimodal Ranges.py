"""
Multimodal Ranges.py

Trade study comparing flash LiDAR, scanning LiDAR, radar, and camera sensor
performance as a function of sensor field of view (FoV).

For each FoV, a minimum stand-off range is derived from an uncertainty ellipse
assumption, and then the required transmit power or resolution for each sensor
type is computed at that range. The goal is to understand the FoV-driven trade
between sensitivity/power and coverage area.
"""

import numpy as np
import Radar_Performance as radar
import LiDAR_Performance as lidar

# Minimum stand-off: sensor must be far enough that the uncertainty ellipse
# subtends a small enough angle. Minimum_Range grows with FoV because a wider
# beam needs to stay further back to resolve the target within the uncertainty volume.
Uncertainty_Ellipse_Radius = 5000
Sensor_FoV    = np.concatenate((np.arange(0.05, 2, 0.5), np.arange(2, 180, 10)))
Minimum_Range = Uncertainty_Ellipse_Radius + Uncertainty_Ellipse_Radius / np.sin(np.radians(Sensor_FoV / 2))

n = len(Sensor_FoV)

radar_Pt       = np.zeros(n)  # radar peak transmit power (W)
ss_lidar_Pe    = np.zeros(n)  # flash LiDAR pulse energy (J)
scan_lidar_Pt  = np.zeros(n)  # scanning LiDAR average power (W)
scan_lidar_Pe  = np.zeros(n)  # scanning LiDAR pulse energy (J)
The_Sun        = np.zeros(n)  # solar power across the scan surface (W), for reference
camera_FoV_per_px = np.zeros(n)
camera_H_Res   = np.zeros(n)
radar_P        = np.zeros(n)  # radar average power (W)

for i in range(n):
    FoV = Sensor_FoV[i]
    R   = Minimum_Range[i]
    print(FoV)

    area_scan_rate = 1  # Hz — rate at which the full FoV must be scanned

    # ── Radar ─────────────────────────────────────────────────────────
    pulse_duration = 5e-6

    lambda_radar           = 0.056
    radar_aperture_diameter = 0.3
    # Gain_Approx clamps beamwidth to the physical minimum for the aperture
    radar_gain = radar.Gain_Approx(FoV, lambda_radar, radar_aperture_diameter)
    Pr_radar   = 0.9E-15   # minimum detectable received power (W)
    RCS_radar  = 20        # target radar cross-section (m^2)
    L_radar    = 1         # system loss factor

    # Solve radar range equation for the peak transmit power needed to reach Pr_radar at range R
    radar_Pt[i] = radar.radar_solve_transmit_power(Pr_radar, radar_gain, lambda_radar, RCS_radar, R, L_radar)

    # Average power = peak power × pulse duration × scan rate (energy per unit time)
    radar_P[i] = radar_Pt[i] * pulse_duration * area_scan_rate

    # ── Flash LiDAR ───────────────────────────────────────────────────
    # Flash LiDAR illuminates the full FoV in one pulse; pulse energy scales with range and FoV
    divergence_rad  = np.radians(FoV)
    ss_lidar_Pe[i]  = lidar.compute_pulse_energy(R, divergence_rad) * area_scan_rate

    # ── Scanning LiDAR ────────────────────────────────────────────────
    spot_divergence_rad = 1.8e-3  # narrow spot beam half-angle (rad)
    scan_spot_area      = 2 * np.pi * (1 - np.cos(spot_divergence_rad / 2)) * R**2
    divergence_steradian = 2 * np.pi * (1 - np.cos(divergence_rad / 2))
    scan_surface_area    = divergence_steradian * R**2

    # Polling rate: how many spot pulses per second are needed to cover the FoV at area_scan_rate Hz
    polling_rate    = area_scan_rate * scan_surface_area / scan_spot_area
    scan_lidar_Pe[i] = lidar.compute_pulse_energy(R, spot_divergence_rad)
    scan_lidar_Pt[i] = scan_lidar_Pe[i] * polling_rate  # average power = energy × pulse rate

    ss_lidar_Pe[i] = lidar.compute_pulse_energy(R, divergence_rad) * area_scan_rate

    The_Sun[i] = 1380 * scan_surface_area  # solar constant × solid scan area (W), reference only

    # ── Camera ────────────────────────────────────────────────────────
    pixel_fill_req = 0.5  # fraction of target that must be covered by a pixel for detection

    # Angular size of a 1m target at range R, divided by pixel fill requirement
    camera_FoV_per_px[i] = np.degrees(np.atan((1**0.5) / R)) / pixel_fill_req
    camera_H_Res[i]      = FoV / camera_FoV_per_px[i]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)

    # ── Top plot: minimum range and transmit power vs FoV ─────────────────
    l1, = ax1.plot(Sensor_FoV, Minimum_Range, color="black", label="Minimum Range")
    ax1.set_ylabel("Minimum Range (m)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    l2, = ax2.plot(Sensor_FoV, ss_lidar_Pe,    linestyle="--", label="Flash LiDAR Pulse Energy")
    l3, = ax2.plot(Sensor_FoV, radar_Pt,       linestyle="--", label="Radar Transmit Power")
    l4, = ax2.plot(Sensor_FoV, scan_lidar_Pt,                  label="Scan Lidar avg Power")
    l6, = ax2.plot(Sensor_FoV, radar_P,                        label="Radar avg Power")
    l7, = ax2.plot(Sensor_FoV, scan_lidar_Pe,  linestyle="--", label="Scan Lidar Pulse Energy")
    ax2.set_ylabel("Transmit Power")
    ax2.set_yscale("log")

    lines  = [l1, l2, l3, l4, l6, l7]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_ylim(bottom=0, top=400000)
    ax2.set_ylim(top=1000000)

    # ── Bottom plot: camera angular resolution and pixel count vs FoV ──────
    l8, = ax3.plot(Sensor_FoV, camera_FoV_per_px, color="purple", label="Camera FoV per pixel")
    ax3.set_ylabel("Camera FoV per pixel (deg)")
    ax3.set_xlabel("Sensor Field of View (deg)")
    ax3.grid(True)

    # Reference lines for known sensor products
    ax3.axhline(0.00048 / pixel_fill_req, linestyle="--")
    ax3.text(175,  0.00048 / pixel_fill_req, " CAVU",            va="bottom", ha="right")
    ax3.axhline(0.0002  / pixel_fill_req, linestyle="--")
    ax3.text(160,  0.0002  / pixel_fill_req, " SOP 200",         va="bottom", ha="right")
    ax3.axhline(0.00013 / pixel_fill_req, linestyle="--")
    ax3.text(150,  0.00013 / pixel_fill_req, " HEO Adler",       va="top",    ha="right")
    ax3.axhline(0.00137 / pixel_fill_req, linestyle="--")
    ax3.text(175,  0.00137 / pixel_fill_req, " Blackfly FL 100mm", va="bottom", ha="right")
    ax3.axhline(0.02, linestyle="--", color="red")

    ax3.set_ylabel("Camera FoV per pixel (deg)")
    ax4 = ax3.twinx()
    l9, = ax4.plot(Sensor_FoV, camera_H_Res, color="green", linestyle="-.", label="Camera Horizontal Resolution")
    ax4.set_ylabel("Camera Horizontal Resolution (px)")
    lines  = [l8, l9]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="best")

    plt.suptitle("Sensor FoV Trade vs Minimum Range and Required Transmit Energy")
    ax4.set_ylim(bottom=0, top=10000)
    ax3.set_ylim(bottom=0, top=0.01)
    plt.show()
