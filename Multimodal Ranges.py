## Multi modal sensor characterisation

import numpy as np
import Radar_Performance as radar
import LiDAR_Performance as lidar
from LiDAR import target_area_m2
from Radar_Performance import lambda_radar, Pr_radar, Pt_radar, RCS_radar

Uncertainty_Ellipse_Radius = 5000
Sensor_FoV = np.concatenate((np.arange(0.05, 2, 0.1), np.arange(2, 180, 5)))

Minimum_Range = Uncertainty_Ellipse_Radius + Uncertainty_Ellipse_Radius / np.sin(np.radians(Sensor_FoV / 2))

n = len(Sensor_FoV)

radar_Pt = np.zeros(n)
ss_lidar_Pe = np.zeros(n)
scan_lidar_Pt = np.zeros(n)
scan_lidar_Pe = np.zeros(n)
The_Sun = np.zeros(n)
camera_FoV_per_px = np.zeros(n)
camera_H_Res = np.zeros(n)
radar_P = np.zeros(n)
for i in range(n):

    FoV = Sensor_FoV[i]
    R = Minimum_Range[i]
    print(FoV)
    #print(R)
    area_scan_rate = 1  # Hz

    ##Radar
    pulse_duration = 5e-6

    lambda_radar = 0.056
    radar_aperture_diameter = 0.3
    radar_gain = radar.Gain_Approx(FoV,lambda_radar,radar_aperture_diameter)

    radar_Pt[i] = radar.compute_missing_radar(Pr_radar = 7.8E-18,Pt_radar = None,G_radar=radar_gain,lambda_radar = lambda_radar,RCS_radar =1, R_radar=R,L_radar =  1) # [Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar]



    radar_P[i] = radar_Pt[i]*pulse_duration*area_scan_rate #this is to compare apples to apples, be Very careful with this as its a likely cause of Errors
    ##Lidar

    divergence_rad = np.radians(FoV)
    spot_divergence_rad = np.radians(lidar.diffraction_limited_divergence_deg(1,1470e-9,0.002))
    #spot_divergence_rad = 1.8e-3
    ss_lidar_Pe[i] = lidar.compute_pulse_energy(R,divergence_rad) * area_scan_rate
    scan_spot_area = 2 * np.pi * (1 - np.cos(spot_divergence_rad/2)) * R**2
    divergence_steradian = 2 * np.pi * (1 - np.cos(divergence_rad/2))
    #spot_size_m =range_m * np.tan(divergence_rad)
    scan_surface_area =  divergence_steradian * R**2
    polling_rate = area_scan_rate*scan_surface_area/scan_spot_area
    #print(scan_spot_area)
    #print(polling_rate)

    scan_lidar_Pe[i] = lidar.compute_pulse_energy(R,spot_divergence_rad)
    scan_lidar_Pt[i] = scan_lidar_Pe[i] * polling_rate

    ss_lidar_Pe[i] = lidar.compute_pulse_energy(R,divergence_rad) * area_scan_rate

    The_Sun[i]  = 1380 * scan_surface_area


    #Optical
    pixel_fill_req = 0.5
    #
    # sun_power_reflected = 1361 * 0.45 * target_area_m2
    #
    #
    # power_intensity_at_observer = sun_power_reflected/(2*np.pi*R**2)
    # collected_power = power_intensity_at_observer * np.pi*(0.05/2)**2
    # incident_light_angle = np.atan((target_area_m2**0.5)/R)
    # print(np.degrees(incident_light_angle))
    #
    # Nphoton_min = 40000 #from chatty
    # wavelength = 550e-9  # m (green light)
    # h = 6.626e-34
    # c = 3e8
    # E_photon = h * c / wavelength
    # N_photon_sec = collected_power / E_photon
    # print(N_photon_sec)
    # if N_photon_sec > Nphoton_min:
    #     print("Pixel will saturate")
    # else:
    #     print(f"Pixel receives {N_photon_sec:.1e} photons/sec, below saturation")
    #


    camera_FoV_per_px[i] = np.degrees(np.atan((1**0.5)/R))/pixel_fill_req
    camera_H_Res[i] = FoV/camera_FoV_per_px[i]






import matplotlib.pyplot as plt

fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)

# ----- TOP PLOT -----
# Minimum range
l1, = ax1.plot(Sensor_FoV, Minimum_Range, color="black", label="Minimum Range")
ax1.set_ylabel("Minimum Range (m)")
ax1.grid(True)

# Power axis
ax2 = ax1.twinx()

l2, = ax2.plot(Sensor_FoV, ss_lidar_Pe, label="Flash LiDAR Pulse Energy")
l3, = ax2.plot(Sensor_FoV, radar_Pt, label="Radar Transmit Power")
l4, = ax2.plot(Sensor_FoV, scan_lidar_Pt, label="Scan Lidar avg Power")
l5, = ax2.plot(Sensor_FoV, The_Sun, label="The Sun (across area of scan)")

l6, = ax2.plot(Sensor_FoV, radar_P, label="Radar avg Power ")

l7, = ax2.plot(Sensor_FoV, scan_lidar_Pe, label="Scan Lidar Pulse Energy ")

ax2.set_ylabel("Transmit Power")
ax2.set_yscale("log")

lines = [l1, l2, l3, l4, l5,l6, l7]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")
ax1.set_ylim(bottom=0, top = 200000)
ax2.set_ylim(bottom=10e-9, top = 10000)
# ----- BOTTOM PLOT -----
ax3.plot(Sensor_FoV, camera_FoV_per_px, color="purple", label="Camera FoV per pixel")
ax3.set_ylabel("Camera FoV per pixel (deg)")
ax3.set_xlabel("Sensor Field of View (deg)")
ax3.grid(True)

# Reference horizontal lines
ax3.axhline(0.00048/pixel_fill_req, linestyle="--")
ax3.text(175, 0.00048/pixel_fill_req, " CAVU", va="bottom", ha="right")
ax3.axhline(0.0002/pixel_fill_req, linestyle="--")
ax3.text(160, 0.0002/pixel_fill_req, " SOP 200", va="bottom", ha="right")
ax3.axhline(0.00013/pixel_fill_req, linestyle="--")
ax3.text(150, 0.00013/pixel_fill_req, " HEO Adler", va="top", ha="right")
ax3.axhline(0.00137/pixel_fill_req, linestyle="--")
ax3.text(175, 0.00137/pixel_fill_req, " Blackfly FL 100mm", va="bottom", ha="right")
ax3.axhline(0.02, linestyle="--", color="red")
ax3.text(175, 0.02, " Terma T3 ST", va="bottom", ha="right")
#ax3.axhline(0.002, linestyle="--", color="red")
#ax3.text(181, 0.002, " Teledyne ST", va="center", ha="left")

# Left and right axes
ax3.set_ylabel("Camera FoV per pixel (deg)")
ax4 = ax3.twinx()
ax4.plot(Sensor_FoV, camera_H_Res, color="green", linestyle="-.", label="Camera Horizontal Resolution")
ax4.set_ylabel("Camera Horizontal Resolution (px)")

# secax = ax3.secondary_xaxis('top', functions=(lambda fov: fov, lambda R: R))
# secax.set_xlabel("Minimum Range (m)")
# secax.set_ticks(Sensor_FoV)
# secax.set_xticklabels([f"{r:.1f}" for r in Minimum_Range])

plt.suptitle("Sensor FoV Trade vs Minimum Range and Required Transmit Energy")
ax4.set_ylim(bottom=0, top = 6000)
ax3.set_ylim(bottom=0, top = 0.0075)
plt.show()