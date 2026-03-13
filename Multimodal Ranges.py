## Multi modal sensor characterisation

import numpy as np
import Radar_Performance as radar
import LiDAR_Performance as lidar
from Radar_Performance import lambda_radar, Pr_radar, Pt_radar

Uncertainty_Ellipse_Radius = 5000
Sensor_FoV = np.arange(1, 180, 1)

Minimum_Range = Uncertainty_Ellipse_Radius + Uncertainty_Ellipse_Radius / np.sin(np.radians(Sensor_FoV / 2))

n = len(Sensor_FoV)

radar_Pt = np.zeros(n)
lidar_Pt = np.zeros(n)

for i in range(n):

    FoV = Sensor_FoV[i]
    R = Minimum_Range[i]
    print(FoV)
    print(R)
    radar_gain = radar.Gain_Approx(FoV)

    radar_Pt[i] = radar.compute_missing_radar(Pr_radar = 7.8E-18,Pt_radar = None,G_radar=radar_gain,lambda_radar = 0.056,RCS_radar =1, R_radar=R,L_radar =  1) # [Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar]
    divergence_rad = FoV * np.pi / 180
    lidar_Pt[i] = lidar.compute_pulse_energy(R,divergence_rad) * 1000 #(Pt @1khz)




# ----- Plot -----
import matplotlib.pyplot as plt
# ----- Plot -----

fig, ax1 = plt.subplots()

# Minimum range
l1, = ax1.plot(Sensor_FoV, Minimum_Range,color="black", label="Minimum Range")
ax1.set_xlabel("Sensor Field of View (deg)")
ax1.set_ylabel("Minimum Range (m)")
ax1.grid(True)

# Power axis
ax2 = ax1.twinx()

l2, = ax2.plot(Sensor_FoV, lidar_Pt, label="LiDAR Pulse Energy")
l3, = ax2.plot(Sensor_FoV, radar_Pt, label="Radar Transmit Power")

ax2.set_ylabel("Transmit Energy / Power")
ax2.set_yscale("log")

# Combined legend
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.title("Sensor FoV Trade vs Minimum Range and Required Transmit Energy")

plt.show()