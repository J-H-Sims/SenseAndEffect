"""
SurfaceParameterFitting.py

Fits a Lambertian + specular BRDF model to measured LiDAR return data for each
material and writes the resulting parameters to lidar_params.json.

The model is:
    R(theta) = kd * cos(theta) + kr * cos(theta)^beta

where:
    kd   : diffuse (Lambertian) coefficient
    kr   : specular coefficient
    beta : specular lobe exponent (higher = narrower highlight)
    theta: incidence angle (radians)

Measured data are return fractions at discrete incidence angles (0-90 deg).
curve_fit optimises [kd, kr, beta] for each material. If fitting fails, a
fallback of [kd0, kr0, beta0] is stored so downstream code always has valid
parameters. The output JSON is consumed by CuboidLiDARModel.py and CuboidSolarModel.py.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# ── Measured reflectance data ─────────────────────────────────────────
# Values are return fractions (0-1) at each angle; raw data were in percent
data = {
    "Angle": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 90],
    "Brushed H Al":                    [v/100 for v in [99.6, 25.1, 5.6, 2.2, 1.3, 0.9, 0.0, 0.5, 0.1, 1.3, 1.0, 0.7, 0.4, 0.3, 0.1, 0.0]],
    "Brushed V Al":                    [v/100 for v in [100.2, 79.9, 57.8, 24.0, 5.0, 4.1, 2.4, 1.5, 0.9, 0.6, 0.5, 0.4, 0.2, 0.0, 0.0, 0.0]],
    "Smooth Mylar":                    [v/100 for v in [9.1, 3.5, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.4]],
    "Crinkled Mylar Average":          [v/100 for v in [3.7, 2.6, 2.2, 2.2, 0.9, 0.5, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.5, 1.2, 0.9, 0.9]],
    "Matte Solar Panel":               [v/100 for v in [0.9, 0.9, 0.4, 0.4, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.7, 0.9, 0.9]],
    "Black Anodised Al Brushed V":     [v/100 for v in [100.0, 61.9, 11.3, 3.0, 1.7, 1.5, 1.3, 0.0, 0.7, 0.4, 0.7, 1.0, 1.3, 1.3, 1.3, 0.0]],
    "Gloss Laminate on Black Background": [v/100 for v in [81.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
}

df         = pd.DataFrame(data)
angles_deg = df["Angle"].values
theta_plot = np.linspace(0, 90, 200)

def lidar_model(theta, kd, kr, beta):
    """Lambertian + specular BRDF model evaluated at incidence angle theta (degrees)."""
    theta     = np.deg2rad(theta)
    cos_theta = np.cos(theta)
    return kd * cos_theta + kr * cos_theta**beta

params = {}  # stores fitted [kd, kr, beta] per material

plt.figure(figsize=(10, 6))

for col in df.columns[1:]:
    y = df[col].values

    # Initial guesses: small diffuse, specular seeded from near-normal return, moderate lobe
    kd0   = 0.01
    kr0   = np.clip(y[0], 0, 1)
    beta0 = 20
    p0    = [kd0, kr0, beta0]

    try:
        popt, _ = curve_fit(lidar_model, angles_deg, y, p0=p0, bounds=([0, 0, 0], [1, 1, 1000]))
        params[col] = popt.tolist()
    except Exception as e:
        # Fit failed (e.g. data too noisy or degenerate) — use initial guess as fallback
        print(f"Fit failed for {col}: {e}. Using fallback.")
        params[col] = [kd0, kr0, beta0]

    plt.plot(angles_deg, y,                          'o', label=f"{col} measured")
    plt.plot(theta_plot, lidar_model(theta_plot, *params[col]), '-', label=f"{col} fit")

plt.xlabel("Incidence angle (deg)")
plt.ylabel("Return fraction")
plt.title("LiDAR scattering model fit per material")
plt.legend()
plt.grid()
plt.show()

# Write after the loop — guarantees params is fully populated before saving
with open("lidar_params.json", "w") as f:
    json.dump(params, f, indent=4)

print("Saved parameters to lidar_params.json:")
for mat, p in params.items():
    print(f"{mat}: {p}")
