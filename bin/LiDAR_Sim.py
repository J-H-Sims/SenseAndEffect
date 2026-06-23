"""
LiDAR_Sim.py

Legacy single-shot LiDAR simulation script.

Propagates a Gaussian beam to the target, reflects off a cuboid, then models
the return as a second Gaussian beam with the target's characteristic dimension
as the waist. Superseded by LiDAR_Performance.py which uses solid-angle
geometry instead of a two-stage Gaussian propagation.
"""

import numpy as np

import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import LiDAR_Performance as LP
from LiDAR_Performance import aperture_area_m2

# Cuboid geometry (meters)
length, width, height = 2.0, 2.0, 2

# Orientation (radians)
roll  = np.deg2rad(90)
pitch = np.deg2rad(0)
yaw   = np.deg2rad(0)

# All faces are the same brushed aluminium material
face_materials = ["Brushed V Al"] * 6

# Outgoing Gaussian beam parameters
w0         = 0.001          # beam waist at focus (m)
wavelength = 1.55e-6        # 1550 nm in metres
theta_user = np.deg2rad(0.1)
P_total    = 1.0            # emitted power (W)
z          = 10000.0        # range to target (m)

collector_aperture_diameter = 0.03
aperture_area = np.pi * (collector_aperture_diameter / 2)**2

# ── Stage 1: outgoing beam → target ───────────────────────────────────
X, Y    = 0, 0
I, w    = Gbeam.gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)

# reflected_energy is the total power leaving the cuboid (W) weighted by the BRDF
reflected_energy = I * target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)

# ── Stage 2: return beam → aperture ───────────────────────────────────
# Model the return as a Gaussian with waist equal to the mean target dimension
characteristic_width = (length + width + height) / 3
received_intensity, return_spot_diameter = Gbeam.gaussian_beam_wm2(
    0, 0, z, characteristic_width, wavelength, reflected_energy, theta_user)

received_energy = received_intensity * aperture_area
print(received_energy)

# ── Photon count and SNR ───────────────────────────────────────────────
PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT  = 299792458
# wavelength here is already in metres, so no 1e-9 factor needed
photon_energy_J   = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength * 1e-9)
print(photon_energy_J)
photons_per_pulse = received_energy / photon_energy_J
print(photons_per_pulse)

solar_photons = LP.compute_solar_photons(z)
SNR = photons_per_pulse / solar_photons
print(SNR * 100, "%")
