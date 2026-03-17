import numpy as np

import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import LiDAR_Performance as LP
from LiDAR_Performance import aperture_area_m2

# Cuboid size (meters)
length, width, height = 2.0, 2.0, 2

# Orientation in radians
roll = np.deg2rad(90)
pitch = np.deg2rad(0)
yaw = np.deg2rad(0)

# Materials per face: +X, -X, +Y, -Y, +Z, -Z
face_materials = [
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al"
]

w0 = 0.001  # 5 cm beam waist
wavelength = 1.55e-6  # 1550 nm
theta_user = np.deg2rad(0.1)  # 0.5 deg divergence
P_total = 1.0
z = 10000.0  # meters downrange

collector_aperture_diameter = 0.03
aperture_area = np.pi*(collector_aperture_diameter/2)**2


X, Y = 0,0
I, w = Gbeam.gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)

reflected_energy = I * target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)

characteristic_width = (length+width+height)/3

received_intensity, return_spot_diameter =  Gbeam.gaussian_beam_wm2(0, 0, z, characteristic_width, wavelength, reflected_energy, theta_user)

received_energy= received_intensity * aperture_area

print(received_energy)


PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458
photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength * 1e-9)
print(photon_energy_J)
photons_per_pulse = (received_energy) / photon_energy_J
print(photons_per_pulse)
solar_photons = LP.compute_solar_photons(z)

SNR = photons_per_pulse/solar_photons

print(SNR*100, "%")