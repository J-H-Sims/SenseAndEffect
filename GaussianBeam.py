import numpy as np
import matplotlib.pyplot as plt

def gaussian_beam_radius(z, w0, wavelength, theta_user=0.0):
    """
    Returns Gaussian beam radius at distance z, plus beam waist w0.

    Parameters:
        z : distance along propagation axis (m)
        w0 : beam waist (1/e^2 radius) at focus (m)
        wavelength : laser wavelength (m)
        theta_user : desired divergence angle (full-angle in rad)

    Returns:
        w_z : beam radius at distance z (1/e^2 of peak)
        w0  : beam waist radius (1/e^2 of peak)
    """
    # Diffraction-limited divergence
    theta_diff = wavelength *(10**-9) / (np.pi * w0)
    #print(f"theta diff: {theta_diff}")
    theta_eff = max(theta_diff, theta_user)

    # Beam radius at distance z
    w_z = w0 + z * np.tan(theta_eff / 2)

    return w_z

def gaussian_beam_wm2(x, y, z, w0, wavelength, P_total, theta_user=0.0):
    """
    Gaussian beam in W/m².
    """
    theta_diff = wavelength*(10**-9)  / (np.pi * w0)
    theta_eff = max(theta_diff, theta_user)
    w_z = w0 + z * np.tan(theta_eff / 2)
    r2 = x**2 + y**2
    I = (2 * P_total / (np.pi * w_z**2)) * np.exp(-2 * r2 / w_z**2)

    spot_radius = gaussian_beam_radius(z, w0, wavelength, theta_user)

    return I, spot_radius
#
# # -----------------------------
# # Example usage
# # -----------------------------
# w0 = 0.001  # 5 cm beam waist
# wavelength = 1.55e-6  # 1550 nm
# theta_user = np.deg2rad(1)  # 0.5 deg divergence
# P_total = 1.0
# z = 10000.0  # meters downrange
#
# x = np.linspace(-100, 100, 200)
# y = np.linspace(-100, 100, 200)
# X, Y = np.meshgrid(x, y)
# I, w = gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)
# print(w)
# print(z*np.tan(theta_user)/2)
# print(gaussian_beam_wm2(0,0, z, w0, wavelength, P_total, theta_user))
# plt.figure(figsize=(6, 5))
# plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')
# plt.colorbar(label=f'Beam intensity at z={z} m')
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title("Gaussian beam: diffraction + user divergence")
# plt.axis('equal')
# plt.show()