"""
GaussianBeam.py

Models Gaussian beam propagation for a laser.

The beam has a waist w0 at focus and diverges with a half-angle determined by
the larger of the diffraction limit (lambda / pi*w0) and a user-supplied
divergence. This lets callers override the ideal diffraction limit to simulate
a non-ideal or truncated beam.
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian_beam_radius(z, w0, wavelength, theta_user=0.0):
    """Return the 1/e^2 beam radius at distance z along the propagation axis.

    Parameters:
        z          : axial distance from the waist (m)
        w0         : beam waist radius at focus (m)
        wavelength : laser wavelength (nm)
        theta_user : desired full-angle divergence (rad); 0 means diffraction-limited

    The effective half-angle is clamped to max(theta_diff, theta_user/2) so a
    larger user divergence always wins over the diffraction limit.
    """
    # Diffraction-limited half-angle divergence (lambda in nm converted to m)
    theta_diff = wavelength * (10**-9) / (np.pi * w0)
    # Use the larger of the two divergences so user can only make the beam wider
    theta_eff = max(theta_diff, theta_user)

    # Linear approximation: w(z) = w0 + z * tan(half-angle)
    w_z = w0 + z * np.tan(theta_eff / 2)

    return w_z

def gaussian_beam_wm2(x, y, z, w0, wavelength, P_total, theta_user=0.0):
    """Return the Gaussian beam irradiance (W/m^2) at transverse position (x, y) and axial distance z.

    Uses the same divergence model as gaussian_beam_radius. The Gaussian profile
    gives peak irradiance 2*P / (pi*w^2) on-axis, falling off as exp(-2r^2/w^2).

    Returns:
        I           : irradiance at (x, y, z) in W/m^2
        spot_radius : 1/e^2 beam radius at z (m), for reference
    """
    theta_diff = wavelength * (10**-9) / (np.pi * w0)
    theta_eff  = max(theta_diff, theta_user)
    w_z = w0 + z * np.tan(theta_eff / 2)

    r2 = x**2 + y**2
    # Standard Gaussian irradiance profile
    I = (2 * P_total / (np.pi * w_z**2)) * np.exp(-2 * r2 / w_z**2)

    spot_radius = gaussian_beam_radius(z, w0, wavelength, theta_user)

    return I, spot_radius
