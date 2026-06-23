"""
Radar_Performance.py

Monostatic radar range equation: numeric core, symbolic solver, and gain.

The standard monostatic radar equation is:
    Pr = Pt * G^2 * lambda^2 * RCS / ((4*pi)^3 * R^4 * L)

radar_received_power / radar_solve_transmit_power are the single numeric source
of truth for this equation and are fast enough to call inside Monte Carlo
loops. Radar_Sim and Multimodal Ranges both use them rather than re-typing the
formula. compute_missing_radar is a slower sympy-based helper for solving for an
arbitrary unknown; it is not meant for hot loops. Gain_Approx provides a quick
gain estimate from beamwidth, clamped to the physical minimum imposed by the
aperture diameter.
"""

import numpy as np
import sympy as sp

def Gain_Approx(Beamwidth, lamda_radar, aperture_diameter):
    """Estimate antenna gain from beamwidth, clamped to the diffraction-limited minimum.

    Gain is derived from the fraction of the full sphere covered by the beam:
        G = 4*pi / solid_angle_of_beam

    If the requested beamwidth is narrower than the aperture's diffraction limit
    (1.22 * lambda / D), the physical minimum beamwidth is used instead.
    """
    Beamwidth = np.radians(Beamwidth)
    if lamda_radar != 0 or aperture_diameter != 0: #this check fails if either vriable has been left blank - sometimes we may not want to constrain the simulation to a given aperture diameter
        theoretical_min_beamwidth = 1.22 * lamda_radar / aperture_diameter
        if theoretical_min_beamwidth > Beamwidth:
            # Can't achieve this beamwidth with this aperture — clamp to physical limit
            Beamwidth = theoretical_min_beamwidth

    # Solid angle of a cone with half-angle Beamwidth/2
    Gain = 4 * np.pi / (2 * np.pi * (1 - np.cos(Beamwidth / 2)))
    return Gain


# ── Numeric radar equation (single source of truth) ───────────────────
def radar_received_power(Pt, G, wavelength, rcs, R, L=1.0):
    """Monostatic radar equation, forward form: received power Pr.

        Pr = Pt * G^2 * wavelength^2 * RCS / ((4*pi)^3 * R^4 * L)

    Plain numeric and numpy-friendly, so it is safe to call per sample inside a
    Monte Carlo loop. L defaults to 1 (lossless), matching the inline forms it
    replaces.
    """
    return Pt * G**2 * wavelength**2 * rcs / ((4 * np.pi)**3 * R**4 * L)


def radar_solve_transmit_power(Pr, G, wavelength, rcs, R, L=1.0):
    """Invert the radar equation for the transmit power needed to achieve Pr."""
    return Pr * (4 * np.pi)**3 * R**4 * L / (G**2 * wavelength**2 * rcs)


# ── Symbolic radar equation ───────────────────────────────────────────
# Defining as sympy symbols allows solve() to find any one unknown given the rest
Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar = sp.symbols(
    'Pr_radar Pt_radar G_radar lambda_radar RCS_radar R_radar L_radar'
)

# Monostatic radar range equation (linear power form, not dB)
eq_radar = Pr_radar - (Pt_radar * G_radar**2 * lambda_radar**2 * RCS_radar) / (
    (4 * sp.pi)**3 * R_radar**4 * L_radar)


def compute_missing_radar(**kwargs):
    """Solve the radar equation for the one variable left as None.

    Pass keyword arguments matching the symbol names above. Exactly one must be
    None; the rest must be numeric. Returns the positive real solution as a float.
    """
    symbols = [Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar]
    missing = [v for v in symbols if kwargs.get(v.name) is None]
    if len(missing) != 1:
        raise ValueError("Exactly one variable must be None for radar")

    subs = {v: kwargs[v.name] for v in symbols if kwargs.get(v.name) is not None}
    sol  = sp.solve(eq_radar.subs(subs), missing[0])

    # The radar equation always has a unique positive real solution for physical inputs
    sol_real_positive = [s.evalf() for s in sol if s.is_real and s > 0]
    if not sol_real_positive:
        raise ValueError("No positive real solution found. Check inputs.")
    return float(sol_real_positive[0])
