"""
Radar_Performance.py

Monostatic radar performance: the range equation core plus an SNR Monte Carlo.

Signal path:
  Transmit power → monostatic radar equation (G^2, lambda^2, RCS, 1/R^4) → received power Pr.

Noise path:
  Radiant flux environment (solar, albedo, earth IR) → target equilibrium
  temperature via Stefan-Boltzmann → Johnson-Nyquist noise floor P_noise = k*T*B.

SNR = Pr / P_noise. A minimum detectable received power is also applied — if Pr
falls below Pr_radar_min, SNR is forced to zero regardless of the ratio.

The module is two layers in one file:
  1. Range equation (single source of truth). radar_received_power /
     radar_solve_transmit_power are the numeric forms, fast enough to call inside
     Monte Carlo loops; Multimodal Ranges uses them rather than re-typing the
     formula. compute_missing_radar is a slower sympy helper for solving for an
     arbitrary unknown (not for hot loops). Gain_Approx estimates gain from
     beamwidth, clamped to the aperture diffraction limit.
  2. Monte Carlo runner (monte_carlo_SNR) draws random orientations, ranges, and
     sun geometries to build a statistical picture of detection coverage.
"""

import numpy as np
import random
import sympy as sp
import radar_cross_section as rcs
import radiant_flux

## ── Default system parameters (overridden via function arguments) ──────
# Physics constants
k  = 1.38e-23  # Boltzmann constant (J/K)
sb = 5.6e-8    # Stefan-Boltzmann constant (W/m^2/K^4)

## Transmitter / antenna
Pt_radar                = 50       # peak transmit power (W)
radar_gain              = 10       # antenna gain — fixed, or compute via Gain_Approx(FoV, lambda_radar, radar_aperture_diameter)
lambda_radar            = 0.056    # radar wavelength (m)
FoV                     = 25       # sensor field of view (deg) — used only if gain is computed from the aperture
radar_aperture_diameter = 0.3      # antenna aperture diameter (m) — used only if gain is computed from the aperture

# Receiver
Pr_radar_min = 7.8e-18  # minimum detectable received power (W)
B            = 1e6      # radar receiver bandwidth (Hz)

# Target properties (default placeholder target; a caller/plugin overrides these)
DEFAULT_TARGET_CHARACTERISTIC_LENGTH = 5         # physical size of satellite (m) — scales RCS from the reference 6U cubesat to the target
DEFAULT_ABSORPTION = 1                           # surface solar absorptivity
DEFAULT_EMISSIVITY = 1                           # surface thermal emissivity
DEFAULT_AP         = 0.3 * 0.3                   # projected area for solar absorption (m^2)
DEFAULT_AR         = 0.3 * 0.3                   # radiating area (m^2)
DEFAULT_AVG_POWER  = 10                          # average spacecraft bus power dissipated as heat (W)

min_SNR = 0.3  # minimum SNR to count as a valid detection - arbitary value - useful for tuning against a datasheet


# ══ Range equation core (reusable, no simulation state) ════════════════
def Gain_Approx(Beamwidth, lamda_radar, aperture_diameter):
    """Estimate antenna gain from beamwidth, clamped to the diffraction-limited minimum.

    Gain is derived from the fraction of the full sphere covered by the beam:
        G = 4*pi / solid_angle_of_beam

    If the requested beamwidth is narrower than the aperture's diffraction limit
    (1.22 * lambda / D), the physical minimum beamwidth is used instead.
    """
    Beamwidth = np.radians(Beamwidth)
    if lamda_radar != 0 or aperture_diameter != 0:  # skip the clamp if either is left blank — sometimes we don't want to constrain to a given aperture
        theoretical_min_beamwidth = 1.22 * lamda_radar / aperture_diameter
        if theoretical_min_beamwidth > Beamwidth:
            # Can't achieve this beamwidth with this aperture — clamp to physical limit
            Beamwidth = theoretical_min_beamwidth

    # Solid angle of a cone with half-angle Beamwidth/2
    Gain = 4 * np.pi / (2 * np.pi * (1 - np.cos(Beamwidth / 2)))
    return Gain


def radar_received_power(Pt, G, wavelength, rcs, R, L=1.0):
    """Monostatic radar equation, forward form: received power Pr.

        Pr = Pt * G^2 * wavelength^2 * RCS / ((4*pi)^3 * R^4 * L)

    Plain numeric and numpy-friendly, so it is safe to call per sample inside a
    Monte Carlo loop. L defaults to 1 (lossless).
    """
    return Pt * G**2 * wavelength**2 * rcs / ((4 * np.pi)**3 * R**4 * L)


def radar_solve_transmit_power(Pr, G, wavelength, rcs, R, L=1.0):
    """Invert the radar equation for the transmit power Pt needed to achieve Pr."""
    return Pr * (4 * np.pi)**3 * R**4 * L / (G**2 * wavelength**2 * rcs)


# Symbolic form: defining the equation in sympy lets solve() find any one unknown
# given the rest. The Python bindings carry a sym_ prefix so they do not collide
# with the numeric simulation parameters (Pt_radar, lambda_radar) below; the symbol
# .name strings stay unchanged so the compute_missing_radar keyword API is unaffected.
sym_Pr, sym_Pt, sym_G, sym_lambda, sym_RCS, sym_R, sym_L = sp.symbols(
    'Pr_radar Pt_radar G_radar lambda_radar RCS_radar R_radar L_radar'
)
eq_radar = sym_Pr - (sym_Pt * sym_G**2 * sym_lambda**2 * sym_RCS) / (
    (4 * sp.pi)**3 * sym_R**4 * sym_L)


def compute_missing_radar(**kwargs):
    """Solve the radar equation for the one variable left as None.

    Pass keyword arguments matching the symbol names (Pr_radar, Pt_radar,
    G_radar, lambda_radar, RCS_radar, R_radar, L_radar). Exactly one must be
    None; the rest must be numeric. Returns the positive real solution as a float.
    """
    symbols = [sym_Pr, sym_Pt, sym_G, sym_lambda, sym_RCS, sym_R, sym_L]
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


# ══ Simulation ═════════════════════════════════════════════════════════
def get_pointing_from_azimuth(azimuth_deg):
    """Map a spacecraft azimuth angle to one of four orbital quadrants.

    Quadrants (centred on multiples of 90 deg) represent: anti-sun, nadir,
    sun-facing, and zenith. Used to look up the radiant flux environment.
    """
    az = azimuth_deg % 360
    if az >= 315 or az < 45:
        return "anti-sun"
    elif az >= 45 and az < 135:
        return "nadir"
    elif az >= 135 and az < 225:
        return "sun"
    elif az >= 225 and az < 315:
        return "zenith"


def compute_radar_returns(orientation_deg, R, pointing, case, beta):
    """Return (SNR, Pr, P_noise) for a single radar trial.

    Pr is forced through the detector floor: SNR is zeroed if Pr falls below
    Pr_radar_min, regardless of the noise level.

      Pr      : received power from the monostatic radar equation (W).
      P_noise : Johnson-Nyquist thermal noise floor k*T*B (W), where T is the
                target equilibrium temperature from the radiant flux environment
                via Stefan-Boltzmann: T = ((S*alpha*Ap + P_bus) / (eps*Ar*sigma))^0.25.
      SNR     : Pr / P_noise, or 0 below the detection floor.
    """
    # RCS scaled from the table's 6U cubesat reference area to the target area
    target_rcs = rcs.get_rcs_m2(orientation_deg) * DEFAULT_TARGET_CHARACTERISTIC_LENGTH ** 2 / (0.3 * 0.2)

    # Monostatic radar equation for received power
    Pr = radar_received_power(Pt_radar, radar_gain, lambda_radar, target_rcs, R)

    # Total incident radiant flux (W/m^2) for this orbital geometry
    S = radiant_flux.get_radiant_flux(case, 500, pointing, beta)

    # Equilibrium temperature from power balance: absorbed flux + bus power = radiated flux
    T       = ((S * DEFAULT_ABSORPTION * DEFAULT_AP + DEFAULT_AVG_POWER) / (DEFAULT_EMISSIVITY * DEFAULT_AR * sb)) ** 0.25
    P_noise = k * T * B

    SNR = Pr / P_noise if Pr >= Pr_radar_min else 0  # below minimum detectable signal → no detection
    return SNR, Pr, P_noise


def monte_carlo_SNR(max_range, samples):
    """Run `samples` Monte Carlo trials with randomised orientation, range, and sun geometry.

    Each trial draws:
      - Target orientation uniformly in [-180, 180] deg.
      - Range uniformly in [1, max_range].
      - Spacecraft azimuth uniformly in [-180, 180] deg, mapped to an orbital quadrant.
      - Thermal case (hot / cold) and solar beta angle uniformly from their sets.

    Physics is deferred to compute_radar_returns. Returns a list of dicts with
    per-sample geometry, SNR, and the underlying Pr / P_noise for downstream analysis.
    """
    results = []
    for i in range(samples):
        orientation = np.random.uniform(-180, 180)  # target orientation (deg)
        R           = np.random.uniform(1, max_range)

        azimuth  = np.random.uniform(-180, 180)
        pointing = get_pointing_from_azimuth(azimuth)

        case = random.choice(["hot", "cold"])   # worst-case hot or cold thermal environment
        beta = random.choice([0, 45, 70, 90])   # solar beta angle (deg)

        SNR, Pr, P_noise = compute_radar_returns(orientation, R, pointing, case, beta)

        results.append({
            "range_m":     R,
            "azimuth":     np.deg2rad(azimuth),
            "orientation": np.deg2rad(orientation),
            "SNR":         SNR,
            "Pr":          Pr,
            "P_noise":     P_noise,
        })

        percentage = 100 * i / samples
        if percentage % 1 == 0:
            print(f"Progress: {int(percentage)}%")

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    plt.close('all')

    def plotter():
        """Produce two polar plots: raw SNR scatter and P(detection) contours vs bearing and range."""
        import random
        print(f"plotter called with {len(results)} results")
        sample     = random.sample(results, min(50000, len(results)))
        rand_above = [r for r in sample if r["SNR"] > min_SNR]

        n_angle_bins     = 360
        range_bin_size_m = 1000

        max_r        = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))

        angle_edges   = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
        range_edges   = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)

        counts_total = np.zeros((n_angle_bins, n_range_bins))
        counts_above = np.zeros((n_angle_bins, n_range_bins))

        for r in results:
            a_idx = int(np.searchsorted(angle_edges[1:], r["azimuth"]))
            a_idx = np.clip(a_idx, 0, n_angle_bins - 1)
            r_idx = int(r["range_m"] // range_bin_size_m)
            r_idx = np.clip(r_idx, 0, n_range_bins - 1)
            counts_total[a_idx, r_idx] += 1
            if r["SNR"] > min_SNR:
                counts_above[a_idx, r_idx] += 1

        thresholds = [0.90, 0.50, 0.25, 0.1, 0.05, 0.01]
        colours = {0.90: "#1a9641", 0.50: "#fdae61", 0.25: "#d7191c",
                   0.10: "#b2182b", 0.05: "#762a83", 0.01: "#2d004b"}

        # For each angle bin and threshold, find the outermost range where detection probability >= threshold
        threshold_ranges = {t: np.full(n_angle_bins, np.nan) for t in thresholds}
        for a_idx in range(n_angle_bins):
            for t in thresholds:
                for r_idx in range(n_range_bins):
                    if counts_total[a_idx, r_idx] < 5:
                        continue
                    p = counts_above[a_idx, r_idx] / counts_total[a_idx, r_idx]
                    if p >= t:
                        threshold_ranges[t][a_idx] = range_edges[r_idx + 1]

        def smooth(arr):
            out   = np.full(len(arr), np.nan)
            valid = ~np.isnan(arr)
            if valid.sum() > 0:
                out[valid] = uniform_filter1d(arr[valid], size=15)
            return out

        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))

        # Sector boundary lines and labels for the four orbital quadrants
        sector_boundaries = np.deg2rad([45, 135, 225, 315])
        sector_labels = {
            np.deg2rad(0):   "anti-sun",
            np.deg2rad(90):  "nadir",
            np.deg2rad(180): "sun",
            np.deg2rad(270): "zenith",
        }
        for ax in (ax1, ax2):
            for angle in sector_boundaries:
                ax.plot([angle, angle], [0, max_r], color="black", linewidth=3, linestyle="--", alpha=0.5)
            for angle, label in sector_labels.items():
                ax.text(angle, max_r * 0.8, label, ha="center", va="center",
                        fontsize=15, color="black", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

        rand_zero  = [r for r in sample if r["SNR"] == 0]
        rand_below = [r for r in sample if 0 < r["SNR"] <= min_SNR]

        ax1.scatter([r["azimuth"] for r in rand_zero],  [r["range_m"] for r in rand_zero],
                    c="black", alpha=0.3, s=1, label="SNR = 0")
        ax1.scatter([r["azimuth"] for r in rand_below], [r["range_m"] for r in rand_below],
                    c="red",   alpha=0.3, s=1, label=f"0 < SNR <= {min_SNR}")
        ax1.scatter([r["azimuth"] for r in rand_above], [r["range_m"] for r in rand_above],
                    c="blue",  s=1,       label=f"SNR > {min_SNR}")
        ax1.set_title("Range vs Bearing (SNR)")
        ax1.legend(loc="upper right")
        ax1.set_rlim(0, max_r)
        ax2.set_rlim(0, max_r)

        for t in thresholds:
            smoothed = smooth(threshold_ranges[t])
            ax2.plot(angle_centres, smoothed, linewidth=2, c=colours[t], label=f"{int(t * 100)}% detection")
        ax2.set_title("Detection Range vs Bearing")
        ax2.legend()

    max_range = 10000
    results   = monte_carlo_SNR(max_range, 50000)

    plotter()
    plt.tight_layout()
    plt.show()
