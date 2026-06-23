"""
Radar_Sim.py

Monte Carlo radar SNR simulation.

Each trial draws a random target orientation, range, and spacecraft azimuth.
RCS is interpolated from a look-up table (radar_cross_section.py). The noise
floor uses the Johnson-Nyquist formula: N = k * T * B, where T is estimated
from the radiant flux environment (solar, albedo, earth IR) via the
Stefan-Boltzmann law. SNR = Pr / N.

Detection contours showing P(detection) vs bearing and range are computed
by binning trials into (angle, range) cells and computing the fraction above
min_SNR in each cell.
"""

import numpy as np
import random
import radar_cross_section as rcs
import radiant_flux
import Radar_Performance as radar

# ── Simulation parameters ─────────────────────────────────────────────
FoV                  = 25       # sensor field of view (deg) — inactive while gain is hardcoded; only used by Gain_Approx
Pt_radar             = 50       # peak transmit power (W)
Pr_radar_min         = 7.8e-18  # minimum detectable received power (W)
lambda_radar         = 0.056    # radar wavelength (m)
radar_aperture_diameter = 0.3   # (m) — inactive while gain is hardcoded; only used by Gain_Approx
avg_power            = 10       # average spacecraft bus power dissipated as heat (W)

# Gain hardcoded to 10. To derive it from FoV/aperture/wavelength instead, replace with:
#   radar_gain = radar.Gain_Approx(FoV, lambda_radar, radar_aperture_diameter)
radar_gain = 10
print(radar_gain)

# ── Target parameters ─────────────────────────────────────────────────
absorption   = 1          # surface solar absorptivity
emmisivity   = 1          # surface thermal emissivity
Ap           = 0.3 * 0.3  # projected area for solar absorption (m^2)
Ar           = 0.3 * 0.3  # radiating area (m^2)

# ── Physics constants ─────────────────────────────────────────────────
B  = 1e6      # radar receiver bandwidth (Hz)
k  = 1.38e-23 # Boltzmann constant (J/K)
sb = 5.6e-8   # Stefan-Boltzmann constant (W/m^2/K^4)

min_SNR = 0.3  # detection threshold


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


def monte_carlo_SNR(max_range, samples):
    """Run `samples` Monte Carlo trials and return per-trial SNR results.

    For each trial:
      1. Draw random target orientation and range.
      2. Look up RCS at the drawn orientation (interpolated from table).
      3. Compute received power Pr using the monostatic radar equation.
      4. Estimate thermal noise temperature T from the radiant flux environment
         using Stefan-Boltzmann: T = ((S*alpha*Ap + P_bus) / (eps*Ar*sigma))^0.25.
      5. Compute SNR = Pr / (k * T * B). Set to 0 if Pr < Pr_radar_min.
    """
    results = []
    for i in range(samples):
        orientation = np.random.uniform(-180, 180)  # target orientation (deg)
        R           = np.random.uniform(1, max_range)

        # RCS scaled by a geometric factor (16*5) to convert from the table's reference area
        target_rcs = rcs.get_rcs_m2(orientation) * 16 * 5

        # Monostatic radar equation for received power (shared core in Radar_Performance)
        Pr = radar.radar_received_power(Pt_radar, radar_gain, lambda_radar, target_rcs, R)

        azimuth = np.random.uniform(-180, 180)
        pointing = get_pointing_from_azimuth(azimuth)

        case = random.choice(["hot", "cold"])   # worst-case hot or cold thermal environment
        beta = random.choice([0, 45, 70, 90])   # solar beta angle (deg)

        # Total incident radiant flux (W/m^2) for this orbital geometry
        S = radiant_flux.get_radiant_flux(case, 500, pointing, beta)

        # Equilibrium temperature from power balance: absorbed flux + bus power = radiated flux
        T = ((S * absorption * Ap + avg_power) / (emmisivity * Ar * sb)) ** 0.25

        if Pr < Pr_radar_min:
            SNR = 0  # below minimum detectable signal
        else:
            SNR = Pr / (k * T * B)

        results.append({
            "range_m":     R,
            "azimuth":     np.deg2rad(azimuth),
            "orientation": np.deg2rad(orientation),
            "SNR":         SNR
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

        angle_edges  = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
        range_edges  = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)

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
        colours = {
            0.90: "#1a9641", 0.50: "#fdae61",
            0.25: "#d7191c", 0.10: "#b2182b", 0.05: "#762a83", 0.01: "#2d004b"
        }
        # For each angle bin, find the outermost range where P(detection) >= threshold
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
            np.deg2rad(270): "zenith"
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
