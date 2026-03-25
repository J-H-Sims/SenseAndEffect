import numpy as np
from sympy import elliptic_f
import random

from sympy.physics.units import boltzmann

import radar_cross_section as rcs
import Radar_Performance as radar
import radiant_flux
# Simulation parameters
FoV = 25  # degrees
Pt_radar = 700  # W
Pr_radar_min = 7.8e-18  # W
lambda_radar = 0.056  # m
radar_aperture_diameter = 0.3  # m
radar_gain = 100
#radar_gain = radar.Gain_Approx(FoV, lambda_radar, radar_aperture_diameter)
print(radar_gain)
absorption = 1
emmisivity = 1
Ap = 0.3*0.3
Ar = 0.3*0.3


B = 1e6  # radar bandwidth [Hz]
k = 1.38e-23  # Boltzmann constant
sb = 5.6e-8
avg_power = 10

min_SNR = 0.3
def get_pointing_from_azimuth(azimuth_deg):
    az = azimuth_deg+180
    az = azimuth_deg % 360  # normalize

    if (az >= 315 or az < 45):
        return "anti-sun"
    elif (az >= 45 and az < 135):
        return "nadir"
    elif (az >= 135 and az < 225):
        return "sun"
    elif (az >= 225 and az < 315):
        return "zenith"

def monte_carlo_SNR(max_range, samples):
    results = []
    for i in range(samples):
        orientation = np.random.uniform(-180, 180)  # degrees
        R = np.random.uniform(1, max_range)  # avoid R=0

        target_rcs =  rcs.get_rcs_m2(orientation)*16

        Pr = Pt_radar * radar_gain ** 2 * lambda_radar ** 2 * target_rcs / ((4 * np.pi) ** 3 * R ** 4)

        azimuth = np.random.uniform(-180, 180)

        pointing = get_pointing_from_azimuth(azimuth)


        case = random.choice(["hot", "cold"])
        beta =  random.choice([0, 45, 70, 90])

        S = radiant_flux.get_radiant_flux(case, 500, pointing, beta)

        T = ((S*absorption*Ap+avg_power)/(emmisivity*Ar*sb))**0.25
        #print(S, pointing, T)

        #Trx = 0
        #Tant = np.random.uniform(0, 84)
        #Tsys =Trx + Tant

        if Pr < Pr_radar_min:
            SNR = 0
        else:
            SNR = Pr / (k * T * B)
            #print(k*T*B)

        results.append({
            "range_m": R,
            "azimuth": np.deg2rad(azimuth),
            "orientation": np.deg2rad(orientation),  # store in radians for polar plot
            "SNR": SNR
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
        import random
        print(f"plotter called with {len(results)} results")
        sample = random.sample(results, min(50000, len(results)))
        rand_above = [r for r in sample if r["SNR"] > min_SNR]
        rand_below = [r for r in sample if r["SNR"] <= min_SNR]

        n_angle_bins = 360
        range_bin_size_m = 1000

        max_r = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))

        angle_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
        range_edges = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)

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

        thresholds = [0.90, 0.75, 0.50, 0.25, 0.1, 0.05, 0.01]
        colours = {
            0.90: "#1a9641", 0.75: "#a6d96a", 0.50: "#fdae61",
            0.25: "#d7191c", 0.10: "#b2182b", 0.05: "#762a83", 0.01: "#2d004b"
        }
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
            out = np.full(len(arr), np.nan)
            valid = ~np.isnan(arr)
            if valid.sum() > 0:
                out[valid] = uniform_filter1d(arr[valid], size=15)
            return out

        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))

        # Sector boundaries and labels
        sector_boundaries = np.deg2rad([45, 135, 225, 315])
        sector_labels = {
            np.deg2rad(0): "anti-sun",
            np.deg2rad(90): "nadir",
            np.deg2rad(180): "sun",
            np.deg2rad(270): "zenith"
        }

        for ax in (ax1, ax2):
            for angle in sector_boundaries:
                ax.plot([angle, angle], [0, max_r], color="black", linewidth=3, linestyle="--", alpha=0.5)
            for angle, label in sector_labels.items():
                ax.text(angle, max_r * 0.8, label, ha="center", va="center", fontsize=15, color="black",bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

        rand_zero = [r for r in sample if r["SNR"] == 0]
        rand_below = [r for r in sample if 0 < r["SNR"] <= min_SNR]

        ax1.scatter([r["azimuth"] for r in rand_zero], [r["range_m"] for r in rand_zero],
                    c="black", alpha=0.3, s=1, label="SNR = 0")
        ax1.scatter([r["azimuth"] for r in rand_below], [r["range_m"] for r in rand_below],
                    c="red", alpha=0.3, s=1, label=f"0 < SNR <= {min_SNR}")
        ax1.scatter([r["azimuth"] for r in rand_above], [r["range_m"] for r in rand_above],
                    c="blue", s=1, label=f"SNR > {min_SNR}")
        ax1.set_title("Range vs Bearing (SNR)")
        ax1.legend(loc="upper right")
        ax1.set_rlim(0, max_r)
        ax2.set_rlim(0, max_r)

        for t in thresholds:
            smoothed = smooth(threshold_ranges[t])
            ax2.plot(angle_centres, smoothed, linewidth=2, c=colours[t], label=f"{int(t * 100)}% detection")

        ax2.set_title("Detection Range vs Bearing")
        ax2.legend()

    max_range = 45000
    results = monte_carlo_SNR(max_range, 5000000)

    plotter()

    plt.tight_layout()
    plt.show()