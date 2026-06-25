"""
Sensor_Suite.py

Top-level multimodal sensor performance harness.

Wires together the shared target definition and both sensor models, configures
them for a scenario in one place, and exposes the two per sample physics
functions ready to fire:

    compute_lidar_returns(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y, ...)
    compute_radar_returns(orientation_deg, R, pointing, case, beta)

Run directly (`python Sensor_Suite.py`) to execute a combined Monte Carlo that
evaluates BOTH sensors on the same drawn scene and plots detection coverage:
    green  = both sensors detect
    red    = LiDAR only
    blue   = radar only
    grey   = neither (faint, shows the sampled space)

The Monte Carlo draws one shared scene per sample: a range, a random target
orientation, and a sun direction. The sun direction sets the LiDAR illumination
vector and the plotted bearing (its ZY-plane angle, matching the LiDAR plotter).
The radar's orbital thermal environment (pointing quadrant, beta, hot/cold) is
drawn independently per sample — it does not depend on the sun bearing, so tying
it to the plot axis would only band the coverage along the quadrant boundaries.
The target aspect (yaw) is shared by both sensors.
"""

import numpy as np
import random

import target_definition as tgt
import LiDAR_Performance as lidar
import Radar_Performance as radar

# ── Scenario configuration (edit here) ─────────────────────────────────
# Shared target — both sensors observe this single object.
tgt.configure(
    DEFAULT_LENGTH=1.0, DEFAULT_WIDTH=1.0, DEFAULT_HEIGHT=1.0,
    DEFAULT_FACE_MATERIALS=["Lambertian 20%"] * 6,
)

# Sensors — apply the named presets, then override individual parameters as needed.
lidar.configure_preset("Sapphire 2")
radar.configure_preset("TTP Phase B")

# ── Performance computations, loaded and ready to fire ─────────────────
compute_lidar_returns = lidar.compute_lidar_returns
compute_radar_returns = radar.compute_radar_returns

RADAR_BETAS     = [0, 45, 70, 90]                          # tabulated solar beta angles (deg)
RADAR_POINTINGS = ["anti-sun", "nadir", "sun", "zenith"]   # orbital quadrants for the thermal lookup


def monte_carlo(max_range=10000, n_samples=30000):
    """Evaluate both sensors over n_samples shared scenes.

    Each sample draws:
      - range uniformly in [0, max_range],
      - target orientation roll/pitch/yaw uniformly over the sphere (radians),
      - sun direction uniformly over the sphere (azimuth + arcsin elevation).

    The plotted bearing is the sun's ZY-plane angle (angle_zy), matching the LiDAR
    plotter. The LiDAR uses the full illumination vector; the radar's orbital
    thermal environment (pointing, beta, hot/cold) is drawn independently since it
    is unrelated to the sun bearing, while the target aspect (yaw) is shared.
    Returns a list of dicts with per-sample geometry, both SNRs, and detection flags.
    """
    results = []
    for n in range(n_samples):
        range_m = np.random.uniform(0, max_range)

        roll  = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)
        yaw   = np.random.uniform(-np.pi, np.pi)

        sun_az = np.random.uniform(-np.pi, np.pi)
        sun_el = np.arcsin(np.random.uniform(-1, 1))
        illum_dir = np.array([
            np.cos(sun_el) * np.cos(sun_az),
            np.cos(sun_el) * np.sin(sun_az),
            np.sin(sun_el),
        ])
        # Plotted bearing: the sun's ZY-plane angle, matching the LiDAR plotter axis
        _, bearing, _ = lidar.relative_angles([0, 0, 1], illum_dir)

        # LiDAR beam pointing error scales with range and divergence
        apc = (range_m / 2) * np.tan(lidar.half_beam_divergence_rad / 2)
        X, Y = np.random.uniform(-apc, apc), np.random.uniform(-apc, apc)

        # ── LiDAR ──
        snr_l, *_ = compute_lidar_returns(
            range_m, lidar.pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y)
        lidar_detect = snr_l > lidar.min_SNR

        # ── Radar ──
        # Orbital thermal environment is independent of the sun bearing → draw it
        # per sample (otherwise coverage bands along the quadrant axes). Aspect is shared.
        pointing = random.choice(RADAR_POINTINGS)
        beta     = random.choice(RADAR_BETAS)
        case     = random.choice(["hot", "cold"])
        snr_r, *_ = compute_radar_returns(np.degrees(yaw), range_m, pointing, case, beta)
        radar_detect = snr_r > radar.min_SNR

        results.append({
            "range_m":      range_m,
            "bearing":      bearing,       # sun ZY-plane angle (rad), the polar bearing
            "snr_lidar":    snr_l,
            "snr_radar":    snr_r,
            "lidar_detect": lidar_detect,
            "radar_detect": radar_detect,
        })

        percentage = 100 * (n + 1) / n_samples
        if percentage % 1 == 0:
            print(f"Progress: {int(percentage)}%")

    return results


def plotter(results):
    """Two polar panels (LiDAR plotter convention):

    Left : range vs bearing scatter, coloured by which sensor(s) detect
           (green both, red LiDAR only, blue radar only, faint grey neither).
    Right: smoothed detection-range contours vs bearing at several probability
           thresholds, LiDAR in a red family and radar in a blue family.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    sample = random.sample(results, min(50000, len(results)))
    max_r  = max(r["range_m"] for r in results)

    n_angle_bins     = 360
    range_bin_size_m = 1000
    n_range_bins = int(np.ceil(max_r / range_bin_size_m))

    angle_edges   = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
    range_edges   = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)

    counts_total = np.zeros((n_angle_bins, n_range_bins))
    counts_lidar = np.zeros((n_angle_bins, n_range_bins))
    counts_radar = np.zeros((n_angle_bins, n_range_bins))
    for r in results:
        a_idx = int(np.searchsorted(angle_edges[1:], r["bearing"]))
        a_idx = np.clip(a_idx, 0, n_angle_bins - 1)
        r_idx = int(np.clip(r["range_m"] // range_bin_size_m, 0, n_range_bins - 1))
        counts_total[a_idx, r_idx] += 1
        if r["lidar_detect"]:
            counts_lidar[a_idx, r_idx] += 1
        if r["radar_detect"]:
            counts_radar[a_idx, r_idx] += 1

    thresholds = [0.90, 0.50, 0.10]

    def coverage(counts, t):
        # Outermost range per angle bin where detection probability >= t
        out = np.full(n_angle_bins, np.nan)
        for a_idx in range(n_angle_bins):
            for r_idx in range(n_range_bins):
                if counts_total[a_idx, r_idx] < 5:
                    continue
                if counts[a_idx, r_idx] / counts_total[a_idx, r_idx] >= t:
                    out[a_idx] = range_edges[r_idx + 1]
        return out

    def smooth(arr):
        out   = np.full(len(arr), np.nan)
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            out[valid] = uniform_filter1d(arr[valid], size=15)
        return out

    both       = [r for r in sample if r["lidar_detect"] and r["radar_detect"]]
    lidar_only = [r for r in sample if r["lidar_detect"] and not r["radar_detect"]]
    radar_only = [r for r in sample if r["radar_detect"] and not r["lidar_detect"]]
    neither    = [r for r in sample if not r["lidar_detect"] and not r["radar_detect"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))

    # Left: detection-category scatter (neither first as background, both on top)
    ax1.scatter([r["bearing"] for r in neither],    [r["range_m"] for r in neither],
                c="lightgrey", s=1, alpha=0.2, label="neither")
    ax1.scatter([r["bearing"] for r in radar_only], [r["range_m"] for r in radar_only],
                c="blue",  s=2, alpha=0.5, label="radar only")
    ax1.scatter([r["bearing"] for r in lidar_only], [r["range_m"] for r in lidar_only],
                c="red",   s=2, alpha=0.5, label="LiDAR only")
    ax1.scatter([r["bearing"] for r in both],       [r["range_m"] for r in both],
                c="green", s=2, alpha=0.6, label="both")
    ax1.set_title("Detection by sensor vs bearing and range")
    ax1.legend(loc="upper right", markerscale=4)
    ax1.set_rlim(0, max_r)
    ax2.set_rlim(0, max_r)

    # Right: per-sensor detection-range contours (LiDAR red family, radar blue family)
    lidar_shades = {0.90: "#7f0000", 0.50: "#d7191c", 0.10: "#fdae61"}
    radar_shades = {0.90: "#08306b", 0.50: "#2171b5", 0.10: "#9ecae1"}
    for t in thresholds:
        ax2.plot(angle_centres, smooth(coverage(counts_lidar, t)),
                 c=lidar_shades[t], linewidth=2, label=f"LiDAR {int(t * 100)}%")
        ax2.plot(angle_centres, smooth(coverage(counts_radar, t)),
                 c=radar_shades[t], linewidth=2, label=f"Radar {int(t * 100)}%")
    ax2.set_title("Detection range vs bearing")
    ax2.legend(loc="upper right", fontsize=8)
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")

    results = monte_carlo(max_range=30000, n_samples=100000)

    n_l = sum(r["lidar_detect"] for r in results)
    n_r = sum(r["radar_detect"] for r in results)
    n_b = sum(r["lidar_detect"] and r["radar_detect"] for r in results)
    print(f"LiDAR detections: {n_l}  Radar detections: {n_r}  Both: {n_b}  of {len(results)}")

    plotter(results)
    plt.tight_layout()
    plt.show()
