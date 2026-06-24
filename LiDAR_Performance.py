"""
LiDAR_Performance.py

Full LiDAR performance simulation combining Gaussian beam propagation, cuboid
target reflectance, and solar background estimation.

Signal path:
  Gaussian beam → cuboid BRDF return → solid-angle collection → photon count.

Background path:
  Solar irradiance × cuboid solar BRDF → solid-angle collection → photon count,
  plus a direct-glare term when the sun falls within or near the sensor FoV.

SNR = signal_photons / background_photons. A minimum photon-count threshold is
also applied — if total detected photons fall below min_photons_to_detect, SNR
is forced to zero regardless of the ratio.

The Monte Carlo runner (monte_carlo_SNR) draws random orientations, ranges, and
sun directions to build a statistical picture of detection coverage.
"""

import numpy as np
import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import CuboidSolarModel as solar

solar_tracker = []  # accumulates per-sample background photon counts for later plotting
PLANCK_CONSTANT  = 6.62607015e-34
SPEED_OF_LIGHT   = 299792458
## ── Default system parameters (overridden via function arguments) ──────
## Laser beam properties
# NOTE: Deriving the background exposure from range resolution is an approximation.
# A real detector integrates solar background over its full electrical gate / readout
# window, which is set by the range bin the system resolves (range_resolution / c),
# not by the laser pulse width. This is deliberately more conservative than a pulse
# width derived exposure: the range gate is wider than the pulse, so it admits more
# background light and yields a lower, more pessimistic SNR. It does not capture
# detector bandwidth, dead time, or multi bin integration effects.
range_resolution = 100          # desired range resolution (m) - sets the detector time gate; finer resolution lets in less background light
exposure_time    = range_resolution / SPEED_OF_LIGHT  # detector gate window (s) - the time the detector integrates background light for each range bin
wavelength_nm    = 1550           # laser wavelength (nm)
pulse_energy_J = 0.0009 #J - reality checks. Spiral Blue Sapphire 2 has option for 400uJ or 900 uJ. Laser demo used 650 nJ. https://brightsolutions.it/products/ - this is a good supplier of very high power lasers - see microchip and Aero
half_beam_divergence_rad  = 0.0003   # laser half-angle divergence (rad) this is the desired beam divergence and is used by the gaussian beam model to compute the beam radius at a given range. Set to zero for diffraction limited. Note that Gaussian beam will overwrite this is a beam divergence below the diffraction limit is specified
beam_waist   = 0.001    # beam waist at focus (m)- a wider beam will have a larger beam waist and will diverge less quickly. This is used by the gaussian beam model to compute the beam radius at a given range. Set to zero for diffraction limited. Note that Gaussian beam will overwrite this is a beam waist below the diffraction limit is specified

#Reciever Properties
aperture_radius  = 0.015
aperture_area_m2 = np.pi * (aperture_radius ** 2)


bandwidth        = 100             # optical bandpass filter width (nm)
polarity_filter  = 0.2            # fraction of solar light passed by polarisation filter
SOLAR_SPECTRAL_IRRADIANCE_W_M2_NM = 0.4  # solar spectral irradiance at 1550 nm (W/m²/nm)
min_photons_to_detect = 20        # detector threshold — fewer photons means no detection
min_SNR               = 0.3      # minimum SNR to count as a valid detection - arbitary value - useful for tuning against a datasheet

#Target properties
DEFAULT_LENGTH = 0.1 #m
DEFAULT_WIDTH  = 0.1 #m
DEFAULT_HEIGHT = 0.1 #m
DEFAULT_ROLL   = np.radians(45)  #rad
DEFAULT_PITCH  = np.radians(45)  #rad
DEFAULT_YAW    = np.radians(45)  #rad
DEFAULT_FACE_MATERIALS = ["Lambertian 20%"] * 6 # drawn from the material parameter JSON. the 20% lambertian is from the SBIR requirement

#Scene Properties
DEFAULT_RANGE_M = 10000 #m


def compute_photons_p_pulse(X=0, Y=0, z=DEFAULT_RANGE_M, w0=0.001, wavelength=wavelength_nm, P_total=0.001, theta_user=0,
                             length=DEFAULT_LENGTH, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                             roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=DEFAULT_YAW, face_materials=DEFAULT_FACE_MATERIALS):
    """Return signal photons collected per pulse.

    Uses the Gaussian beam irradiance at (X, Y, z) as the incident flux, then
    multiplies by the cuboid BRDF-weighted return factor R. Energy reaching the
    aperture is estimated via solid-angle geometry (aperture area / range^2)
    rather than propagating a return Gaussian, which avoids needing a return
    beam waist estimate.
    """
    I, _ = Gbeam.gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)

    # R is the normalised BRDF-weighted return (J/J); the projected area term is unused here
    R, _ = target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)

    # Solid angle subtended by the aperture at range z
    solid_angle = aperture_area_m2 / z ** 2
    received_energy = R * solid_angle * I

    photon_energy_J    = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength * 1e-9)
    photons_per_pulse  = received_energy / photon_energy_J

    return photons_per_pulse


def compute_solar_photons(range_m, length=DEFAULT_LENGTH, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                           roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=DEFAULT_YAW, face_materials=DEFAULT_FACE_MATERIALS,
                           illum_dir=[0, 0, 1], obs_dir=[0, 0, 1], wavelength_nm=wavelength_nm):
    """Return (total_solar_photons, direct_glare_photons) per pulse.

    Background has two components:
      1. Diffuse: solar light reflected off the target, collected through the aperture.
      2. Direct glare: solar flux landing directly on the aperture when the sun is
         within or near the sensor boresight (modelled as a cone of radius solar_disk_radius).

    The glare model uses relative_angles to compute how far the sun sits from the
    view direction in each 2D projection plane, then adds:
      - Full glare if the sun is within the sensor FoV (angular offset > FoV boundary).
      - Partial glare with linear fall-off between the FoV edge and the solar disk boundary.
    """
    R, _ = solar.solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir)

    solid_angle = aperture_area_m2 / range_m**2

    # Diffuse solar background: solar spectral irradiance × bandwidth × polarisation filter × BRDF × solid angle
    solar_collected_W = SOLAR_SPECTRAL_IRRADIANCE_W_M2_NM * bandwidth * polarity_filter * R * solid_angle

    photon_energy_J  = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)
    solar_photons    = (solar_collected_W / photon_energy_J) * exposure_time

    angle_zx, angle_zy, _ = relative_angles(obs_dir, illum_dir)

    # FoV angle outside which the sun causes direct glare; clamped to 5 deg minimum
    FoV = np.pi - max(half_beam_divergence_rad / 2, np.radians(5))
    # Angular radius of the solar disk (0.265 deg half-angle)
    solar_disk_radius = np.pi - np.radians(0.265)


    # Calculates the direct solar photons entering the aperture when the sun is within or near the FoV. This is an approximation and is basically a user defined FoV - 
    direct_solar_photons = 0
    if abs(angle_zx) > solar_disk_radius or abs(angle_zy) > solar_disk_radius:
        # Sun is on the boresight — full direct irradiance enters the aperture
        direct_solar_photons = (SOLAR_SPECTRAL_IRRADIANCE_W_M2_NM * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * exposure_time
        solar_photons += direct_solar_photons
    elif abs(angle_zx) > FoV or abs(angle_zy) > FoV:
        # Sun is just outside the FoV — linear fall-off from disk edge to FoV boundary
        x = (np.radians(0.265)) / (np.pi - abs(angle_zx))
        direct_solar_photons = x * (SOLAR_SPECTRAL_IRRADIANCE_W_M2_NM * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * exposure_time
        solar_photons += direct_solar_photons

    return solar_photons, direct_solar_photons


def compute_lidar_returns(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y, theta=half_beam_divergence_rad):
    """Return (SNR, photons_per_pulse, reflected_solar_photons_per_pulse, direct_solar_photons).

    SNR is forced to zero if total detected photons fall below min_photons_to_detect.
    The photon threshold guards against detections with too few photons to be
    statistically reliable regardless of the signal-to-background ratio.
    """
    reflected_solar_photons_per_pulse, direct_solar_photons = compute_solar_photons(
        range_m, DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_HEIGHT, roll, pitch, yaw, DEFAULT_FACE_MATERIALS, illum_dir, [0, 0, 1], wavelength_nm)
    photons_per_pulse = compute_photons_p_pulse(
        X, Y, range_m, beam_waist, wavelength_nm, pulse_energy_J, theta,
        DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_HEIGHT, roll, pitch, yaw, DEFAULT_FACE_MATERIALS)

    SNR = photons_per_pulse / reflected_solar_photons_per_pulse if reflected_solar_photons_per_pulse > 0 else np.inf

    # Zero SNR if total photons (signal + diffuse background) is below detector threshold
    if photons_per_pulse + (reflected_solar_photons_per_pulse - direct_solar_photons) < min_photons_to_detect:
        SNR = 0

    solar_tracker.append({"solar photons": reflected_solar_photons_per_pulse - direct_solar_photons})
    return SNR, photons_per_pulse, reflected_solar_photons_per_pulse, direct_solar_photons


def compute_range(pulse_energy_J, theta=half_beam_divergence_rad):
    """Step-search for maximum detection range by incrementing range until SNR < min_SNR."""
    range_m = 1000
    SNR, *_ = compute_lidar_returns(range_m, pulse_energy_J, DEFAULT_ROLL, DEFAULT_PITCH, DEFAULT_YAW, [0, 0, 1], 0, 0, theta)
    dr      = 100

    if SNR <= min_SNR:
        return 0

    while SNR > min_SNR:
        range_m += dr
        SNR, *_ = compute_lidar_returns(range_m, pulse_energy_J, DEFAULT_ROLL, DEFAULT_PITCH, DEFAULT_YAW, [0, 0, 1], 0, 0, theta)

    return range_m


def compute_pulse_energy(range, theta=half_beam_divergence_rad):
    """Descend from a very large pulse energy until SNR falls to min_SNR.

    The descent uses a fractional step (0.5% per iteration) so it converges
    from above without overshooting. This works because signal photons are
    linear in pulse energy.
    """
    pulse_energy_J = 10000000000000  # start absurdly high to guarantee SNR > min_SNR
    illum_dir = np.array([0, 0, 1])

    SNR, *_ = compute_lidar_returns(range, pulse_energy_J, DEFAULT_ROLL, DEFAULT_PITCH, DEFAULT_YAW, illum_dir, 0, 0, theta)

    while SNR > min_SNR:
        dE             = pulse_energy_J * 0.005
        pulse_energy_J -= dE
        SNR, *_        = compute_lidar_returns(range, pulse_energy_J, DEFAULT_ROLL, DEFAULT_PITCH, DEFAULT_YAW, illum_dir, 0, 0, theta)

    return pulse_energy_J


def relative_angles(view_dir, illum_dir):
    """Return the sun-to-boresight angular offset projected onto the ZX, ZY, and XY planes.

    Projects both vectors onto each 2D plane using atan2 and takes the difference.
    Used to normalise solar geometry to the sensor boresight for polar-plot visualisation.
    """
    angle_zx = np.arctan2(illum_dir[0], illum_dir[2]) - np.arctan2(view_dir[0], view_dir[2])
    angle_zy = np.arctan2(illum_dir[1], illum_dir[2]) - np.arctan2(view_dir[1], view_dir[2])
    angle_xy = np.arctan2(illum_dir[1], illum_dir[0]) - np.arctan2(view_dir[1], view_dir[0])
    return angle_zx, angle_zy, angle_xy




def monte_carlo_SNR(max_range = DEFAULT_RANGE_M, pulse_energy_J = pulse_energy_J, n_samples=1000):
    """Run n_samples Monte Carlo trials with randomised orientation, range, and sun direction.

    Each trial draws:
      - Roll, pitch, yaw uniformly over the full sphere.
      - Range uniformly in [0, max_range].
      - Boresight pointing error: uniform over a square with side = avg_pointing_accuracy_cartesian.
      - Sun direction sampled uniformly over the sphere (uniform elevation via arcsin).

    Returns a list of dicts with per-sample geometry and SNR for downstream analysis.
    """
    solar_tracker.clear()
    results = []
    n = 0

    for _ in range(n_samples):
        n += 1
        roll  = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)
        yaw   = np.random.uniform(-np.pi, np.pi)

        range_m = np.random.uniform(0, max_range)
        # Cartesian pointing error scales with range and half-angle
        avg_pointing_accuracy_cartesian = (range_m / 2) * np.tan(half_beam_divergence_rad / 2)
        X, Y = (np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian),
                np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian))

        azimuth   = np.random.uniform(-np.pi, np.pi)
        # arcsin of uniform[-1,1] gives uniform distribution over elevation
        elevation = np.arcsin(np.random.uniform(-1, 1))
        illum_dir = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])

        SNR, photons_per_pulse, reflected_solar_photons_per_pulse, direct_solar_photons = compute_lidar_returns(
            range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y)

        view_dir = [0, 0, 1]
        angle_zx, angle_zy, _ = relative_angles(view_dir, illum_dir)

        results.append({
            "range_m":                       range_m,
            "roll":                          roll,
            "pitch":                         pitch,
            "yaw":                           yaw,
            "angle_zx":                      angle_zx,
            "angle_zy":                      angle_zy,
            "pulse_energy":                  pulse_energy_J,
            "SNR":                           SNR,
            "photons_per_pulse":             photons_per_pulse,
            "reflected_solar_photons":       reflected_solar_photons_per_pulse,
            "direct_solar_photons":          direct_solar_photons,
        })

        percentage = 100 * n / n_samples
        if percentage % 1 == 0:
            print(f"Progress: {int(percentage)}%")

    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close('all')

    from scipy.ndimage import uniform_filter1d

    def plot_solar_tracker():
        """Plot solar background photon distribution vs range as percentile bands."""
        if len(solar_tracker) != len(results):
            print(f"solar_tracker/results length mismatch: {len(solar_tracker)} vs {len(results)}")
            return

        range_bin_size_m = 1000
        max_r        = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))
        range_edges  = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)
        range_centres = (range_edges[:-1] + range_edges[1:]) / 2

        binned = [[] for _ in range(n_range_bins)]
        for entry, r in zip(solar_tracker, results):
            r_idx = int(np.clip(r["range_m"] // range_bin_size_m, 0, n_range_bins - 1))
            val   = entry["solar photons"]
            if val > 0:
                binned[r_idx].append(val)

        percentiles = {p: np.full(n_range_bins, np.nan) for p in [5, 25, 50, 75, 95]}
        for i, vals in enumerate(binned):
            if len(vals) >= 5:
                for p in percentiles:
                    percentiles[p][i] = np.percentile(vals, p)

        valid = ~np.isnan(percentiles[50])
        rc    = range_centres[valid]
        plt.rcParams.update({'font.size': 15})

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(rc, percentiles[5][valid],  percentiles[95][valid], alpha=0.15, color='steelblue', label='5-95%')
        ax.fill_between(rc, percentiles[25][valid], percentiles[75][valid], alpha=0.35, color='steelblue', label='25-75%')
        ax.plot(rc, percentiles[50][valid], color='steelblue', linewidth=2, label='Median')

        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Solar photons per pulse')
        ax.set_title('Solar background photons vs range')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.axhline(photon_detect_lim, color='red', linestyle='--', linewidth=1.5, label='Detection limit')
        ax.legend()
        plt.tight_layout()

    def plotter():
        """Produce two polar plots: raw SNR scatter and P(detection) contours vs bearing and range."""
        import random
        print(f"plotter called with {len(results)} results")
        sample      = random.sample(results, min(50000, len(results)))
        rand_above  = [r for r in sample if r["SNR"] > min_SNR]

        n_angle_bins     = 360
        range_bin_size_m = 1000

        max_r        = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))

        angle_edges   = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
        range_edges   = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)
        range_centres = (range_edges[:-1] + range_edges[1:]) / 2

        counts_total = np.zeros((n_angle_bins, n_range_bins))
        counts_above = np.zeros((n_angle_bins, n_range_bins))

        for r in results:
            a_idx = int(np.searchsorted(angle_edges[1:], r["angle_zy"]))
            a_idx = np.clip(a_idx, 0, n_angle_bins - 1)
            r_idx = int(r["range_m"] // range_bin_size_m)
            r_idx = np.clip(r_idx, 0, n_range_bins - 1)
            counts_total[a_idx, r_idx] += 1
            if r["SNR"] > min_SNR:
                counts_above[a_idx, r_idx] += 1

        thresholds = [0.90, 0.75, 0.50, 0.25, 0.1, 0.05, 0.01]
        colours = {0.90: "#1a9641", 0.75: "#a6d96a", 0.50: "#fdae61", 0.25: "#d7191c",
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

        rand_zero  = [r for r in sample if r["SNR"] == 0]
        rand_below = [r for r in sample if 0 < r["SNR"] <= min_SNR]

        ax1.scatter([r["angle_zy"] for r in rand_zero],  [r["range_m"] for r in rand_zero],
                    c="black", alpha=0.3, s=1, label="SNR = 0")
        ax1.scatter([r["angle_zy"] for r in rand_below], [r["range_m"] for r in rand_below],
                    c="red",   alpha=0.3, s=1, label=f"0 < SNR <= {min_SNR}")
        ax1.scatter([r["angle_zy"] for r in rand_above], [r["range_m"] for r in rand_above],
                    c="blue",  s=1,       label=f"SNR > {min_SNR}")
        ax1.set_title("Range vs Bearing (SNR)")
        ax1.legend(loc="upper right")
        ax1.set_rlim(0, max_r)
        ax2.set_rlim(0, max_r)

        for t in reversed(thresholds):
            smoothed = smooth(threshold_ranges[t])
            ax2.plot(angle_centres, smoothed, linewidth=2, c=colours[t], label=f"{int(t * 100)}% detection")
        ax2.set_title("Detection Range vs Bearing")
        ax2.legend()

    max_range = 10000
    results   = monte_carlo_SNR(max_range, pulse_energy_J, 30000)

    detect_SNR         = 1
    dark_current       = 25
    QE                 = 1
    optical_efficiency = 1
    photon_detect_lim  = detect_SNR * dark_current / (QE * optical_efficiency)

    plotter()
    plot_solar_tracker()

    plt.tight_layout()
    plt.show()
