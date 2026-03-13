import numpy as np
import sympy as sp

# ── Physical constants ────────────────────────────────────────────────
PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT  = 299792458

# ── Fixed system parameters ───────────────────────────────────────────
divergence_rad         = 0.018
target_area_m2         = 1
target_reflectance     = 0.1
scatter_scalar         = 2
aperture_area_m2       = 3.14 * (0.015**2)
pulse_width_s          = 1.3e-9
wavelength_nm          = 1470
solar_return_divergence_rad = 0.05
solar_intensity_per_nm = 400000
solar_bandwidth        = 10
solar_polarity_filter_pass = 1
SNR_BASELINE           = 0.30  # 30%

# ── Photon energy (constant) ──────────────────────────────────────────
photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)


def compute_signal_photons(range_m, emitted_power_W):
    """Signal photons per pulse for a given range and emitted power."""
    spot_size_m              = range_m * np.tan(divergence_rad)
    laser_intensity          = emitted_power_W / (np.pi * (spot_size_m / 2)**2)
    incident_energy_W        = laser_intensity * target_area_m2
    reflected_energy_W       = target_reflectance * incident_energy_W
    char_dim_m               = target_area_m2**0.5
    return_div_rad           = scatter_scalar * divergence_rad
    return_spot_m            = (range_m * np.tan(return_div_rad)) + char_dim_m
    return_intensity         = reflected_energy_W / (np.pi * (return_spot_m / 2)**2)
    collected_power_W        = return_intensity * aperture_area_m2
    photons_per_second       = collected_power_W / photon_energy_J
    return photons_per_second * pulse_width_s


def compute_solar_photons(range_m):
    """Solar background photons per pulse for a given range."""
    solar_intensity_apparent = solar_intensity_per_nm * solar_bandwidth * solar_polarity_filter_pass
    solar_incident_W         = solar_intensity_apparent * target_area_m2
    solar_reflected_W        = target_reflectance * solar_incident_W
    char_dim_m               = target_area_m2**0.5
    solar_spot_m             = (range_m * np.tan(solar_return_divergence_rad)) + char_dim_m
    solar_intensity_return   = solar_reflected_W / (np.pi * (solar_spot_m / 2)**2)
    solar_collected_W        = solar_intensity_return * aperture_area_m2
    solar_photons_per_second = solar_collected_W / photon_energy_J
    return solar_photons_per_second * pulse_width_s


from scipy.optimize import brentq

def snr_at_range(range_m, emitted_power_W):
    sig = compute_signal_photons(range_m, emitted_power_W)
    sol = compute_solar_photons(range_m)
    return sig / sol

def solve_for_range(emitted_power_W, snr_target=SNR_BASELINE):
    """Find max range where SNR drops to snr_target."""
    f = lambda r: snr_at_range(r, emitted_power_W) - snr_target

    # Quick diagnostic
    for r_test in [100, 1000, 5000, 10000, 50000]:
        print(f"  SNR at {r_test} m: {snr_at_range(r_test, emitted_power_W)*100:.2f}%")

    # Find bracket where SNR crosses the target
    r_low, r_high = 1, 1e7
    if f(r_low) < 0:
        raise RuntimeError("SNR already below target at 1 m - check parameters")
    if f(r_high) > 0:
        raise RuntimeError("SNR still above target at 10,000 km - check parameters")

    return brentq(f, r_low, r_high, xtol=1)

def solve_for_power(range_m, snr_target=SNR_BASELINE):
    """
    Solve for minimum emitted power to meet SNR at a given range.
    Power appears linearly so this is solved directly.
    """
    solar_photons = compute_solar_photons(range_m)
    required_signal_photons = snr_target * solar_photons

    # Signal photons scale linearly with power, so solve by ratio
    signal_at_1W = compute_signal_photons(range_m, emitted_power_W=1)
    required_power_W = required_signal_photons / signal_at_1W
    return required_power_W


# ── Example usage ─────────────────────────────────────────────────────
if __name__ == "__main__":
    emitted_power_W = 30000

    max_range = solve_for_range(emitted_power_W)
    print(f"Max range for {emitted_power_W:.0f} W at {SNR_BASELINE*100:.0f}% SNR: {max_range:.1f} m")

    target_range = 10000
    min_power = solve_for_power(target_range)
    print(f"Min power for {target_range} m at {SNR_BASELINE*100:.0f}% SNR:       {min_power:.2f} W")

    # Sanity check SNR at nominal conditions
    sig  = compute_signal_photons(emitted_power_W=emitted_power_W, range_m=target_range)
    sol  = compute_solar_photons(range_m=target_range)
    print(f"SNR at {target_range} m, {emitted_power_W:.0f} W:                    {sig/sol*100:.2f}%")


sig = compute_signal_photons(emitted_power_W=300000, range_m=10000)
sol = compute_solar_photons(range_m=10000)
print(sig, sol, sig / sol)