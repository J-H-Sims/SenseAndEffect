# Sensor return functions reference

Detailed reference for the two per sample physics functions intended for use as
Basilisk spacecraft sim plugins:

- `compute_lidar_returns` in `LiDAR_Performance.py`
- `compute_radar_returns` in `Radar_Performance.py`

Both are pure (no side effects, no shared mutable state) and both lead their
return tuple with `SNR`. Each reads a set of module level configuration globals
in addition to its per call arguments; a plugin configures the sensor by setting
those globals once, then calls the function each timestep with the changing
geometry.

## Configuring a system at sim initialisation

Each module exposes a `configure(**overrides)` function that sets its
configuration globals in one call, intended for sim init:

```python
import LiDAR_Performance as lidar
import Radar_Performance as radar

lidar.configure(pulse_energy_J=4e-4, aperture_radius=0.03, range_resolution=1.5,
                DEFAULT_LENGTH=1.0, DEFAULT_WIDTH=1.0, DEFAULT_HEIGHT=1.0)
radar.configure(Pt_radar=200, radar_gain=30, DEFAULT_TARGET_CHARACTERISTIC_LENGTH=10)
```

Behaviour:
- Keyword names are validated against the module's `_CONFIG_KEYS`; an unknown
  name raises `KeyError`.
- LiDAR derived globals are recomputed automatically: `aperture_area_m2` from
  `aperture_radius`, `exposure_time` from `range_resolution`. Set the source
  parameter, not the derived one.
- Returns the resulting configuration as a dict for logging.
- The compute functions read these globals at call time (the LiDAR geometry /
  theta arguments resolve to the globals when left as `None`), so `configure()`
  values are honoured on the next call without redefining anything.

---

## `compute_lidar_returns`

```python
compute_lidar_returns(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y,
                      theta=half_beam_divergence_rad,
                      length=DEFAULT_LENGTH, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                      face_materials=DEFAULT_FACE_MATERIALS)
```

Computes the LiDAR signal to background ratio for a single look at a cuboid
target, combining a Gaussian beam signal return with a solar background (diffuse
target reflection plus a direct glare term).

### Required arguments (per call geometry)

| Argument | Type | Units | Meaning |
|---|---|---|---|
| `range_m` | float | m | Sensor to target range along boresight. |
| `pulse_energy_J` | float | J | Transmitted laser pulse energy. |
| `roll` | float | **radians** | Target orientation, X axis of intrinsic ZYX rotation. |
| `pitch` | float | **radians** | Target orientation, Y axis. |
| `yaw` | float | **radians** | Target orientation, Z axis. |
| `illum_dir` | 3 vector | unit | Direction to the sun in the sensor world frame (+Z is boresight). Normalised internally. |
| `X` | float | m | Transverse beam pointing offset at the target in the cross boresight X axis (beam centred at 0,0). |
| `Y` | float | m | Transverse beam pointing offset in the cross boresight Y axis. |

### Optional arguments (default to module globals)

| Argument | Default | Units | Meaning |
|---|---|---|---|
| `theta` | `half_beam_divergence_rad` | radians | Laser half angle divergence override. |
| `length` | `DEFAULT_LENGTH` | m | Target cuboid length. |
| `width` | `DEFAULT_WIDTH` | m | Target cuboid width. |
| `height` | `DEFAULT_HEIGHT` | m | Target cuboid height. |
| `face_materials` | `DEFAULT_FACE_MATERIALS` | list[str] | Six material names, one per face in order +X, -X, +Y, -Y, +Z, -Z. Each must be a key in `lidar_params.json`. |

### Module configuration read (set these to define the sensor)

| Global | Units | Role |
|---|---|---|
| `beam_waist` | m | Beam waist at focus, drives the Gaussian spot size. |
| `wavelength_nm` | nm | Laser wavelength, sets photon energy. |
| `aperture_area_m2` | m^2 | Receiver collecting area (derived from `aperture_radius`). |
| `bandwidth` | nm | Optical bandpass filter width, scales solar background. |
| `polarity_filter` | fraction | Fraction of solar light passed by the polarisation filter. |
| `SOLAR_SPECTRAL_IRRADIANCE_W_M2_NM` | W/m^2/nm | Solar spectral irradiance at the wavelength. |
| `exposure_time` | s | Background integration gate (derived from `range_resolution / c`). |
| `min_photons_to_detect` | count | Detection floor (see detection logic). |
| `half_beam_divergence_rad` | radians | Default for `theta`. |
| `DEFAULT_LENGTH` / `DEFAULT_WIDTH` / `DEFAULT_HEIGHT` / `DEFAULT_FACE_MATERIALS` | — | Defaults for the geometry arguments. |
| `PLANCK_CONSTANT`, `SPEED_OF_LIGHT` | SI | Physical constants. |

### Returns

`(SNR, photons_per_pulse, reflected_solar_photons_per_pulse, direct_solar_photons)`

| Element | Units | Meaning |
|---|---|---|
| `SNR` | — | `photons_per_pulse / reflected_solar_photons_per_pulse`. `np.inf` if the background is zero; `0` if below the detection floor. |
| `photons_per_pulse` | count | Signal photons collected per pulse. |
| `reflected_solar_photons_per_pulse` | count | Total solar background per gate (diffuse target reflection plus direct glare). |
| `direct_solar_photons` | count | The direct glare component alone (a subset of the total above). |

### Detection logic

`SNR` is forced to `0` when

```
photons_per_pulse + (reflected_solar_photons_per_pulse - direct_solar_photons) < min_photons_to_detect
```

that is, when the signal plus the diffuse background (glare excluded) falls below
the photon floor.

### Dependencies and conventions

- Sensor boresight is +Z; the target sits downrange at +Z; `illum_dir` is in the
  same world frame. Observer direction is fixed internally at `[0, 0, 1]`.
- `roll`, `pitch`, `yaw` are in radians, intrinsic ZYX (yaw, then pitch, then roll).
- Calls `CuboidLiDARModel.lidar_return_cuboid`, `CuboidSolarModel.solar_return_cuboid`,
  and `GaussianBeam.gaussian_beam_wm2`. The two cuboid modules open
  `lidar_params.json` at import using a relative path, so the working directory
  must contain that file (a known caveat for plugin use).
- No side effects: the per sample solar bookkeeping (`solar_tracker`) is the
  caller's responsibility.

---

## `compute_radar_returns`

```python
compute_radar_returns(orientation_deg, R, pointing, case, beta)
```

Computes the radar signal to noise ratio for a single look, using the monostatic
radar equation for received power and a Johnson Nyquist thermal noise floor whose
temperature is set by the orbital radiant flux environment.

### Required arguments (per call geometry and environment)

| Argument | Type | Units / domain | Meaning |
|---|---|---|---|
| `orientation_deg` | float | degrees, -180 to 180 | Target azimuth for the RCS table lookup (linearly interpolated). |
| `R` | float | m | Sensor to target range. |
| `pointing` | str | `'zenith'`, `'nadir'`, `'sun'`, `'anti-sun'`, `'ram'` | Spacecraft face orientation for the flux lookup. `get_pointing_from_azimuth` produces the first four. |
| `case` | str | `'hot'` or `'cold'` | Bounding thermal environment case. |
| `beta` | int | `0`, `45`, `70`, `90` (deg) | Solar beta angle (orbital plane to sun vector). |

### Module configuration read (set these to define the sensor and target)

| Global | Units | Role |
|---|---|---|
| `Pt_radar` | W | Peak transmit power. |
| `radar_gain` | — | Antenna gain (fixed, or precompute with `Gain_Approx`). |
| `lambda_radar` | m | Radar wavelength. |
| `Pr_radar_min` | W | Minimum detectable received power (detection floor). |
| `B` | Hz | Receiver bandwidth. |
| `DEFAULT_TARGET_CHARACTERISTIC_LENGTH` | m | Target size; scales RCS from the 6U cubesat reference. |
| `DEFAULT_ABSORPTION` | — | Surface solar absorptivity. |
| `DEFAULT_EMISSIVITY` | — | Surface thermal emissivity. |
| `DEFAULT_AP` | m^2 | Projected area for solar absorption. |
| `DEFAULT_AR` | m^2 | Radiating area. |
| `DEFAULT_AVG_POWER` | W | Average bus power dissipated as heat. |
| `k`, `sb` | SI | Boltzmann and Stefan Boltzmann constants. |

### Returns

`(SNR, Pr, P_noise)`

| Element | Units | Meaning |
|---|---|---|
| `SNR` | — | `Pr / P_noise`, or `0` if `Pr < Pr_radar_min`. |
| `Pr` | W | Received power from the monostatic radar equation. |
| `P_noise` | W | Thermal noise floor `k * T * B`. |

`T` is the target equilibrium temperature from a power balance (absorbed flux plus
bus power equals radiated flux):

```
T = ((S * DEFAULT_ABSORPTION * DEFAULT_AP + DEFAULT_AVG_POWER) / (DEFAULT_EMISSIVITY * DEFAULT_AR * sb)) ** 0.25
```

where `S` is the total incident radiant flux for the `(case, pointing, beta)`
environment.

### Detection logic

`SNR` is forced to `0` when `Pr < Pr_radar_min`, regardless of the noise level.

### Dependencies and conventions

- Calls `radar_cross_section.get_rcs_m2` (RCS table in dBsm, converted to m^2 and
  interpolated) and `radiant_flux.get_radiant_flux`. Received power uses
  `radar_received_power` (the single source of truth radar equation).
- **Altitude is hardcoded to 500 km** in the radiant flux lookup. If a plugin
  needs a different altitude, this must be exposed as a parameter (700 km is also
  tabulated).
- The RCS scale factor is `get_rcs_m2(orientation) * L^2 / (0.3 * 0.2)`, scaling
  the reference 6U cubesat projected area (0.3 x 0.2 m) up to the target.
- No side effects.

---

## Plugin integration summary

| | LiDAR | Radar |
|---|---|---|
| Per call geometry | range, attitude (roll/pitch/yaw), sun direction, beam offset | range, target azimuth, orbital pointing, thermal case, beta |
| Configure once (globals) | laser, receiver, optics, detection floor, default target | transmitter, receiver, target, thermal properties |
| Returns | `(SNR, signal, solar_total, solar_direct)` | `(SNR, Pr, P_noise)` |
| Detection floor | photon count (`min_photons_to_detect`) | received power (`Pr_radar_min`) |
| Side effects | none | none |

System configuration is set globally via `configure()` at sim init (see above).
Open items before drop in plugin use (tracked separately, not yet done): config
is global to the module, so one process cannot run two differently configured
sensors of the same type at once (no per instance state); the radiant flux
altitude is fixed at 500 km; and the cuboid models load `lidar_params.json` by
relative path.
