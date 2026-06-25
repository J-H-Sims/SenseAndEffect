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

## Shared target and coordinate frame

The target and the world frame are defined once in `target_definition.py` and
imported by both sensor modules (as `tgt`), so a single target is observed
consistently:

- Geometry: `tgt.DEFAULT_LENGTH`, `tgt.DEFAULT_WIDTH`, `tgt.DEFAULT_HEIGHT` (cuboid, m);
  `tgt.DEFAULT_FACE_MATERIALS` (six face materials, LiDAR only).
- Orientation: `tgt.DEFAULT_ROLL`, `tgt.DEFAULT_PITCH`, `tgt.DEFAULT_YAW` (radians, intrinsic ZYX).
- Frame: +Z is the boresight and range axis (target downrange at +Z); X, Y are transverse.
- The LiDAR uses the full cuboid; the radar uses `tgt.characteristic_length()`
  (mean edge length) as a scalar size proxy to scale its RCS table.

Set the target with `tgt.configure(...)`; both sensors see the change on the next call.

## Configuring at sim initialisation

Each module exposes a `configure(**overrides)`; the target has its own in
`target_definition`. All validate keyword names against the module's
`_CONFIG_KEYS` (unknown names raise `KeyError`) and return the resulting config
as a dict:

```python
import target_definition as tgt
import LiDAR_Performance as lidar
import Radar_Performance as radar

tgt.configure(DEFAULT_LENGTH=5, DEFAULT_WIDTH=5, DEFAULT_HEIGHT=5)   # shared target
lidar.configure(pulse_energy_J=4e-4, aperture_radius=0.03, range_resolution=1.5)
radar.configure(Pt_radar=500, radar_gain=30)
```

Behaviour notes:
- LiDAR derived globals are recomputed automatically: `aperture_area_m2` from
  `aperture_radius`, `exposure_time` from `range_resolution`. Set the source
  parameter, not the derived one.
- The compute functions read these globals at call time (the LiDAR geometry /
  theta arguments resolve to `tgt`/the globals when left as `None`), so
  `configure()` values are honoured on the next call without redefining anything.

---

## `compute_lidar_returns`

```python
compute_lidar_returns(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y,
                      theta=None, length=None, width=None, height=None, face_materials=None)
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

### Optional arguments (None resolves to the shared/global default at call time)

| Argument | Resolves to | Units | Meaning |
|---|---|---|---|
| `theta` | `half_beam_divergence_rad` | radians | Laser half angle divergence override. |
| `length` | `tgt.DEFAULT_LENGTH` | m | Target cuboid length. |
| `width` | `tgt.DEFAULT_WIDTH` | m | Target cuboid width. |
| `height` | `tgt.DEFAULT_HEIGHT` | m | Target cuboid height. |
| `face_materials` | `tgt.DEFAULT_FACE_MATERIALS` | list[str] | Six material names, one per face in order +X, -X, +Y, -Y, +Z, -Z. Each must be a key in `lidar_params.json`. |

Target geometry defaults live in `target_definition.py` (shared with the radar);
pass them explicitly or set them with `tgt.configure(...)`.

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
| `PLANCK_CONSTANT`, `SPEED_OF_LIGHT` | SI | Physical constants. |

Target geometry (`tgt.DEFAULT_LENGTH/WIDTH/HEIGHT`, `tgt.DEFAULT_FACE_MATERIALS`,
orientation) lives in `target_definition.py`, not this module.

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
| `orientation_deg` | float | degrees, -180 to 180 | Target azimuth for the RCS table lookup (linearly interpolated). 1 DOF in degrees: an inherent difference from the LiDAR 3 DOF radian attitude, because the RCS table is azimuth only. |
| `R` | float | m | Sensor to target range (the LiDAR calls the same quantity `range_m`). |
| `pointing` | str | `'zenith'`, `'nadir'`, `'sun'`, `'anti-sun'`, `'ram'` | Spacecraft face orientation for the flux lookup. `get_pointing_from_azimuth` produces the first four. |
| `case` | str | `'hot'` or `'cold'` | Bounding thermal environment case. |
| `beta` | int | `0`, `45`, `70`, `90` (deg) | Solar beta angle (orbital plane to sun vector). |

### Module configuration read (set these to define the sensor)

| Global | Units | Role |
|---|---|---|
| `Pt_radar` | W | Peak transmit power. |
| `radar_gain` | — | Antenna gain (fixed, or precompute with `Gain_Approx`). |
| `lambda_radar` | m | Radar wavelength. |
| `Pr_radar_min` | W | Minimum detectable received power (detection floor). |
| `B` | Hz | Receiver bandwidth. |
| `DEFAULT_RADOME_ABSORPTIVITY` | — | Radome surface solar absorptivity (receiver, not target). |
| `DEFAULT_RADOME_EMISSIVITY` | — | Radome surface thermal emissivity (receiver, not target). |
| `DEFAULT_RADOME_PROJECTED_AREA` | m^2 | Sun-facing radome area absorbing incident flux. |
| `DEFAULT_RADOME_RADIATING_AREA` | m^2 | Radome area radiating heat to space. |
| `DEFAULT_BUS_POWER` | W | Spacecraft bus power dissipated as heat into the radome. |
| `k`, `sb` | SI | Boltzmann and Stefan Boltzmann constants. |

Target size is **not** in this module: it comes from `target_definition.py` via
`tgt.characteristic_length()`. The radome thermal properties above belong to the
**receiver** and set its temperature, not the target.

### Returns

`(SNR, Pr, P_noise)`

| Element | Units | Meaning |
|---|---|---|
| `SNR` | — | `Pr / P_noise`, or `0` if `Pr < Pr_radar_min`. |
| `Pr` | W | Received power from the monostatic radar equation. |
| `P_noise` | W | Thermal noise floor `k * T * B`. |

`T` is the **radome (receiver)** equilibrium temperature from a power balance
(absorbed flux plus bus power equals radiated flux):

```
T = ((S * DEFAULT_RADOME_ABSORPTIVITY * DEFAULT_RADOME_PROJECTED_AREA + DEFAULT_BUS_POWER)
     / (DEFAULT_RADOME_EMISSIVITY * DEFAULT_RADOME_RADIATING_AREA * sb)) ** 0.25
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
- The RCS scale factor is `get_rcs_m2(orientation) * tgt.characteristic_length()^2 / (0.3 * 0.2)`,
  scaling the reference 6U cubesat projected area (0.3 x 0.2 m) up to the shared target.
- No side effects.

---

## Plugin integration summary

| | LiDAR | Radar |
|---|---|---|
| Per call geometry | range, attitude (roll/pitch/yaw), sun direction, beam offset | range, target azimuth, orbital pointing, thermal case, beta |
| Configure once (sensor globals) | laser, receiver, optics, detection floor | transmitter, receiver, radome thermal properties |
| Target (shared) | `target_definition.py` — cuboid dims, orientation, materials | same module, used as `characteristic_length()` for RCS |
| Returns | `(SNR, signal, solar_total, solar_direct)` | `(SNR, Pr, P_noise)` |
| Detection floor | photon count (`min_photons_to_detect`) | received power (`Pr_radar_min`) |
| Side effects | none | none |

System configuration is set globally via `configure()` at sim init (see above).
Open items before drop in plugin use (tracked separately, not yet done): config
is global to the module, so one process cannot run two differently configured
sensors of the same type at once (no per instance state); the radiant flux
altitude is fixed at 500 km; and the cuboid models load `lidar_params.json` by
relative path.
