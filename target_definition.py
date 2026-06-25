"""
target_definition.py

Shared target geometry and coordinate frame definition for the sensor performance
models (LiDAR_Performance, Radar_Performance). Single source of truth so both
sensors observe the same target in the same world frame.

Coordinate frame (shared by both sensors):
  +Z is the sensor boresight and range axis; the target sits downrange at +Z.
  X and Y are the transverse axes (the LiDAR beam pointing offset is in X, Y).
  Target orientation is roll, pitch, yaw in RADIANS, intrinsic ZYX (yaw first,
  then pitch, then roll), matching rotation_matrix in the cuboid models.

The target is modelled as a cuboid. The LiDAR uses the full length / width /
height and per face materials for its BRDF; the radar uses characteristic_length()
as a scalar size proxy to scale its RCS table. Set the dimensions to the real
target with configure() at sim init.
"""

import numpy as np

# Target geometry (cuboid)
DEFAULT_LENGTH = 0.1   # m
DEFAULT_WIDTH  = 0.1   # m
DEFAULT_HEIGHT = 0.1   # m
DEFAULT_FACE_MATERIALS = ["Lambertian 20%"] * 6  # per face +X,-X,+Y,-Y,+Z,-Z; keys into lidar_params.json (LiDAR only)

# Target orientation (radians, intrinsic ZYX)
DEFAULT_ROLL  = np.radians(45)
DEFAULT_PITCH = np.radians(45)
DEFAULT_YAW   = np.radians(45)


def characteristic_length(length=None, width=None, height=None):
    """Mean cuboid edge length (m): a scalar size proxy, e.g. for radar RCS scaling.

    Defaults to the module target dimensions when arguments are omitted. The same
    mean edge formula is used by the LiDAR return model for its return spot width.
    """
    if length is None: length = DEFAULT_LENGTH
    if width is None:  width  = DEFAULT_WIDTH
    if height is None: height = DEFAULT_HEIGHT
    return (length + width + height) / 3


# Settable target parameters; configure() validates keyword names against this set.
_CONFIG_KEYS = {
    "DEFAULT_LENGTH", "DEFAULT_WIDTH", "DEFAULT_HEIGHT",
    "DEFAULT_ROLL", "DEFAULT_PITCH", "DEFAULT_YAW",
    "DEFAULT_FACE_MATERIALS",
}


def configure(**overrides):
    """Set the shared target geometry / orientation globally at sim initialisation.

    Pass any of the settable globals (see _CONFIG_KEYS) as keywords; unknown names
    raise KeyError. Both sensor modules read these at call time, so the new target
    takes effect on the next call. Returns the resulting target configuration as a dict.
    """
    unknown = set(overrides) - _CONFIG_KEYS
    if unknown:
        raise KeyError(f"Unknown target parameter(s): {sorted(unknown)}")
    globals().update(overrides)
    return {key: globals()[key] for key in _CONFIG_KEYS}
