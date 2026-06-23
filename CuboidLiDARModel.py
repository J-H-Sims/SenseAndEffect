"""
CuboidLiDARModel.py

Monostatic LiDAR return model for a cuboid target.

Monostatic means the transmitter and receiver are co-located, so illumination
and observation share the same direction. A single angle (cos_theta, between
the face normal and the sensor boresight) therefore drives both the Lambertian
and specular terms of the BRDF. Material coefficients are loaded from
lidar_params.json as (kd, kr, beta) per material name.
"""

import json
import numpy as np

with open("lidar_params.json", "r") as f:
    material_params = json.load(f)

def rotation_matrix(roll, pitch, yaw):
    """Build a 3x3 rotation matrix from Euler angles (radians).

    Composition order is Rz @ Ry @ Rx, so yaw is applied first in the body
    frame, then pitch, then roll (intrinsic ZYX convention).
    """
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll),-np.sin(roll)],
                   [0,np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw), np.cos(yaw),0],
                   [0,0,1]])
    return Rz @ Ry @ Rx

def cuboid_faces():
    """Return unit outward normals for each of the 6 cuboid faces in the local body frame.

    Takes no dimensions: normals are direction only. They are not yet rotated
    into the world/observer frame (that happens in the caller), and face areas
    are computed in the caller rather than here. Face order: +X, -X, +Y, -Y, +Z, -Z.
    """
    normals = np.array([
        [1,0,0], [-1,0,0],  # X faces
        [0,1,0], [0,-1,0],  # Y faces
        [0,0,1], [0,0,-1]   # Z faces
    ])
    return normals

def lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials):
    """Compute the total normalised LiDAR return (J/J) from all visible faces.

    face_materials: list of 6 strings matching keys in material_params.

    Monostatic geometry: obs_dir (+Z) serves as both illumination and viewing
    direction, so a single cos_theta drives both BRDF terms.
    """
    obs_dir = np.array([0,0,1])  # sensor boresight in world frame
    R = rotation_matrix(roll, pitch, yaw)
    normals = cuboid_faces()
    normals_rot = normals @ R.T  # rotate body-frame normals into world frame

    face_areas = np.array([
        height * width,   # +X
        height * width,   # -X
        length * height,  # +Y
        length * height,  # -Y
        length * width,   # +Z
        length * width    # -Z
    ])

    total_return = 0.0
    target_area = 0.0

    # Iterate over each face: accumulate BRDF-weighted projected area.
    for n, mat, A_face in zip(normals_rot, face_materials, face_areas):
        cos_theta = np.dot(n, obs_dir)
        if cos_theta <= 0:
            # Back-facing face — no illumination and not visible to the sensor.
            continue

        # kd: Lambertian (diffuse) coefficient
        # kr: specular coefficient
        # beta: specular lobe width (higher = narrower highlight)
        kd, kr, beta = material_params[mat]

        # Combined Lambertian + specular BRDF (monostatic, so same angle for both).
        r_frac = (kd/np.pi) * cos_theta + kr * cos_theta ** beta

        # Multiply by A_face * cos_theta to get the projected (foreshortened)
        # face area as seen by the sensor, then weight by the BRDF.
        total_return += r_frac * A_face * cos_theta
        target_area  += A_face * cos_theta  # purely geometric projected area

    return total_return, target_area

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    length, width, height = 0.01, 0.01, 0.01

    roll  = np.deg2rad(45)
    pitch = np.deg2rad(45)
    yaw   = np.deg2rad(45)

    face_materials = ["Lambertian 20%"] * 6

    total_power, target_area = lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)
    print(f"Total LiDAR return for cuboid: {total_power:.6f}  J/J")
    print(target_area)
