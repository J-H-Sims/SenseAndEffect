import json
import numpy as np

# Load material parameters
with open("lidar_params.json", "r") as f:
    material_params = json.load(f)

def rotation_matrix(roll, pitch, yaw):
    """Roll around x, pitch around y, yaw around z"""
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

def cuboid_faces(length, width, height):
    """
    Returns:
        normals: 6x3 array of face normals in local frame
    Face order: +X, -X, +Y, -Y, +Z, -Z
    """
    normals = np.array([
        [1,0,0], [-1,0,0],  # X faces
        [0,1,0], [0,-1,0],  # Y faces
        [0,0,1], [0,0,-1]   # Z faces
    ])
    return normals

def lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials):
    """
    face_materials: list of 6 strings matching keys in material_params
    Returns total LiDAR return from all visible faces
    """
    obs_dir = np.array([0,0,1])  # observer along +Z
    R = rotation_matrix(roll, pitch, yaw)
    normals = cuboid_faces(length, width, height)
    normals_rot = normals @ R.T  # rotate into observer frame

    face_areas = np.array([
        height * width,  # +X
        height * width,  # -X
        length * height,  # +Y
        length * height,  # -Y
        length * width,  # +Z
        length * width  # -Z
    ])

    total_return = 0.0
    for n, mat, A_face in zip(normals_rot, face_materials, face_areas):
        cos_theta = np.dot(n, obs_dir)
        if cos_theta <= 0:
            continue
        kd, kr, beta = material_params[mat]
        r_frac = kd * cos_theta + kr * cos_theta ** beta
        total_return += r_frac * A_face * cos_theta  # projected area included
    return total_return

# -----------------------------
# Example usage
# -----------------------------

# Cuboid size (meters)
length, width, height = 2.0, 2.0, 2

# Orientation in radians
roll = np.deg2rad(90)
pitch = np.deg2rad(0)
yaw = np.deg2rad(0)

# Materials per face: +X, -X, +Y, -Y, +Z, -Z
face_materials = [
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al",
    "Brushed V Al"
]

total_power = lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)
print(f"Total LiDAR return for cuboid: {total_power:.6f}  J/J")