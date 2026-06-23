"""
CuboidSolarModel.py

Bistatic passive-observation solar reflection model for a cuboid target.

Unlike the monostatic LiDAR model, illumination (sun) and observation
(sensor or eye) are independent directions. Each face is evaluated with
separate incident (cos_i) and exitant (cos_o) angles. The specular term
uses Phong-style geometry: the ideal mirror-reflection of the sun direction
about the face normal is computed, and its alignment with the observer
direction determines the specular highlight intensity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

with open("lidar_params.json") as f:
    material_params = json.load(f)

# -----------------------------
# Geometry functions
# -----------------------------
def rotation_matrix(roll, pitch, yaw):
    """Build a 3x3 rotation matrix (intrinsic ZYX: yaw first, then pitch, then roll)."""
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx

def cuboid_faces(length, width, height):
    """Return outward unit normals and face areas for each of the 6 cuboid faces.

    Unlike the LiDAR version, this returns areas alongside normals so callers
    do not need to recompute them. Face order: +X, -X, +Y, -Y, +Z, -Z.
    """
    normals = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    areas   = np.array([height*width, height*width, length*height, length*height, length*width, length*width])
    return normals, areas

# -----------------------------
# Solar reflection model
# -----------------------------
def solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir):
    """Compute total reflected solar intensity seen by the observer.

    illum_dir : unit vector in global frame pointing TO the sun
    obs_dir   : unit vector in global frame pointing TO the observer

    Bistatic geometry: cos_i (incident, sun side) and cos_o (exitant, observer
    side) are computed separately. Both are clamped to zero so back-facing
    geometry is excluded; a face must be both lit AND visible to contribute.

    Returns (total_return, contributions) where contributions is a per-face array.
    """
    illum_dir = illum_dir / np.linalg.norm(illum_dir)
    obs_dir   = obs_dir   / np.linalg.norm(obs_dir)

    R = rotation_matrix(roll, pitch, yaw)
    normals, areas = cuboid_faces(length, width, height)
    normals_rot = normals @ R.T  # rotate body-frame normals into world frame

    total_return  = 0.0
    contributions = []

    for n, A_face, mat in zip(normals_rot, areas, face_materials):
        cos_i = max(0.0, np.dot(n, illum_dir))  # how directly the sun hits the face
        cos_o = max(0.0, np.dot(n, obs_dir))    # how directly the observer sees the face
        if cos_i <= 0 or cos_o <= 0:
            # Face is either unlit or not visible from the observer — skip.
            contributions.append(0.0)
            continue

        kd, kr, beta = material_params[mat]

        # Lambertian term: scattered energy proportional to incident irradiance (cos_i).
        lambert = (kd/np.pi) * cos_i

        # Specular term: r_ideal is the mirror-reflection of illum_dir about n.
        # cos_spec measures how well the observer aligns with that mirror direction.
        # This is Phong-style specular adapted for bistatic (sun != sensor) geometry.
        r_ideal  = 2 * np.dot(n, illum_dir) * n - illum_dir
        cos_spec = max(0.0, np.dot(r_ideal, obs_dir))
        specular = kr * cos_spec ** beta

        # cos_o projects the face area onto the observer's plane of sight.
        projected   = A_face * cos_o
        face_return = (lambert + specular) * projected

        contributions.append(face_return)
        total_return += face_return

    return total_return, np.array(contributions)

# -----------------------------
# 3D plotting
# -----------------------------
def plot_cuboid(length, width, height, roll, pitch, yaw, illum_dir, obs_dir):
    """Render the rotated cuboid with sun and observer direction arrows."""
    l, w, h = length/2, width/2, height/2
    vertices = np.array([
        [ l,  w,  h],[ l, -w,  h],[-l, -w,  h],[-l,  w,  h],
        [ l,  w, -h],[ l, -w, -h],[-l, -w, -h],[-l,  w, -h]
    ])
    R = rotation_matrix(roll, pitch, yaw)
    verts_rot = vertices @ R.T

    faces_idx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')

    for idx in faces_idx:
        poly = Poly3DCollection([verts_rot[idx]], alpha=0.5, facecolor='lightgray')
        ax.add_collection3d(poly)

    center = np.array([0,0,0])
    ax.scatter(*center, color='k', s=20)

    illum_dir = illum_dir / np.linalg.norm(illum_dir)
    obs_dir   = obs_dir   / np.linalg.norm(obs_dir)
    ax.quiver(*center, *(illum_dir*2), color='orange', linewidth=2, label='Sun')
    ax.quiver(*center, *(obs_dir*2),   color='blue',   linewidth=2, label='Observer')

    max_dim = max(length, width, height)
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([-max_dim, max_dim])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Cuboid with Sun and Observer Vectors')
    ax.legend()
    plt.show()


def yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir, steps=360):
    """Sweep the cuboid's yaw from 0 to 360 deg while holding roll and pitch fixed.

    Simulates a tumbling object observed from a fixed sun/sensor geometry.
    Returns (yaws_deg, returns) arrays for plotting or analysis.
    """
    yaws    = np.linspace(0, 2*np.pi, steps)
    returns = []

    for yaw in yaws:
        total, _ = solar_return_cuboid(
            length, width, height,
            roll, pitch, yaw,
            face_materials,
            illum_dir,
            obs_dir
        )
        returns.append(total)

    return np.rad2deg(yaws), np.array(returns)


def plot_yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir):
    """Plot reflected solar return as a function of yaw angle."""
    yaws_deg, returns = yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir)
    plt.figure(figsize=(8,4))
    plt.plot(yaws_deg, returns, '-o')
    plt.xlabel('Yaw angle (deg)')
    plt.ylabel('Reflected intensity')
    plt.title('Cuboid reflected solar return vs yaw angle')
    plt.grid(True)
    plt.show()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    length, width, height = 0.01, 0.01, 0.01
    roll, pitch = np.deg2rad(0), np.deg2rad(0)
    face_materials = ["Lambertian 20%"] * 6

    obs_dir   = np.array([0, 1, 0])
    illum_dir = np.array([0, 1, 0])

    plot_yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir)
