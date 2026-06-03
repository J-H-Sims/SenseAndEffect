import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Load material parameters
# -----------------------------
with open("lidar_params.json") as f:
    material_params = json.load(f)

# -----------------------------
# Geometry functions
# -----------------------------
def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx

def cuboid_faces(length, width, height):
    normals = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    areas = np.array([height*width, height*width, length*height, length*height, length*width, length*width])
    return normals, areas

# -----------------------------
# Solar reflection model
# -----------------------------
def solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir):
    """
    illum_dir : unit vector in global frame TO sun
    obs_dir   : unit vector in global frame TO observer
    """
    illum_dir = illum_dir / np.linalg.norm(illum_dir)
    obs_dir   = obs_dir / np.linalg.norm(obs_dir)

    R = rotation_matrix(roll, pitch, yaw)
    normals, areas = cuboid_faces(length, width, height)
    normals_rot = normals @ R.T

    total_return = 0.0
    contributions = []

    for n, A_face, mat in zip(normals_rot, areas, face_materials):
        cos_i = max(0.0, np.dot(n, illum_dir))
        cos_o = max(0.0, np.dot(n, obs_dir))
        if cos_i <= 0 or cos_o <= 0:
            contributions.append(0.0)
            continue

        kd, kr, beta = material_params[mat]

        # Lambertian component (incident-driven)
        lambert = (kd/np.pi) * cos_i

        # Specular/retro component: ideal reflection direction
        r_ideal = 2 * np.dot(n, illum_dir) * n - illum_dir
        cos_spec = max(0.0, np.dot(r_ideal, obs_dir))
        specular = kr * cos_spec ** beta

        projected = A_face * cos_o
        face_return = (lambert + specular) * projected

        contributions.append(face_return)
        total_return += face_return


    return total_return, np.array(contributions)

# -----------------------------
# 3D plotting
# -----------------------------
def plot_cuboid(length, width, height, roll, pitch, yaw, illum_dir, obs_dir):
    l, w, h = length/2, width/2, height/2
    vertices = np.array([
        [ l,  w,  h],[ l, -w,  h],[-l, -w,  h],[-l,  w,  h],
        [ l,  w, -h],[ l, -w, -h],[-l, -w, -h],[-l,  w, -h]
    ])
    R = rotation_matrix(roll, pitch, yaw)
    verts_rot = vertices @ R.T

    faces_idx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for idx in faces_idx:
        poly = Poly3DCollection([verts_rot[idx]], alpha=0.5, facecolor='lightgray')
        ax.add_collection3d(poly)

    center = np.array([0,0,0])
    ax.scatter(*center, color='k', s=20)

    illum_dir = illum_dir / np.linalg.norm(illum_dir)
    obs_dir = obs_dir / np.linalg.norm(obs_dir)
    ax.quiver(*center, *(illum_dir*2), color='orange', linewidth=2, label='Sun')
    ax.quiver(*center, *(obs_dir*2), color='blue', linewidth=2, label='Observer')

    max_dim = max(length, width, height)
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([-max_dim, max_dim])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Cuboid with Sun and Observer Vectors')
    ax.legend()
    plt.show()


# #-----------------------------
# #Example usage
# #-----------------------------
# if __name__ == "__main__":
#     length, width, height = 2.0, 1.0, 0.5
#     roll, pitch, yaw = np.deg2rad(20), np.deg2rad(10), np.deg2rad(30)
#     face_materials = ["Brushed V Al"]*6
#
#     # Observer direction (LiDAR)
#     obs_dir = np.array([0, 0, 1])
#
#     # Solar illumination in global frame (unit vector)
#     illum_dir = np.array([0.5, 0.0, 0.866])  # example: ~30° elevation, 0° azimuth
#
#     total, contributions = solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir)
#     print(f"Total solar return: {total:.6f}")
#     print(f"Per-face contributions: {contributions}")
#
#     plot_cuboid(length, width, height, roll, pitch, yaw, illum_dir, obs_dir)
def yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir, steps=360):
    """
    Rotates the cuboid around Z-axis (yaw) from 0 to 360 deg in 2D plane.
    Returns total solar return at each step.
    """
    yaws = np.linspace(0, 2*np.pi, steps)
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
    yaws_deg, returns = yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir)
    plt.figure(figsize=(8,4))
    plt.plot(yaws_deg, returns, '-o')
    plt.xlabel('Yaw angle (deg)')
    plt.ylabel('Reflected intensity')
    plt.title('Cuboid reflected solar return vs yaw angle')
    plt.grid(True)
    plt.show()


# -----------------------------
# Example usage for check
# -----------------------------
if __name__ == "__main__":
    # dimensions and orientation
    length, width, height = 0.01,0.01,0.01
    roll, pitch = np.deg2rad(0), np.deg2rad(0)  # keep fixed for 2D sweep
    face_materials = ["Lambertian 20%"]*6

    # constant illumination and observation
    obs_dir = np.array([0,1, 0])
    illum_dir = np.array([0, 1, 0])  # sun ~30° elevation

    plot_yaw_sweep(length, width, height, roll, pitch, face_materials, illum_dir, obs_dir)