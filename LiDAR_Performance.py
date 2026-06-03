import numpy as np
import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import CuboidSolarModel as solar
from LiDAR import SNR_BASELINE

# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-f')
solar_tracker = []

#Default Parameters - many of these are overwritten as inputs but these values serve as fall backs
aperture_radius = 0.015
aperture_area_m2 = np.pi * (aperture_radius ** 2)  # receiver aperture area (mm^2)
#aperture_area_m2 = np.pi * (0.03 ** 2)/4000000  # receiver aperture area (mm^2)
pulse_width_s =10e-9 #0.016 # # laser pulse duration (s)
wavelength_nm = 1550  # laser wavelength (nm)

PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458
bandwidth = 10
polarity_filter = 0.1

min_photons_to_detect =20

min_SNR = 0.3


#Parameters defining the targets geometry and materials
length1=0.1
width1=0.1
height1=0.1
roll1=45
pitch1=45
yaw1=45
face_materials1 = ["Lambertian 20%","Lambertian 20%","Lambertian 20%","Lambertian 20%","Lambertian 20%","Lambertian 20%"]

range_m=10000
pulse_energy = 0.0009
pulse_energy_J = pulse_energy
obs_dir = np.array([0, 0, 1])
illum_dir = np.array([0, 0,1])
theta_user1 =0.0003#np.radians(50)
#theta_user1 =0.007
beam_waist = 0.001

#for basilisk
# Compute SNR
# if snr> snr threshold
#     return range_m,snr,solarphotons,photons



def compute_photons_p_pulse(X=0, Y=0, z=10000, w0=0.001, wavelength=wavelength_nm, P_total=0.001, theta_user=0,length=length1, width=width1, height=height1, roll=roll1, pitch=pitch1, yaw=yaw1, face_materials = face_materials1):

    I, w = Gbeam.gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)

    R, target_area_m2 = target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)

    reflected_energy = I * R


    characteristic_width = (length+width+height)/3


    # received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, z, characteristic_width, wavelength,
    #                                                                    reflected_energy, 0.0125)
    #received_energy = received_intensity * aperture_area_m2

    solid_angle = aperture_area_m2 / range_m ** 2

    received_energy = R * solid_angle *I


    photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)


    photons_per_pulse = received_energy / photon_energy_J

    return photons_per_pulse


def compute_solar_photons(range_m, length=length1, width=width1, height=height1, roll=roll1, pitch=pitch1, yaw=yaw1, face_materials = face_materials1, illum_dir=[0, 0,1], obs_dir=[0, 0, 1]):
    #computes the background noise from solar photons
    R, contributions = solar.solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir)

    solar_reflected_W  = 0.4 * bandwidth * polarity_filter * R

    characteristic_width = (length+width+height)/3
    #solar_received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, range_m, characteristic_width, wavelength_nm,solar_reflected_W, 0)
    #solar_received_intensity = R*solar_reflected_W
    #print(return_spot_radius)

    solid_angle = aperture_area_m2/range_m**2

    solar_received_intensity = R*solid_angle*600  #this needs sorting out to account for area...maybe
    solar_received_intensity = 0.4 * bandwidth * polarity_filter * R *solid_angle
    #if aperture_radius<return_spot_radius:
    solar_collected_W  = solar_received_intensity

    #else:
       # solar_collected_W = solar_received_intensity * np.pi*(return_spot_radius**2)

    photon_energy_J    = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)

    solar_photons = (solar_collected_W / photon_energy_J) * pulse_width_s
    #print(solar_photons)
    view_dir = obs_dir
    angle_zx, angle_zy, angle_xy = relative_angles(view_dir, illum_dir)
    FoV = np.pi - max(theta_user1/2,np.radians(5)) # models the FoV at which the sun is considered to "glare". A minimum of 5 degrees is modelled
    solar_disk_radius = np.pi - np.radians(0.265) # projects a cone based on the radius of the sun

    direct_solar_photons = 0
    #Handles direct glare -
    if abs(angle_zx) > solar_disk_radius or abs(angle_zy) > solar_disk_radius:
        #direct glare, when the sun falls on the view axis
        #note this has a dependency on fov too which is not accurately modelled
        direct_solar_photons = (0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons =solar_photons+ direct_solar_photons
        #print("direct glare")
    elif abs(angle_zx) > FoV or abs(angle_zy) > FoV :
        #print("sun glare at" , direct_solar_photons)
        #models partial off axis glare, where we model a linear fall off of incident light as the sun moves across the FoV from boresight
        x = (np.radians(0.265))/(np.pi - abs(angle_zx))
        direct_solar_photons = x*(0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons = solar_photons + direct_solar_photons
        #print("zx glare",x)



    return solar_photons, direct_solar_photons

def compute_SNR(range_m,pulse_energy_J,roll, pitch, yaw,illum_dir,X,Y,theta=theta_user1):
    #print("compute SNR")
    reflected_solar_photons_per_pulse,direct_solar_photons = compute_solar_photons(range_m, length1, width1, height1, roll, pitch, yaw, face_materials1, illum_dir, [0, 0, 1])
    photons_per_pulse = compute_photons_p_pulse(X, Y, range_m, beam_waist, wavelength_nm, pulse_energy_J, theta,length1, width1, height1, roll, pitch, yaw,  face_materials1)

    #print(photons_per_pulse)
    SNR = (photons_per_pulse / reflected_solar_photons_per_pulse)
    #SNR = reflected_solar_photons_per_pulse /(direct_solar_photons+100)
    #print(reflected_solar_photons_per_pulse)
    #use this if you want to include untracked detections
   # if (reflected_solar_photons_per_pulse-direct_solar_photons)<min_photons_to_detect:
        #print("below threshold")
        #SNR = 0

     #or this to show only valid tracks
    if photons_per_pulse+(reflected_solar_photons_per_pulse-direct_solar_photons)<min_photons_to_detect:
        #print("below threshold")
        SNR = 0

    solar_tracker.append({"solar photons" : reflected_solar_photons_per_pulse-direct_solar_photons})
    return SNR

def compute_range(pulse_energy_J,theta=theta_user1):
    #computes maximum range for given setup
    range = 1000
    SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0,theta)

    dr = 100
    while SNR>min_SNR:
        range = range + dr
        SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0,theta)

    return range


def compute_pulse_energy(range,theta=theta_user1):
    #computes pulse energy required to achieve a given range
    pulse_energy_J = 10000000000000
    azimuth = np.pi*3/2
    elevation = 0

    illum_dir = np.array([
        0,
        0,
        1
    ])
    #print(illum_dir)
    SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,illum_dir,0,0,theta)

    while SNR>min_SNR:
        dE = pulse_energy_J*0.005
        pulse_energy_J = pulse_energy_J - dE
        SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,illum_dir,0,0,theta)
       # print(SNR)
       # print(pulse_energy_J)
    return pulse_energy_J


def relative_angles(view_dir, illum_dir):
    #calculates the angle between the view direction vector and solar illumination vector, centred on the sensor, and projected on to 2D planes
    #Used to normalise graph and plot 3D random vectors on a single polar plot, normalised to the suns direction
    # angle in ZX plane (elevation-like)
    angle_zx = np.arctan2(illum_dir[0], illum_dir[2]) - np.arctan2(view_dir[0], view_dir[2])
    # angle in ZY plane (azimuth-like)
    angle_zy = np.arctan2(illum_dir[1], illum_dir[2]) - np.arctan2(view_dir[1], view_dir[2])
    # angle in XY plane
    angle_xy = np.arctan2(illum_dir[1], illum_dir[0]) - np.arctan2(view_dir[1], view_dir[0])
    return angle_zx, angle_zy, angle_xy



def monte_carlo_SNR(max_range, pulse_energy_J, n_samples=1000):
    results = []
    n = 0
    print(n)

    for _ in range(n_samples):
        n += 1
        roll = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)

        yaw = np.random.uniform(-np.pi, np.pi)


        range_m = np.random.uniform(0, max_range)
        avg_pointing_accuracy_cartesian = (range_m / 2) * np.tan(theta_user1 / 2)

        X, Y = np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian),np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian)

        azimuth =  np.random.uniform(-np.pi, np.pi)
        elevation = np.arcsin(np.random.uniform(-1, 1))

        illum_dir = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])

        SNR = compute_SNR(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y)

        view_dir = [0,0,1]
        angle_zx, angle_zy, angle_xy = relative_angles(view_dir, illum_dir)

        #if SNR == 0:
           # range_m = 0
        results.append({
            "range_m": range_m,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "angle_zx": angle_zx,
            "angle_zy": angle_zy,
            "angle_xy": angle_xy,
            "pulse_energy": pulse_energy_J,
            "SNR": SNR
        })

        percentage = 100 * n / n_samples
        if percentage % 1 == 0:
            print(f"Progress: {int(percentage)}%")

    return results

if __name__ == "__main__":
    #print(compute_pulse_energy(29000))
    import matplotlib.pyplot as plt
    plt.close('all')

    from scipy.ndimage import uniform_filter1d


    def plot_solar_tracker():
        if len(solar_tracker) != len(results):
            print(f"solar_tracker/results length mismatch: {len(solar_tracker)} vs {len(results)}")
            return

        range_bin_size_m = 1000
        max_r = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))
        range_edges = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)
        range_centres = (range_edges[:-1] + range_edges[1:]) / 2

        binned = [[] for _ in range(n_range_bins)]
        for entry, r in zip(solar_tracker, results):
            r_idx = int(np.clip(r["range_m"] // range_bin_size_m, 0, n_range_bins - 1))
            val = entry["solar photons"]
            if val > 0:
                binned[r_idx].append(val)

        percentiles = {p: np.full(n_range_bins, np.nan) for p in [5, 25, 50, 75, 95]}
        for i, vals in enumerate(binned):
            if len(vals) >= 5:
                for p in percentiles:
                    percentiles[p][i] = np.percentile(vals, p)

        valid = ~np.isnan(percentiles[50])
        rc = range_centres[valid]
        plt.rcParams.update({'font.size': 15})

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(rc, percentiles[5][valid], percentiles[95][valid],
                        alpha=0.15, color='steelblue', label='5–95%')
        ax.fill_between(rc, percentiles[25][valid], percentiles[75][valid],
                        alpha=0.35, color='steelblue', label='25–75%')
        ax.plot(rc, percentiles[50][valid], color='steelblue', linewidth=2, label='Median')

        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Solar photons per pulse')
        ax.set_title('Solar background photons vs range')
        ax.set_yscale('log')  # remove if linear scale preferred - likely spans orders of magnitude

        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.axhline(photon_detect_lim, color='red', linestyle='--', linewidth=1.5, label='Detection limit')
        ax.legend()

        plt.tight_layout()
    def plotter():
        import random
        print(f"plotter called with {len(results)} results")
        sample = random.sample(results, min(50000, len(results)))
        rand_above = [r for r in sample if r["SNR"] > min_SNR]
        rand_below = [r for r in sample if r["SNR"] <= min_SNR]

        n_angle_bins = 360
        range_bin_size_m = 1000

        max_r = max(r["range_m"] for r in results)
        n_range_bins = int(np.ceil(max_r / range_bin_size_m))

        angle_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angle_centres = (angle_edges[:-1] + angle_edges[1:]) / 2
        range_edges = np.arange(0, (n_range_bins + 1) * range_bin_size_m, range_bin_size_m)
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

        colours = {0.90: "#1a9641", 0.75: "#a6d96a", 0.50: "#fdae61", 0.25: "#d7191c", 0.10: "#b2182b", 0.05: "#762a83",
                   0.01: "#2d004b"}

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
            out = np.full(len(arr), np.nan)
            valid = ~np.isnan(arr)
            if valid.sum() > 0:
                out[valid] = uniform_filter1d(arr[valid], size=15)
            return out

        #colours = {0.90: "green", 0.75: "blue", 0.50: "orange", 0.25: "red"}
        #colours = {0.90: "#1a9641", 0.75: "#a6d96a", 0.50: "#fdae61", 0.25: "#d7191c"}
        above = [r for r in results if r["SNR"] > min_SNR]
        below = [r for r in results if r["SNR"] <= min_SNR]

        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))

        rand_zero = [r for r in sample if r["SNR"] == 0]
        rand_below = [r for r in sample if 0 < r["SNR"] <= min_SNR]

        ax1.scatter([r["angle_zy"] for r in rand_zero], [r["range_m"] for r in rand_zero],
                    c="black", alpha=0.3, s=1, label="SNR = 0")
        ax1.scatter([r["angle_zy"] for r in rand_below], [r["range_m"] for r in rand_below],
                    c="red", alpha=0.3, s=1, label=f"0 < SNR <= {min_SNR}")
        ax1.scatter([r["angle_zy"] for r in rand_above], [r["range_m"] for r in rand_above],
                    c="blue", s=1, label=f"SNR > {min_SNR}")
        ax1.set_title("Range vs Bearing (SNR)")
        ax1.legend(loc="upper right")
        ax1.set_rlim(0, max_r)
        ax2.set_rlim(0, max_r)
        for t in reversed(thresholds):
            smoothed = smooth(threshold_ranges[t])
            ax2.plot(angle_centres, smoothed, linewidth=2, c=colours[t], label=f"{int(t * 100)}% detection")
        ax2.set_title("Detection Range vs Bearing")
        ax2.legend()
    from scipy.special import expit

    # --- Run simulation ---
    max_range = 10000#compute_range(pulse_energy_J)
    #pulse_energy_J = 0.0025
    results = monte_carlo_SNR(max_range, pulse_energy_J, 30000)

    detect_SNR = 1
    dark_current = 25
    QE=1
    optical_efficiency = 1

    photon_detect_lim = detect_SNR*dark_current/(QE*optical_efficiency)
    plotter()
    plot_solar_tracker()


    plt.tight_layout()
    plt.show()

