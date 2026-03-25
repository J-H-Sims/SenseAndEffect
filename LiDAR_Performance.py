import numpy as np
import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import CuboidSolarModel as solar
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-f')

aperture_area_m2 = np.pi * (0.015 ** 2)  # receiver aperture area (mm^2)

pulse_width_s = 5e-9  # laser pulse duration (s)
wavelength_nm = 1470  # laser wavelength (nm)

PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458
bandwidth = 10
polarity_filter = 1

min_photons_to_detect =20

min_SNR = 0.3


length1=1
width1=1
height1=1
roll1=45
pitch1=45
yaw1=45
face_materials1 = ["Brushed V Al","Brushed V Al","Brushed V Al","Brushed V Al","Brushed V Al","Brushed V Al"]

range_m=10000
pulse_energy = 0.001
pulse_energy_J = pulse_energy
obs_dir = np.array([0, 0, 1])
illum_dir = np.array([0, 0,1])
theta_user1 =np.radians(0.1)
#theta_user1 =0.007
beam_waist = 0.001

def compute_photons_p_pulse(X=0, Y=0, z=10000, w0=0.001, wavelength=wavelength_nm, P_total=0.001, theta_user=0,length=length1, width=width1, height=height1, roll=roll1, pitch=pitch1, yaw=yaw1, face_materials = face_materials1):
    I, w = Gbeam.gaussian_beam_wm2(X, Y, z, w0, wavelength, P_total, theta_user)

    R, target_area_m2 = target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)

    reflected_energy = I * R


    characteristic_width = (length+width+height)/3


    received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, z, characteristic_width, wavelength,
                                                                       reflected_energy, 0.0125)


    received_energy = received_intensity * aperture_area_m2


    photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)


    photons_per_pulse = received_energy / photon_energy_J

    return photons_per_pulse


def compute_solar_photons(range_m, length=length1, width=width1, height=height1, roll=roll1, pitch=pitch1, yaw=yaw1, face_materials = face_materials1, illum_dir=[0, 0,1], obs_dir=[0, 0, 1]):
    R, contributions = solar.solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir)



    solar_reflected_W  = 0.4 * bandwidth * polarity_filter * R

    characteristic_width = (length+width+height)/3
    solar_received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, range_m, characteristic_width, wavelength_nm,
                                                                       solar_reflected_W, 0)


    solar_collected_W  = solar_received_intensity * aperture_area_m2
    photon_energy_J    = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)

    solar_photons = (solar_collected_W / photon_energy_J) * pulse_width_s

    view_dir = obs_dir
    angle_zx, angle_zy, angle_xy = relative_angles(view_dir, illum_dir)
    FoV = np.pi - np.radians(max(theta_user1,10))
    solar_disk_radius = np.pi - np.radians(0.265)

    if abs(angle_zx) > solar_disk_radius or abs(angle_zy) > solar_disk_radius:

        direct_solar_photons = (0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons =solar_photons+ direct_solar_photons
        #print("direct glare")
    elif abs(angle_zx) > FoV :
        #print("sun glare at" , direct_solar_photons)
        x = (np.radians(0.265))/(np.pi -abs( angle_zx))
        direct_solar_photons = (x)*(0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons = solar_photons + direct_solar_photons
        #print("zx glare",x)
    elif abs(angle_zy) > FoV:
        x = (np.radians(0.265)) / (np.pi - abs(angle_zy))
        #print("sun glare at" , direct_solar_photons)
        direct_solar_photons = x*(0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons = solar_photons + direct_solar_photons
        #print("zy glare",(np.pi -solar_disk_radius)/abs(np.pi - angle_zy))


    return solar_photons

def compute_SNR(range_m,pulse_energy_J,roll, pitch, yaw,illum_dir,X,Y,theta=theta_user1):
    #print("compute SNR")
    reflected_solar_photons_per_pulse = compute_solar_photons(range_m, length1, width1, height1, roll, pitch, yaw, face_materials1, illum_dir, [0, 0, 1])
    photons_per_pulse = compute_photons_p_pulse(X, Y, range_m, beam_waist, wavelength_nm, pulse_energy_J, theta,length1, width1, height1, roll, pitch, yaw,  face_materials1)


    SNR = (photons_per_pulse / reflected_solar_photons_per_pulse)


    if photons_per_pulse<min_photons_to_detect:
        #print("below threshold")
        SNR = 0



    return SNR

def compute_range(pulse_energy_J,theta=theta_user1):
    range = 1000
    SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0,theta)

    dr = 100
    while SNR>min_SNR:
        range = range + dr
        SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0,theta)

    return range



def compute_pulse_energy(range,theta=theta_user1):
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
    print(compute_pulse_energy(29000))
    import matplotlib.pyplot as plt
    plt.close('all')

    from scipy.ndimage import uniform_filter1d

    from scipy.special import expit

    # --- Run simulation ---
    # max_range = 5000#compute_range(pulse_energy_J)
    #
    # results = monte_carlo_SNR(2 * max_range, pulse_energy_J, 100000)
    #
    # def plotter():
    #     print(f"plotter called with {len(results)} results")
    #     above = [r for r in results if r["SNR"] >  0.3]
    #     below = [r for r in results if r["SNR"] <= 0.3]
    #
    #     # --- Binning setup ---
    #     n_bins = 360
    #     bin_edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    #     bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    #     bin_ranges_above = np.full(n_bins, np.nan)
    #     bin_ranges_below = np.full(n_bins, np.nan)
    #     bin_range_90     = np.full(n_bins, np.nan)
    #
    #     # def sigmoid(r, r50, k):
    #     #     return expit(-k * (r - r50))
    #
    #     # --- Per-bin statistics ---
    #     for i in range(n_bins):
    #         in_bin = lambda dataset: [r["range_m"] for r in dataset
    #                                    if bin_edges[i] <= r["angle_zy"] < bin_edges[i + 1]]
    #
    #         above_ranges = in_bin(above)
    #         below_ranges = in_bin(below)
    #
    #         if above_ranges:
    #             bin_ranges_above[i] = np.mean(above_ranges)
    #         if below_ranges:
    #             bin_ranges_below[i] = np.mean(below_ranges)
    #
    #         # sigmoid fit for 90% detection range
    #         bin_data = ([(r, 1) for r in above_ranges] +
    #                     [(r, 0) for r in below_ranges])
    #
    #         if len(bin_data) < 10:
    #             continue
    #
    #         bin_data.sort(key=lambda x: x[0])
    #         ranges = np.array([d[0] for d in bin_data])
    #         detections = np.array([d[1] for d in bin_data])
    #
    #         bin_range_90[i] = ranges[-1]  # default if threshold always met
    #
    #         for j in range(len(bin_data)):
    #             r, detection = bin_data[j]
    #
    #             # look back over previous 100 candidates (or fewer if near the start)
    #             window_start = max(0, j - 99)
    #             window = detections[window_start: j + 1]
    #
    #             if len(window) < 10:  # skip until window is meaningful
    #                 continue
    #
    #             if window.mean() < 0.90:
    #                 bin_range_90[i] = r
    #                 break
    #
    #
    #     # --- Smoothing ---
    #     def smooth(arr):
    #         out = np.full(n_bins, np.nan)
    #         valid = ~np.isnan(arr)
    #         out[valid] = uniform_filter1d(arr[valid], size=5)
    #         return out
    #
    #     smoothed_above = smooth(bin_ranges_above)
    #     smoothed_below = smooth(bin_ranges_below)
    #     smoothed_90    = smooth(bin_range_90)
    #
    #     # --- Plotting ---
    #     fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))
    #
    #     # Plot 1: raw SNR scatter
    #     ax1.scatter([r["angle_zy"] for r in above], [r["range_m"] for r in above],
    #                 c="blue", s=0.1, label="SNR > 0.3")
    #     ax1.scatter([r["angle_zy"] for r in below], [r["range_m"] for r in below],
    #                 c="red",  s=0.1, label="SNR <= 0.3")
    #
    #     ax1.set_title("Range vs Bearing (SNR)")
    #     ax1.legend(loc="upper right")
    #
    #     # Plot 2: binned means + 90% detection range
    #     ax2.scatter(bin_centres, bin_ranges_above, s=5, alpha=0.3, c="blue", label="above (mean)")
    #     ax2.scatter(bin_centres, bin_ranges_below, s=5, alpha=0.3, c="red",  label="below (mean)")
    #     ax2.plot(bin_centres, smoothed_above, linewidth=1.5, c="blue")
    #     ax2.plot(bin_centres, smoothed_below, linewidth=1.5, c="red")
    #     ax2.plot(bin_centres, smoothed_90,    linewidth=2.5, c="green", label="90% detection range")
    #     ax2.set_title("Detection Range vs Bearing")
    #     ax2.legend()
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
        for t in thresholds:
            smoothed = smooth(threshold_ranges[t])
            ax2.plot(angle_centres, smoothed, linewidth=2, c=colours[t], label=f"{int(t * 100)}% detection")

        ax2.set_title("Detection Range vs Bearing")
        ax2.legend()
    from scipy.special import expit

    # --- Run simulation ---
    max_range = 50000#compute_range(pulse_energy_J)
    pulse_energy_J = 0.0025
    results = monte_carlo_SNR(max_range, pulse_energy_J, 50000)

    plotter()


    plt.tight_layout()
    plt.show()