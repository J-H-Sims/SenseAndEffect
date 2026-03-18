import numpy as np
import CuboidLiDARModel as target
import GaussianBeam as Gbeam
import CuboidSolarModel as solar

     # target range (m)
#divergence_rad = 0.018  # transmit beam divergence (mrad)


#target_area_m2 = 1  # area of target component (mm^2)
#target_reflectance = 0.1  # reflectance of surface (0–1)
#scatter_scalar = 1  # factor that increases return divergence

aperture_area_m2 = np.pi * (0.015 ** 2)  # receiver aperture area (mm^2)

pulse_width_s = 5e-9  # laser pulse duration (s)
wavelength_nm = 1470  # laser wavelength (nm)

PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458
bandwidth = 10
polarity_filter = 1

min_photons_to_detect =20

min_SNR = 0.3

#def set_divergence_degrees(divergence_in_degrees):
#    global divergence_rad
#    divergence_rad = divergence_in_degrees *np.pi/180
#    print(divergence_rad)
#    return divergence_rad
# def diffraction_limited_divergence_deg(M2, wavelength, waist):
#     return np.degrees(M2*2*wavelength/(np.pi*(waist/2)))

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
    #print(f"I: {I}, w: {w}")

    R, target_area_m2 = target.lidar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials)
    #print(f"R: {R}, target_area_m2: {target_area_m2}")

    reflected_energy = I * R
    #print(f"reflected_energy: {reflected_energy}")

    characteristic_width = (length+width+height)/3
    #print(f"characteristic_width: {characteristic_width}")

    received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, z, characteristic_width, wavelength,
                                                                       reflected_energy, 0.0125)
    #print(f"received_intensity: {received_intensity}, return_spot_radius: {return_spot_radius}")

    received_energy = received_intensity * aperture_area_m2
    #print(f"received_energy: {received_energy}")

    photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)
    #print(f"photon_energy_J: {photon_energy_J}")

    photons_per_pulse = received_energy / photon_energy_J
    #print(f"photons_per_pulse: {photons_per_pulse}")
    return photons_per_pulse


def compute_solar_photons(range_m, length=length1, width=width1, height=height1, roll=roll1, pitch=pitch1, yaw=yaw1, face_materials = face_materials1, illum_dir=[0, 0,1], obs_dir=[0, 0, 1]):
    R, contributions = solar.solar_return_cuboid(length, width, height, roll, pitch, yaw, face_materials, illum_dir, obs_dir)



    solar_reflected_W  = 0.4 * bandwidth * polarity_filter * R
    #solar_spot_m       = (range_m * np.tan(0.05)) + target_area_m2**0.5
    #solar_divergence_radian =
    #solar_divergence_steradian = 2 * np.pi * (1 - np.cos(solar_divergence_radian / 2))
    #solar_spot_area = solar_divergence_steradian * range_m ** 2 + target_area_m2  # includes approximation to divergence and object area
    characteristic_width = (length+width+height)/3
    solar_received_intensity, return_spot_radius = Gbeam.gaussian_beam_wm2(0, 0, range_m, characteristic_width, wavelength_nm,
                                                                       solar_reflected_W, 0)


    solar_collected_W  = solar_received_intensity * aperture_area_m2
    photon_energy_J    = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)
    #print(solar_collected_W)
    solar_photons = (solar_collected_W / photon_energy_J) * pulse_width_s

    view_dir = obs_dir
    angle_zx, angle_zy, angle_xy = relative_angles(view_dir, illum_dir)
    FoV = np.pi - np.radians(10)
    solar_disk_radius = np.pi - np.radians(0.265)
    #print("tick")
    if abs(angle_zx) > solar_disk_radius or abs(angle_zy) > solar_disk_radius:

        direct_solar_photons = (0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons =solar_photons+ direct_solar_photons
        print("direct glare")
    elif abs(angle_zx) > FoV :
        #print("sun glare at" , direct_solar_photons)
        x = (np.radians(0.265))/(np.pi -abs( angle_zx))
        direct_solar_photons = (x)*(0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons = solar_photons + direct_solar_photons
        print("zx glare",x)
    elif abs(angle_zy) > FoV:
        x = (np.radians(0.265)) / (np.pi - abs(angle_zy))
        #print("sun glare at" , direct_solar_photons)
        direct_solar_photons = x*(0.4 * bandwidth * polarity_filter * aperture_area_m2 / photon_energy_J) * pulse_width_s
        solar_photons = solar_photons + direct_solar_photons
        #print("zy glare",(np.pi -solar_disk_radius)/abs(np.pi - angle_zy))

    # print(f"R: {R}")
    # #print(f"target_area_m2: {target_area_m2}")
    # print(f"solar_reflected_W: {solar_reflected_W}")
    # print(f"characteristic_width: {characteristic_width}")
    # print(f"solar_received_intensity: {solar_received_intensity}")
    # print(f"return_spot_radius: {return_spot_radius}")
    # print(f"solar_collected_W: {solar_collected_W}")
    # print(f"photon_energy_J: {photon_energy_J}")
    return solar_photons

def compute_SNR(range_m,pulse_energy_J,roll, pitch, yaw,illum_dir,X,Y):
    #print("compute SNR")
    reflected_solar_photons_per_pulse = compute_solar_photons(range_m, length1, width1, height1, roll, pitch, yaw, face_materials1, illum_dir, [0, 0, 1])
    photons_per_pulse = compute_photons_p_pulse(X, Y, range_m, beam_waist, wavelength_nm, pulse_energy_J, theta_user1,length1, width1, height1, roll, pitch, yaw,  face_materials1)


    SNR = (photons_per_pulse / reflected_solar_photons_per_pulse)


    if photons_per_pulse<min_photons_to_detect:
        print("below threshold")
        SNR = 0

    #print(f"SNR:           {SNR* 100:.2f} %")
    #print(f"ppp:           {photons_per_pulse:.5f}")

    return SNR

def compute_range(pulse_energy_J):
    range = 1000
    SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0)

    dr = 100
    while SNR>min_SNR:
        range = range + dr
        SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0)

    return range

#print(compute_range(39e-6))

def compute_pulse_energy(range):
    pulse_energy_J = 10000000000000
    #print("SNR")
    SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0)
    #print(SNR)
    #print(pulse_energy_J)
    while SNR>min_SNR:
        dE = pulse_energy_J*0.005
        pulse_energy_J = pulse_energy_J - dE
        SNR = compute_SNR(range,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0)
       # print(SNR)
       # print(pulse_energy_J)
    return pulse_energy_J

#print(compute_pulse_energy(800000, np.radians(1)))

#print(compute_range(900e-6, 7e-3))


#print(compute_photons_p_pulse(z=range, P_total=pulse_energy,theta_user =theta_user))
#print(compute_solar_photons(range, length1, width1, height1, roll1, pitch1, yaw1, face_materials1, illum_dir, obs_dir))
#print(compute_SNR(range_m,pulse_energy_J,roll1,pitch1,yaw1,[0,0,1],0,0)*100)

#print("range " , compute_range(pulse_energy))
#print("pulse energy  " , compute_pulse_energy(10000))
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

    I, w = Gbeam.gaussian_beam_wm2(0, 0, max_range, beam_waist, wavelength_nm, 1, theta_user1)


    for _ in range(n_samples):
        roll = np.random.uniform(-np.pi, np.pi)
        pitch = np.random.uniform(-np.pi, np.pi)

        yaw =0# np.random.uniform(-np.pi, np.pi)


        range_m = np.random.uniform(0, max_range)
        avg_pointing_accuracy_cartesian = (range_m / 2) * np.tan(theta_user1 / 2)

        X, Y = np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian),np.random.uniform(-avg_pointing_accuracy_cartesian, avg_pointing_accuracy_cartesian)
        #X,Y = 0,0
        #azimuth = np.random.uniform(-np.pi , np.pi)
        #elevation = 0 #np.random.uniform(-np.pi / 2, np.pi / 2)

        illum_dir = np.array([
            np.random.uniform(-1 , 1),
            np.random.uniform(-1 , 1),
            np.random.uniform(-1 , 1)
        ])


        SNR = compute_SNR(range_m, pulse_energy_J, roll, pitch, yaw, illum_dir, X, Y)

        view_dir = [0,0,1]
        angle_zx, angle_zy, angle_xy = relative_angles(view_dir, illum_dir)

        if SNR == 0:
            range_m = 0
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

    return results

import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import expit

# --- Run simulation ---
max_range = compute_range(pulse_energy_J)
results = monte_carlo_SNR(2 * max_range, pulse_energy_J, 100000)

above = [r for r in results if r["SNR"] >  0.3]
below = [r for r in results if r["SNR"] <= 0.3]

# --- Binning setup ---
n_bins = 360
bin_edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

bin_ranges_above = np.full(n_bins, np.nan)
bin_ranges_below = np.full(n_bins, np.nan)
bin_range_90     = np.full(n_bins, np.nan)

def sigmoid(r, r50, k):
    return expit(-k * (r - r50))

# --- Per-bin statistics ---
for i in range(n_bins):
    in_bin = lambda dataset: [r["range_m"] for r in dataset
                               if bin_edges[i] <= r["angle_zy"] < bin_edges[i + 1]]

    above_ranges = in_bin(above)
    below_ranges = in_bin(below)

    if above_ranges:
        bin_ranges_above[i] = np.mean(above_ranges)
    if below_ranges:
        bin_ranges_below[i] = np.mean(below_ranges)

    # sigmoid fit for 90% detection range
    bin_data = ([(r, 1) for r in above_ranges] +
                [(r, 0) for r in below_ranges])

    if len(bin_data) < 10:
        continue

    bin_data.sort(key=lambda x: x[0])
    ranges     = np.array([d[0] for d in bin_data])
    detections = np.array([d[1] for d in bin_data])

    if detections.sum() == 0 or detections.sum() == len(detections):
        continue

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(sigmoid, ranges, detections,
                                p0=[np.median(ranges), 0.001],
                                maxfev=5000)
        r50, k = popt
        if k > 0:
            r90 = r50 - np.log(0.5 / 0.1) / k
            if r90 > 0:
                bin_range_90[i] = r90
    except RuntimeError:
        continue

# --- Smoothing ---
def smooth(arr):
    out = np.full(n_bins, np.nan)
    valid = ~np.isnan(arr)
    out[valid] = uniform_filter1d(arr[valid], size=15)
    return out

smoothed_above = smooth(bin_ranges_above)
smoothed_below = smooth(bin_ranges_below)
smoothed_90    = smooth(bin_range_90)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6))

# Plot 1: raw SNR scatter
ax1.scatter([r["angle_zy"] for r in below], [r["range_m"] for r in below],
            c="red",  s=5, label="SNR <= 0.3")
ax1.scatter([r["angle_zy"] for r in above], [r["range_m"] for r in above],
            c="blue", s=5, label="SNR > 0.3")
ax1.set_title("Range vs Bearing (SNR)")
ax1.legend(loc="upper right")

# Plot 2: binned means + 90% detection range
ax2.scatter(bin_centres, bin_ranges_above, s=5, alpha=0.3, c="blue", label="above (mean)")
ax2.scatter(bin_centres, bin_ranges_below, s=5, alpha=0.3, c="red",  label="below (mean)")
ax2.plot(bin_centres, smoothed_above, linewidth=1.5, c="blue")
ax2.plot(bin_centres, smoothed_below, linewidth=1.5, c="red")
ax2.plot(bin_centres, smoothed_90,    linewidth=2.5, c="green", label="90% detection range")
ax2.set_title("Detection Range vs Bearing")
ax2.legend()

plt.tight_layout()
plt.show()