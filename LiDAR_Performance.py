import numpy as np

                      # target range (m)
#divergence_rad = 0.018  # transmit beam divergence (mrad)


target_area_m2 = 1  # area of target component (mm^2)
target_reflectance = 0.1  # reflectance of surface (0–1)
scatter_scalar = 1  # factor that increases return divergence

aperture_area_m2 = np.pi * (0.015 ** 2)  # receiver aperture area (mm^2)

pulse_width_s = 5e-9  # laser pulse duration (s)
wavelength_nm = 1470  # laser wavelength (nm)

PLANCK_CONSTANT = 6.62607015e-34
SPEED_OF_LIGHT = 299792458
bandwidth = 10
polarity_filter = 1

min_photons_to_detect =1

min_SNR = 0.3

#def set_divergence_degrees(divergence_in_degrees):
#    global divergence_rad
#    divergence_rad = divergence_in_degrees *np.pi/180
#    print(divergence_rad)
#    return divergence_rad


def compute_photons_p_pulse(range_m, pulse_energy_J,divergence_rad):

    emitted_power_W = pulse_energy_J/pulse_width_s

    divergence_steradian = 2 * np.pi * (1 - np.cos(divergence_rad/2))
    #spot_size_m =range_m * np.tan(divergence_rad)
    spot_surface_area =  divergence_steradian * range_m**2

    laser_intensity_W_per_m2 =emitted_power_W / spot_surface_area
    incident_energy_W =laser_intensity_W_per_m2 * target_area_m2
    reflected_energy_W =target_reflectance * incident_energy_W
    rough_characteristic_dimension_m =(target_area_m2)**0.5
    return_divergence_rad =scatter_scalar * divergence_rad

    return_divergence_steradian = 2 * np.pi * (1 - np.cos(return_divergence_rad/2))
    return_spot_surface_area =  return_divergence_steradian * range_m**2 + (np.pi*(rough_characteristic_dimension_m/2)**2) #includes approximation to divergence and object area

    #return_spot_diameter_m =((range_m * np.tan(return_divergence_rad))+ (rough_characteristic_dimension_m))
    intensity_of_return_W_per_m2 = reflected_energy_W / (np.pi*(return_spot_surface_area/2)**2)
    collected_power_W = intensity_of_return_W_per_m2 * aperture_area_m2
    photon_energy_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)
    photons_per_second = (collected_power_W) / photon_energy_J
    photons_per_pulse = photons_per_second * pulse_width_s
    #print(photons_per_pulse)
    #print(return_spot_diameter_m)
    return photons_per_pulse

def compute_solar_photons(range_m):


    solar_reflected_W  = 0.4 * bandwidth * polarity_filter * target_area_m2 * target_reflectance
    solar_spot_m       = (range_m * np.tan(0.05)) + target_area_m2**0.5
    solar_divergence_radian = 0.05
    solar_divergence_steradian = 2 * np.pi * (1 - np.cos(solar_divergence_radian / 2))
    solar_spot_area = solar_divergence_steradian * range_m ** 2 + target_area_m2  # includes approximation to divergence and object area


    solar_collected_W  = (solar_reflected_W / solar_spot_area) * aperture_area_m2
    photon_energy_J    = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (wavelength_nm * 1e-9)
    #print(solar_collected_W)
    return (solar_collected_W / photon_energy_J) * pulse_width_s

def compute_SNR(range_m,pulse_energy_J,divergence_rad):
    #print("compute SNR")
    solar_photons_per_pulse = compute_solar_photons(range_m)
    photons_per_pulse = compute_photons_p_pulse(range_m, pulse_energy_J,divergence_rad)


    SNR = (photons_per_pulse / solar_photons_per_pulse)

    if photons_per_pulse<min_photons_to_detect:
        print("below threshold")
        SNR = 0

    #print(f"SNR:           {SNR* 100:.2f} %")
    #print(f"ppp:           {photons_per_pulse:.5f}")
    return SNR

# def compute_range(pulse_energy_J,divergence_rad):
#     range = 1000
#     SNR = compute_SNR(range,pulse_energy_J,divergence_rad)
#
#     dr = 100
#     while SNR>min_SNR:
#         range = range + dr
#         SNR = compute_SNR(range, pulse_energy_J,divergence_rad)
#     return range

#print(compute_range(39e-6))

def compute_pulse_energy(range,divergence_rad):
    pulse_energy_J = 10000000000000
    #print("SNR")
    SNR = compute_SNR(range,pulse_energy_J,divergence_rad)
    #print(SNR)
    while SNR>min_SNR:
        dE = pulse_energy_J*0.005
        pulse_energy_J = pulse_energy_J - dE
        SNR = compute_SNR(range, pulse_energy_J,divergence_rad)
    return pulse_energy_J

#compute_pulse_energy(10000, np.radians(45))



# import matplotlib.pyplot as plt
# # Inputs
# Uncertainty_Ellipse_Radius = 5000
# Sensor_FoV = np.arange(1, 179, 1)
#
# Minimum_Range = Uncertainty_Ellipse_Radius + Uncertainty_Ellipse_Radius / np.sin(np.radians(Sensor_FoV / 2))
# n = len(Sensor_FoV)
#
# # Initialize arrays
# lidar_Pt = np.zeros(n)
# ppp = np.zeros(n)
# sp = np.zeros(n)
#
# # Compute values
# for i in range(n):
#     FoV = Sensor_FoV[i]
#     R = Minimum_Range[i]
#
#     divergence_rad = np.radians(FoV)  # convert FoV to radians
#     lidar_Pt[i] = compute_pulse_energy(R, divergence_rad)
#     ppp[i] = compute_photons_p_pulse(R, lidar_Pt[i], divergence_rad)
#     sp[i] = compute_solar_photons(R)
#
# # Plot
# fig, ax1 = plt.subplots(figsize=(10,6))
#
# color1 = 'tab:blue'
# ax1.set_xlabel('FoV [deg]')
# ax1.set_ylabel('Lidar Pulse Energy [J]', color=color1)
# ax1.plot(Sensor_FoV, lidar_Pt, color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
#
# ax2 = ax1.twinx()  # secondary y-axis
# color2 = 'tab:red'
# ax2.set_ylabel('Photons per Pulse / Solar Photons', color=color2)
# ax2.plot(Sensor_FoV, ppp, '--', label='Photons per Pulse', color='tab:orange')
# ax2.plot(Sensor_FoV, sp, ':', label='Solar Photons', color='tab:red')
# ax2.tick_params(axis='y', labelcolor=color2)
#
# fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
# plt.title('Lidar Pulse and Solar Photons vs FoV')
# plt.show()