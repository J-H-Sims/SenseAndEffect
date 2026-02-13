## To calculate the tolerance to range in accuracies based off the perceived width from a discrete number of pixels

import numpy as np
import matplotlib.pyplot as plt

TargetWidth = 1 #meters
FoVpp = 0.00049 #Degrees per pixel

Range = 7500 #m

Sub_Tgt_Ang = np.atan(TargetWidth/Range) * 360/np.pi

True_Pxl_Width = Sub_Tgt_Ang / FoVpp
print("True Pixel Width:", True_Pxl_Width)

Blur_Circle_Radius_px = 1

Px_W_min= np.floor (True_Pxl_Width - Blur_Circle_Radius_px)
Px_W_max= np.ceil (True_Pxl_Width + Blur_Circle_Radius_px)

Calc_W_min = Range * np.tan( FoVpp * Px_W_min * np.pi / 180 )
Calc_W_max = Range * np.tan( FoVpp * Px_W_max * np.pi / 180 )


W_uncertainty_from_blur = Calc_W_max-Calc_W_min

print("Wdith range:", Calc_W_min, "to", Calc_W_max, ". Uncertainty:", W_uncertainty_from_blur)



Range_step_array = np.arange(1500,20000,100)
Sub_Tgt_Ang = np.atan(TargetWidth/Range_step_array) * 180/np.pi

True_Pxl_Width = Sub_Tgt_Ang / FoVpp

#print(True_Pxl_Width)

#plt.plot(Range_step_array,True_Pxl_Width)
#plt.show()



pxl_count_array = np.arange(5, 120, 1)

# Range corresponding to each pixel count
pxl_count_range = TargetWidth / np.tan(pxl_count_array * FoVpp * np.pi / 180)

# Sensitivity: how much range changes per pixel
pxlrange_grad = np.gradient(pxl_count_range)

print(pxlrange_grad)


#plt.plot(pxl_count_range [1:len(pxl_count_range)-1],-1*pxlrange_grad[1:len(pxl_count_range)-1], linewidth = 1)

#plt.xlabel("Target Range")
#plt.ylabel("Range Sensitivity (m for next  pixel)")
#plt.title("Range Sensitivity vs Pixel Count")
#plt.grid(True)


#plt.show()

Range_step_array = np.arange(min(pxl_count_range),max(pxl_count_range),100)
Sub_Tgt_Ang = np.atan(TargetWidth/Range_step_array) * 180/np.pi

True_Pxl_Width = Sub_Tgt_Ang / FoVpp


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Px Grad / m ', color=color)
ax1.plot(pxl_count_range [1:len(pxl_count_range)-1],-1*pxlrange_grad[1:len(pxl_count_range)-1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Px Width', color=color)  # we already handled the x-label with ax1
ax2.plot(Range_step_array, True_Pxl_Width, color=color)
ax2.tick_params(axis='y', labelcolor=color)
title = "Target Width " + str(TargetWidth) +" m"
plt.title(title)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
