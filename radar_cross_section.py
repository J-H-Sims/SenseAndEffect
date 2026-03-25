def get_rcs_dBsm(azimuth):
    data = {
        180: 4.5, 175: 1.0, 170: -3.5, 165: -5.0, 160: -6.0,
        155: -8.5, 150: -10.5, 145: -9.0, 140: -7.5, 135: -6.5,
        130: -6.0, 125: -7.5, 120: -8.5, 115: -6.0, 110: -2.0,
        105: 2.5, 100: 6.0, 95: 10.5, 90: 12.5, 85: 10.0,
        80: 7.5, 75: 3.0, 70: -2.0, 65: -5.5, 60: -8.0,
        55: -7.0, 50: -6.5, 45: -7.5, 40: -8.5, 35: -6.0,
        30: -4.5, 25: -5.5, 20: -7.5, 15: -2.5, 10: 1.5,
        5: 2.2, 0: 2.5, -5: 2.2, -10: 1.5, -15: -2.5,
        -20: -6.5, -25: -5.5, -30: -4.5, -35: -5.5, -40: -6.0,
        -45: -4.5, -50: -3.5, -55: -6.5, -60: -10.0, -65: -7.5,
        -70: -2.5, -75: 3.0, -80: 7.5, -85: 11.0, -90: 13.0,
        -95: 11.5, -100: 8.5, -105: 3.5, -110: -3.0, -115: -5.5,
        -120: -6.0, -125: -5.0, -130: -4.0, -135: -5.5, -140: -6.5,
        -145: -5.5, -150: -5.0, -155: -4.0, -160: -2.5, -165: -1.0,
        -170: 2.0, -175: 3.5, -180: 4.5
    }

    if azimuth in data:
        return data[azimuth]

    # linear interpolation for values between table entries
    import math
    keys = sorted(data.keys())
    for i in range(len(keys) - 1):
        a0, a1 = keys[i], keys[i + 1]
        if a0 <= azimuth <= a1:
            t = (azimuth - a0) / (a1 - a0)
            return data[a0] + t * (data[a1] - data[a0])

    raise ValueError(f"Azimuth {azimuth} out of range [-180, 180]")

def dbsm_to_m2(dbsm):
    return 10 ** (dbsm / 10)

def get_rcs_m2(azimuth):
    return dbsm_to_m2(get_rcs_dBsm(azimuth))

#print(dbsm_to_m2(13))