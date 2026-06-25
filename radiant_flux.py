"""
radiant_flux.py

Spacecraft radiant flux environment look-up table.

Provides solar, albedo, and Earth IR flux values (W/m^2) as a function of:
    case    : 'hot' or 'cold' — bounding thermal environment cases
    altitude: 500 or 700 (km) — orbital altitude
    pointing: face orientation relative to orbit ('zenith', 'nadir', 'sun', 'anti-sun', 'ram')
    beta    : solar beta angle in degrees (0, 45, 70, 90) — angle between the
              orbital plane and the sun vector; higher beta = more direct sunlight

Data are tabulated as (solar, albedo, earth_ir) tuples in W/m^2. None entries
indicate conditions where that flux component does not apply (e.g. zenith face
receives no direct albedo). get_radiant_flux returns the total flux by summing
all non-None components.

Source: SMAD page 688 
"""


def get_radiant_flux(case, altitude, pointing, beta):
    """Return total radiant flux (W/m^2) for the given orbital environment.

    Parameters:
        case    : 'hot' or 'cold'
        altitude: 500 or 700 (km)
        pointing: 'zenith', 'nadir', 'sun', 'anti-sun', or 'ram'
        beta    : 0, 45, 70, or 90 (degrees)

    Returns the scalar sum of solar + albedo + earth_ir, skipping None components.
    Raises ValueError for any invalid combination.
    """

    data = {
        "cold": {
            500: {
                "zenith": {
                    0:  (418.2, None,  None),
                    45: (295.8, None,  None),
                    70: (143.1, None,  None),
                    90: (1.3,   None,  None),
                },
                "nadir": {
                    0:  (30.4,  79.1,  186.8),
                    45: (44.7,  66.3,  186.8),
                    70: (42.7,  42.7,  186.7),
                    90: (1.3,   15.2,  186.5),
                },
                "sun": {
                    0:  (0.8,   24.5,  58.3),
                    45: (630.7, 23.7,  58.1),
                    70: (123.76,19.0,  58.1),
                    90: (1317.0,16.5,  58.0),
                },
                "anti-sun": {
                    0:  (0.8,   24.5,  58.3),
                    45: (None,  17.5,  58.2),
                    70: (None,  8.3,   58.2),
                    90: (None,  5.3,   58.0),
                },
                "ram": {
                    0:  (287.3, 24.6,  58.0),
                    45: (226.0, 20.6,  57.9),
                    70: (143.1, 13.3,  57.9),
                    90: (1.3,   None,  57.9),
                },
            },
            700: {
                "zenith": {
                    0:  (418.2, None,  None),
                    45: (295.8, None,  None),
                    70: (143.1, None,  None),
                    90: (1.3,   None,  None),
                },
                "nadir": {
                    0:  (41.3,  74.6,  176.4),
                    45: (62.0,  62.6,  176.4),
                    70: (40.9,  40.9,  176.3),
                    90: (1.3,   18.8,  176.2),
                },
                "sun": {
                    0:  (0.8,   21.3,  50.8),
                    45: (661.0, 21.4,  50.6),
                    70: (1237.6,18.1,  50.6),
                    90: (1317.0,18.2,  50.6),
                },
                "anti-sun": {
                    0:  (0.8,   21.3,  50.7),
                    45: (None,  14.6,  50.7),
                    70: (None,  6.3,   50.7),
                    90: (None,  5.8,   50.5),
                },
                "ram": {
                    0:  (299.7, 21.4,  50.5),
                    45: (238.4, 18.0,  50.5),
                    70: (143.1, 11.8,  50.5),
                    90: (1.3,   None,  50.5),
                },
            },
        },

        "hot": {
            500: {
                "zenith": {
                    0:  (450.6, None,  None),
                    45: (318.7, None,  None),
                    70: (154.2, None,  None),
                    90: (1.4,   None,  None),
                },
                "nadir": {
                    0:  (32.7,  123.9, 224.6),
                    45: (48.2,  98.9,  224.7),
                    70: (59.6,  59.6,  224.6),
                    90: (1.4,   19.7,  224.4),
                },
                "sun": {
                    0:  (0.9,   38.4,  70.1),
                    45: (679.5, 35.4,  69.9),
                    70: (1333.4,26.5,  69.9),
                    90: (1419.0,21.4,  69.8),
                },
                "anti-sun": {
                    0:  (0.9,   38.5,  70.0),
                    45: (None,  26.1,  70.0),
                    70: (None,  11.5,  70.0),
                    90: (None,  6.8,   69.8),
                },
                "ram": {
                    0:  (309.5, 38.6,  69.7),
                    45: (243.5, 30.8,  69.7),
                    70: (154.2, 18.6,  69.7),
                    90: (1.4,   None,  69.7),
                },
            },

            700: {
                "zenith": {
                    0:  (450.6, None,  None),
                    45: (318.7, None,  None),
                    70: (154.2, None,  None),
                    90: (1.4,   None,  None),
                },
                "nadir": {
                    0:  (44.5,  116.9, 212.2),
                    45: (66.8,  93.5,  212.2),
                    70: (57.0,  57.0,  212.1),
                    90: (1.4,   24.4,  211.9),
                },
                "sun": {
                    0:  (0.9,   33.3,  61.1),
                    45: (712.2, 31.9,  60.9),
                    70: (1333.4,25.2,  60.9),
                    90: (1419.0,23.6,  60.8),
                },
                "anti-sun": {
                    0:  (0.9,   33.4,  60.9),
                    45: (None,  21.7,  61.0),
                    70: (None,  11.5,  60.9),
                    90: (None,  6.8,   60.8),
                },
                "ram": {
                    0:  (322.9, 33.5,  60.7),
                    45: (256.9, 26.8,  60.7),
                    70: (154.2, 16.4,  60.7),
                    90: (1.4,   7.5,   60.7),
                },
            },
        },
    }

    try:
        components  = data[case.lower()][altitude][pointing.lower()][beta]
        total_flux  = sum(x for x in components if x is not None)
        return total_flux
    except KeyError:
        raise ValueError("Invalid input combination")
