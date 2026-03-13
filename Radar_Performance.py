import numpy as np
import sympy as sp
#_______
##Radar
#_______
def Gain_Approx(Beamwidth):
    Gain = 4*np.pi/(2*np.pi*(1-np.cos(np.radians(Beamwidth/2))))
    return Gain

# Radar symbols
Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar = sp.symbols(
    'Pr_radar Pt_radar G_radar lambda_radar RCS_radar R_radar L_radar'
)

# Monostatic radar equation (linear)
eq_radar = Pr_radar - (Pt_radar * G_radar**2 * lambda_radar**2 * RCS_radar) / ((4*sp.pi)**3 * R_radar**4 * L_radar)

def compute_missing_radar(**kwargs):
    symbols = [Pr_radar, Pt_radar, G_radar, lambda_radar, RCS_radar, R_radar, L_radar]
    missing = [v for v in symbols if kwargs.get(v.name) is None]
    if len(missing) != 1:
        raise ValueError("Exactly one variable must be None for radar")
    subs = {v: kwargs[v.name] for v in symbols if kwargs.get(v.name) is not None}
    sol = sp.solve(eq_radar.subs(subs), missing[0])
    # Filter only positive real solutions
    sol_real_positive = [s.evalf() for s in sol if s.is_real and s > 0]
    if not sol_real_positive:
        raise ValueError("No positive real solution found. Check inputs.")
    return float(sol_real_positive[0])

