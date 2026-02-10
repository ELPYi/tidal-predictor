"""
Harmonic tidal prediction engine.

Implements the standard harmonic method used by NOAA/NOS:
    h(t) = Z0 + sum[ f_i * A_i * cos(speed_i * t + V0_i + u_i - g_i) ]

Astronomical arguments follow Meeus (Astronomical Algorithms).
Nodal corrections follow Schureman (Manual of Harmonic Analysis).
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constituent database
# ---------------------------------------------------------------------------
# Each constituent: speed (deg/hr), Doodson numbers [T, s, h, p, N', p1]
# T = lunar hour angle, s = moon longitude, h = sun longitude,
# p = lunar perigee, N' = negative node longitude, p1 = solar perigee

CONSTITUENTS = {
    # Semi-diurnal
    "M2":      {"speed": 28.984104,  "doodson": [2, -2,  2,  0,  0,  0]},
    "S2":      {"speed": 30.000000,  "doodson": [2,  0,  0,  0,  0,  0]},
    "N2":      {"speed": 28.439730,  "doodson": [2, -3,  2,  1,  0,  0]},
    "K2":      {"speed": 30.082137,  "doodson": [2,  0,  2,  0,  0,  0]},
    "2N2":     {"speed": 27.895355,  "doodson": [2, -4,  2,  2,  0,  0]},
    "mu2":     {"speed": 27.968208,  "doodson": [2, -4,  4,  0,  0,  0]},
    "nu2":     {"speed": 28.512583,  "doodson": [2, -3,  4, -1,  0,  0]},
    "L2":      {"speed": 29.528479,  "doodson": [2, -1,  2, -1,  0,  0]},
    "T2":      {"speed": 29.958933,  "doodson": [2,  0, -1,  0,  0,  1]},
    "R2":      {"speed": 30.041067,  "doodson": [2,  0,  1,  0,  0, -1]},
    "lambda2": {"speed": 29.455625,  "doodson": [2, -1,  0,  1,  0,  0]},
    "2SM2":    {"speed": 31.015896,  "doodson": [2,  2, -2,  0,  0,  0]},

    # Diurnal
    "K1":      {"speed": 15.041069,  "doodson": [1,  0,  1,  0,  0,  0]},
    "O1":      {"speed": 13.943035,  "doodson": [1, -2,  1,  0,  0,  0]},
    "P1":      {"speed": 14.958931,  "doodson": [1,  0, -1,  0,  0,  0]},
    "Q1":      {"speed": 13.398661,  "doodson": [1, -3,  1,  1,  0,  0]},
    "J1":      {"speed": 15.585443,  "doodson": [1,  1,  1, -1,  0,  0]},
    "M1":      {"speed": 14.496694,  "doodson": [1, -1,  1,  0,  0,  0]},
    "OO1":     {"speed": 16.139102,  "doodson": [1,  2,  1,  0,  0,  0]},
    "S1":      {"speed": 15.000000,  "doodson": [1,  0,  0,  0,  0,  0]},
    "2Q1":     {"speed": 12.854286,  "doodson": [1, -4,  1,  2,  0,  0]},
    "rho1":    {"speed": 13.471515,  "doodson": [1, -3,  3, -1,  0,  0]},

    # Long period
    "Mf":      {"speed":  1.098033,  "doodson": [0,  2,  0,  0,  0,  0]},
    "Mm":      {"speed":  0.544375,  "doodson": [0,  1,  0, -1,  0,  0]},
    "Ssa":     {"speed":  0.082137,  "doodson": [0,  0,  2,  0,  0,  0]},
    "Sa":      {"speed":  0.041069,  "doodson": [0,  0,  1,  0,  0,  0]},
    "MSF":     {"speed":  1.015896,  "doodson": [0,  2, -2,  0,  0,  0]},

    # Shallow water / compound
    "M4":      {"speed": 57.968208,  "doodson": [4, -4,  4,  0,  0,  0]},
    "M6":      {"speed": 86.952313,  "doodson": [6, -6,  6,  0,  0,  0]},
    "M8":      {"speed": 115.936417, "doodson": [8, -8,  8,  0,  0,  0]},
    "MK3":     {"speed": 44.025173,  "doodson": [3, -2,  3,  0,  0,  0]},
    "2MK3":    {"speed": 42.927139,  "doodson": [3, -4,  5,  0,  0,  0]},
    "S4":      {"speed": 60.000000,  "doodson": [4,  0,  0,  0,  0,  0]},
    "MN4":     {"speed": 57.423834,  "doodson": [4, -5,  4,  1,  0,  0]},
    "MS4":     {"speed": 58.984104,  "doodson": [4, -2,  2,  0,  0,  0]},
    "S6":      {"speed": 90.000000,  "doodson": [6,  0,  0,  0,  0,  0]},
    "M3":      {"speed": 43.476156,  "doodson": [3, -3,  3,  0,  0,  0]},
}

CONSTITUENT_NAMES = sorted(CONSTITUENTS.keys())


# ---------------------------------------------------------------------------
# Astronomical arguments (Meeus, Astronomical Algorithms)
# ---------------------------------------------------------------------------

def _julian_centuries(dt: datetime) -> float:
    """Julian centuries from J2000.0 (2000-01-01 12:00 TT ≈ UTC)."""
    # Julian day number
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return (jd - 2451545.0) / 36525.0


def _astro_args(T: float) -> dict:
    """
    Compute fundamental astronomical arguments in degrees.

    T: Julian centuries from J2000.0

    Returns dict with keys: s, h, p, N, pp
      s  = mean longitude of the Moon
      h  = mean longitude of the Sun
      p  = longitude of lunar perigee
      N  = longitude of ascending lunar node
      pp = longitude of solar perigee (perihelion)
    """
    # Mean longitude of Moon (s)
    s = (218.3164477 + 481267.88123421 * T
         - 0.0015786 * T**2 + T**3 / 538841.0 - T**4 / 65194000.0)

    # Mean longitude of Sun (h)
    h = (280.46646 + 36000.76983 * T + 0.0003032 * T**2)

    # Longitude of lunar perigee (p)
    p = (83.3532465 + 4069.0137287 * T
         - 0.0103200 * T**2 - T**3 / 80053.0 + T**4 / 18999000.0)

    # Longitude of ascending lunar node (N)
    N = (125.04452 - 1934.136261 * T
         + 0.0020708 * T**2 + T**3 / 450000.0)

    # Longitude of solar perigee (pp) -- very slow motion
    pp = (282.93768 + 1.71946 * T + 0.00045688 * T**2)

    return {"s": s % 360, "h": h % 360, "p": p % 360,
            "N": N % 360, "pp": pp % 360}


# ---------------------------------------------------------------------------
# Schureman nodal corrections
# ---------------------------------------------------------------------------

def _nodal_corrections(N_deg: float, p_deg: float, I_deg: float = None):
    """
    Compute node factor f and node argument u for each constituent.

    Follows Schureman's formulas using the longitude of the lunar node (N)
    and related derived quantities.

    Returns: dict of {constituent_name: (f, u_degrees)}
    """
    N = math.radians(N_deg)
    p = math.radians(p_deg)

    # Schureman's I -- inclination of lunar orbit to equator
    # cos(I) = 0.9136 - 0.0356 cos(N)
    cosI = 0.9136 - 0.0356 * math.cos(N)
    cosI = max(-1.0, min(1.0, cosI))
    I = math.acos(cosI)
    sinI = math.sin(I)
    sin2I = math.sin(2 * I)

    # Derived: xi, nu from Schureman
    # tan(xi) = sin(N) * cos(omega_m) / (cos(N) * cos(I_m) + sin(I_m) * cot(omega_m))
    # Simplified Schureman approximations:
    omega_m = math.radians(23.452)  # obliquity approx

    # nu: longitude in the lunar orbit
    sin_nu = math.sin(N) * math.cos(omega_m) / math.sin(I)
    sin_nu = max(-1.0, min(1.0, sin_nu))
    nu = math.asin(sin_nu)

    # xi
    sin_xi = math.sin(N) * math.sin(omega_m) / math.sin(I)
    sin_xi = max(-1.0, min(1.0, sin_xi))
    xi = math.asin(sin_xi)

    # 2nu'' (nup2) for K2
    sin_2nup = (math.sin(2 * N) * (math.sin(omega_m) ** 2)
                / (math.sin(I) ** 2))
    sin_2nup = max(-1.0, min(1.0, sin_2nup))
    nup2 = 0.5 * math.asin(sin_2nup)

    # nu' (nup) for K1
    sin_2nup1 = math.sin(2 * N) * math.sin(omega_m) * math.cos(omega_m) / math.sin(2 * I)
    sin_2nup1 = max(-1.0, min(1.0, sin_2nup1))
    nup = 0.5 * math.asin(sin_2nup1)

    # Precompute common terms
    sin_I_half = math.sin(I / 2)
    cos_I_half = math.cos(I / 2)

    corrections = {}

    # --- Semi-diurnal ---
    # M2: f = sin^2(2I) / 0.1578 (Schureman eq 227)  -- simplified
    # f_M2 = (cos(I/2))^4 / 0.91544  (Schureman Table 2)
    f_M2 = cos_I_half**4 / 0.9154
    u_M2 = -2 * xi
    corrections["M2"] = (f_M2, math.degrees(u_M2))

    # N2, 2N2, mu2, nu2, lambda2 same as M2
    for name in ["N2", "2N2", "mu2", "nu2", "lambda2"]:
        corrections[name] = (f_M2, math.degrees(u_M2))

    # S2: no nodal correction
    corrections["S2"] = (1.0, 0.0)
    corrections["T2"] = (1.0, 0.0)
    corrections["R2"] = (1.0, 0.0)
    corrections["2SM2"] = (f_M2, math.degrees(u_M2))

    # K2: f = ((sin I)^2 * (1 + cos I / 2)) / 0.1578 ... simplified:
    # f_K2 = ((0.8965 * sin(2I)^2 + 0.6001 * sin(2I) * cos(nu) + 0.1006)^0.5)
    # Schureman simplified:
    f_K2_term = (0.8965 * sin2I**2 + 0.6001 * sin2I * math.cos(2 * nu) + 0.1006)
    f_K2 = max(f_K2_term, 0.0001) ** 0.5
    u_K2 = -2 * nup2
    corrections["K2"] = (f_K2, math.degrees(u_K2))

    # L2: more complex, approximate with M2 factor
    f_L2_term = 1.0 - 12.0 * sin_I_half**2 * cos_I_half**2 * math.cos(2 * p) + 36.0 * sin_I_half**4 * cos_I_half**4
    f_L2 = f_M2 * (max(f_L2_term, 0.0001) ** 0.5)
    # u_L2 approximate
    R_num = -math.sin(2 * p) * sin_I_half**2
    R_den = cos_I_half**2 / 6.0 - math.cos(2 * p) * sin_I_half**2
    if abs(R_den) > 1e-10:
        u_L2 = u_M2 - math.atan2(R_num, R_den)
    else:
        u_L2 = u_M2
    corrections["L2"] = (f_L2, math.degrees(u_L2))

    # --- Diurnal ---
    # O1: f = sin(I) * cos(I/2)^2 / 0.3800
    f_O1 = sinI * cos_I_half**2 / 0.3800
    u_O1 = -2 * xi
    corrections["O1"] = (f_O1, math.degrees(u_O1))

    # Q1, 2Q1, rho1 same as O1
    for name in ["Q1", "2Q1", "rho1"]:
        corrections[name] = (f_O1, math.degrees(u_O1))

    # K1: f from Schureman
    f_K1_term = (0.8965 * sin2I**2 + 0.6001 * sin2I * math.cos(nu) + 0.1006)
    f_K1 = max(f_K1_term, 0.0001) ** 0.5
    u_K1 = -nup
    corrections["K1"] = (f_K1, math.degrees(u_K1))

    # P1: no nodal correction
    corrections["P1"] = (1.0, 0.0)
    corrections["S1"] = (1.0, 0.0)

    # J1: f = sin(2I) / 0.3800
    f_J1 = sin2I / 0.3800
    u_J1 = -nu
    corrections["J1"] = (f_J1, math.degrees(u_J1))

    # M1: approximate
    corrections["M1"] = (f_O1, math.degrees(u_O1))

    # OO1: f = sin(I) * sin(I/2)^2 / 0.01640
    f_OO1 = sinI * sin_I_half**2 / 0.01640
    u_OO1 = -2 * xi
    corrections["OO1"] = (f_OO1, math.degrees(u_OO1))

    # --- Long period ---
    # Mf: f = sin^2(I) / 0.1578
    f_Mf = sinI**2 / 0.1578
    u_Mf = -2 * xi
    corrections["Mf"] = (f_Mf, math.degrees(u_Mf))

    # Mm: f = (2/3 - sin^2(I)) / 0.5021
    f_Mm_num = 2.0 / 3.0 - sinI**2
    f_Mm = abs(f_Mm_num) / 0.5021
    corrections["Mm"] = (f_Mm, 0.0)

    # Ssa, Sa, MSF: no corrections
    corrections["Ssa"] = (1.0, 0.0)
    corrections["Sa"] = (1.0, 0.0)
    corrections["MSF"] = (1.0, 0.0)

    # --- Compound / shallow water ---
    # M4 = 2*M2
    corrections["M4"] = (f_M2**2, 2 * math.degrees(u_M2))
    # M6 = 3*M2
    corrections["M6"] = (f_M2**3, 3 * math.degrees(u_M2))
    # M8 = 4*M2
    corrections["M8"] = (f_M2**4, 4 * math.degrees(u_M2))
    # M3 = M2^(3/2) approx
    corrections["M3"] = (f_M2**1.5, 1.5 * math.degrees(u_M2))
    # MK3 = M2 * K1
    corrections["MK3"] = (f_M2 * f_K1, math.degrees(u_M2) + math.degrees(u_K1))
    # 2MK3 = M2^2 / K1 approx
    corrections["2MK3"] = (f_M2**2 * f_K1, 2 * math.degrees(u_M2) - math.degrees(u_K1))
    # S4: no correction
    corrections["S4"] = (1.0, 0.0)
    corrections["S6"] = (1.0, 0.0)
    # MN4 = M2 * N2
    corrections["MN4"] = (f_M2**2, 2 * math.degrees(u_M2))
    # MS4 = M2 * S2
    corrections["MS4"] = (f_M2, math.degrees(u_M2))

    return corrections


# ---------------------------------------------------------------------------
# Equilibrium argument V0
# ---------------------------------------------------------------------------

def _equilibrium_v0(astro: dict, constituent_name: str) -> float:
    """
    Compute equilibrium argument V0 in degrees for a constituent
    at a given time, from astronomical arguments.

    V0 = sum(doodson_i * astro_arg_i)

    The Doodson numbers multiply [T, s, h, p, N', p1] where
    T (lunar hour angle) = h - s + 90° (relative to lower transit).
    We compute T from h and s.
    """
    info = CONSTITUENTS[constituent_name]
    d = info["doodson"]

    # T = hour angle of mean moon = h - s (+ correction for Greenwich)
    # In standard Doodson notation, T counts from moon's lower transit
    # T = tau = h - s + 180° ... but we handle the 180° in the formula
    # Actually: tau = 15*t + h - s  where t is hours from midnight
    # For V0 we use: V0 = d[0]*tau + d[1]*s + d[2]*h + d[3]*p + d[4]*(-N) + d[5]*pp
    # tau is absorbed into the speed * time term, so for V0 at t=0:
    # V0 = d[0]*(h - s) + d[1]*s + d[2]*h + d[3]*p + d[4]*(-N) + d[5]*pp
    # Simplify: V0 = (d[2]+d[0])*h + (d[1]-d[0])*s + d[3]*p - d[4]*N + d[5]*pp

    s_val = astro["s"]
    h_val = astro["h"]
    p_val = astro["p"]
    N_val = astro["N"]
    pp_val = astro["pp"]

    V0 = ((d[2] + d[0]) * h_val
          + (d[1] - d[0]) * s_val
          + d[3] * p_val
          - d[4] * N_val
          + d[5] * pp_val)

    return V0 % 360


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_tides(
    constituents: List[dict],
    start_dt: datetime,
    end_dt: datetime,
    interval_hours: float = 1.0,
    datum_offset: float = 0.0,
) -> Tuple[List[datetime], np.ndarray]:
    """
    Predict tide levels using the harmonic method.

    Parameters
    ----------
    constituents : list of dict
        Each dict has keys: "name" (str), "amplitude" (float), "phase" (float in degrees).
    start_dt : datetime
        Start of prediction (UTC).
    end_dt : datetime
        End of prediction (UTC).
    interval_hours : float
        Time step in hours (default 1.0).
    datum_offset : float
        Vertical offset (Z0, mean water level above datum).

    Returns
    -------
    times : list of datetime
        Prediction times.
    heights : numpy array
        Predicted water levels.
    """
    if not constituents:
        raise ValueError("At least one tidal constituent is required.")

    # Generate time array
    total_hours = (end_dt - start_dt).total_seconds() / 3600.0
    n_points = int(total_hours / interval_hours) + 1
    hours = np.arange(n_points) * interval_hours
    times = [start_dt + timedelta(hours=float(h)) for h in hours]

    # Midpoint for nodal corrections (standard NOAA practice)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    T_mid = _julian_centuries(mid_dt)
    astro_mid = _astro_args(T_mid)
    nodal = _nodal_corrections(astro_mid["N"], astro_mid["p"])

    # Reference time for V0: start of prediction
    T_ref = _julian_centuries(start_dt)
    astro_ref = _astro_args(T_ref)

    # Compute prediction (vectorized)
    heights = np.full(n_points, datum_offset, dtype=np.float64)

    for c in constituents:
        name = c["name"]
        amplitude = c["amplitude"]
        phase_g = c["phase"]  # Greenwich phase lag in degrees

        if name not in CONSTITUENTS:
            continue

        speed = CONSTITUENTS[name]["speed"]  # deg/hr

        # Get nodal corrections
        f, u = nodal.get(name, (1.0, 0.0))

        # Equilibrium argument at reference time
        V0 = _equilibrium_v0(astro_ref, name)

        # Convert to radians
        speed_rad = np.radians(speed)
        V0_rad = math.radians(V0)
        u_rad = math.radians(u)
        g_rad = math.radians(phase_g)

        # h(t) += f * A * cos(speed * t + V0 + u - g)
        phase = speed_rad * hours + V0_rad + u_rad - g_rad
        heights += f * amplitude * np.cos(phase)

    return times, heights


def get_constituent_names() -> List[str]:
    """Return sorted list of all supported constituent names."""
    return CONSTITUENT_NAMES
