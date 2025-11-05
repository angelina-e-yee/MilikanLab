# compute_q_method2_slip.py
import numpy as np
import pandas as pd
import math

# ---------- Lab constants (SI) ----------
RHO_OIL  = 875.3        # kg/m^3
RHO_AIR  = 1.204        # kg/m^3
ETA      = 1.827e-5     # Pa·s  (used only to get r from v_d)
G        = 9.80         # m/s^2
D_PLATE  = 6.0e-3       # m  (plate separation)
BETA     = 1.061e-7     # m  (Cunningham b/p)
E_CHARGE = 1.602176634e-19  # C

# ---------- Uncertainties ----------
SIG_RHO_OIL = 0.44       # kg/m^3
SIG_RHO_AIR = 0.001      # kg/m^3
SIG_G       = 0.10       # m/s^2
SIG_D_PLATE = 0.05e-3    # m
DEFAULT_SIG_VRISE = 1.0  # V

# ---------- Files ----------
VEL_CSV  = "droplet_velocities_final_V3.csv"
VOLT_CSV = "droplet_voltages.csv"
OUT_CSV  = "Q_method2_V3.csv"

DELTA_RHO = RHO_OIL - RHO_AIR
SIG_DELTA_RHO = math.hypot(SIG_RHO_OIL, SIG_RHO_AIR)

# ---------- Shared helpers (same style as Method 1) ----------
def radius_iter_vd(vd, n_iter=3):
    """
    Slip-corrected radius from downward speed vd.
    Start with Stokes radius, then iterate the Cunningham correction:
      eta_eff = eta / (1 + BETA/r)
    """
    if not np.isfinite(vd) or vd <= 0:
        return np.nan, np.nan
    # Stokes seed (no slip)
    r = math.sqrt(max(1e-30, 9.0 * ETA * vd / (2.0 * DELTA_RHO * G)))
    eta_eff = ETA
    for _ in range(n_iter):
        r = max(r, 1e-12)
        eta_eff = ETA / (1.0 + BETA / r)
        r = math.sqrt(max(1e-30, 9.0 * eta_eff * vd / (2.0 * DELTA_RHO * G)))
    return r, eta_eff

def q_method2_from_v(r, vd, vu, Vrise):
    """
    Method 2 (rise & fall) with slip-corrected radius r:
      q = (4/3)π r^3 Δρ g * d * (vd + vu) / (V_rise * vd)
    """
    if not (np.isfinite(r) and np.isfinite(vd) and np.isfinite(vu) and np.isfinite(Vrise)):
        return np.nan
    if r <= 0 or vd <= 0 or Vrise <= 0:
        return np.nan
    return (4.0/3.0) * np.pi * (r**3) * DELTA_RHO * G * D_PLATE * (vd + vu) / (Vrise * vd)

# ---------- Direct uncertainty for Method 2 (matches your formula) ----------
def q_unc_method2_total(q, vd, vd_se, vu, vu_se, Vrise, Vrise_se):
    """
    Implements (σ_q/q)^2 =
        (σ_vd^2 + σ_vu^2)/(vd + vu)^2   +   (σ_vd/vd)^2
      + (σ_Vrise/Vrise)^2  +  (σ_g/g)^2  +  (σ_d/d)^2  +  (σ_Δρ/Δρ)^2
    Returns absolute σ_q.
    """
    good = all(np.isfinite(x) for x in [q, vd, vd_se, vu, vu_se, Vrise, Vrise_se])
    if (not good) or (q <= 0) or (vd <= 0) or (Vrise <= 0):
        return np.nan

    term_velsum = (vd_se**2 + vu_se**2) / max((vd + vu)**2, 1e-30)
    term_vd     = (vd_se / max(vd, 1e-30))**2
    term_Vr     = (Vrise_se / max(Vrise, 1e-30))**2
    term_g      = (SIG_G / max(G, 1e-30))**2
    term_d      = (SIG_D_PLATE / max(D_PLATE, 1e-30))**2
    term_drho   = (SIG_DELTA_RHO / max(DELTA_RHO, 1e-30))**2

    frac_var = term_velsum + term_vd + term_Vr + term_g + term_d + term_drho
    return abs(q) * math.sqrt(frac_var)

def estimate_elementary_charge(q_values, e_min=0.5e-19, e_max=2.0e-19, n_steps=5000):
    """Same GCD finder as Method 1. Returns (e_best, residual_score)."""
    q_values = np.asarray(q_values)
    q_values = q_values[np.isfinite(q_values) & (q_values > 0)]
    if len(q_values) < 3:
        return np.nan, np.nan

    e_trials = np.linspace(e_min, e_max, n_steps)
    residuals = []
    for e in e_trials:
        multiples = q_values / e
        frac_part = np.abs(multiples - np.round(multiples))
        residuals.append(np.mean(frac_part))
    best_idx = np.argmin(residuals)
    return e_trials[best_idx], residuals[best_idx]

# ---------- Read inputs ----------
vels = pd.read_csv(VEL_CSV)
vols = pd.read_csv(VOLT_CSV)
vols.columns = [c.strip() for c in vols.columns]

# We need up & down velocities and their uncertainties
need = ["droplet_number", "v_down_mps_wls", "v_up_mps_wls",
        "v_down_stderr_wls", "v_up_stderr_wls", "ok_down", "ok_up",
        "N_down", "N_up", "R2_down", "R2_up"]
for c in need:
    if c not in vels.columns:
        raise ValueError(f"Required column '{c}' missing from {VEL_CSV}")

# Rise voltage and its uncertainty (column optional)
if "voltage_rise_sigma" in vols.columns:
    vols["voltage_rise_sigma"] = pd.to_numeric(vols["voltage_rise_sigma"], errors="coerce")
else:
    vols["voltage_rise_sigma"] = DEFAULT_SIG_VRISE

df = vels.merge(
    vols[["droplet_number", "voltage_rise", "voltage_rise_sigma"]],
    on="droplet_number",
    how="left"
)

# ---------- Compute r (from vd with slip), then q_method2 + σ_q ----------
vd     = pd.to_numeric(df["v_down_mps_wls"], errors="coerce").to_numpy(float)
vu     = pd.to_numeric(df["v_up_mps_wls"],   errors="coerce").to_numpy(float)
vd_se  = pd.to_numeric(df["v_down_stderr_wls"], errors="coerce").to_numpy(float)
vu_se  = pd.to_numeric(df["v_up_stderr_wls"],   errors="coerce").to_numpy(float)
Vrise  = pd.to_numeric(df["voltage_rise"], errors="coerce").to_numpy(float)
Vrise_se = pd.to_numeric(df["voltage_rise_sigma"], errors="coerce").to_numpy(float)

r_list, eta_eff_list, q_list, q_se_list = [], [], [], []

for v_d, v_u, se_d, se_u, Vr, Vr_se in zip(vd, vu, vd_se, vu_se, Vrise, Vrise_se):
    r, eta_eff = radius_iter_vd(v_d)            # only to evaluate q formula
    q = q_method2_from_v(r, v_d, v_u, Vr)
    q_se = q_unc_method2_total(q, v_d, se_d, v_u, se_u, Vr, Vr_se)
    r_list.append(r)
    eta_eff_list.append(eta_eff)
    q_list.append(q)
    q_se_list.append(q_se)

out = pd.DataFrame({
    "droplet_number": df["droplet_number"].astype(int),
    "ok_down": df["ok_down"],
    "ok_up": df["ok_up"],
    "N_down": df["N_down"],
    "N_up": df["N_up"],
    "R2_down": df["R2_down"],
    "R2_up": df["R2_up"],
    "v_down_mps_wls": vd,
    "v_up_mps_wls": vu,
    "v_down_stderr_mps_wls": vd_se,
    "v_up_stderr_mps_wls": vu_se,
    "V_rise_V": Vrise,
    "V_rise_sigma_V": Vrise_se,
    "r_m_slip": r_list,
    "eta_eff_Pa_s": eta_eff_list,
    "q_C_method2_slip": q_list,
    "q_stderr_C_total": q_se_list,   # <-- total σ_q (this is the one to use)
    "n_e_estimate": np.array(q_list) / E_CHARGE
})

# ---- Rename to mirror Method 1 renames ----
out = out.rename(columns={
    "r_m_slip": "radius",
    "eta_eff_Pa_s": "effective_viscosity",
    "q_C_method2_slip": "q_method2"
})

out.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(out)} rows.")
print(out.head(12).to_string(index=False))

# ---------- GCD estimate (same as Method 1) ----------
valid_q = out.loc[(out.get("ok_down", True) == True) & (out.get("ok_up", True) == True),
                  "q_method2"].to_numpy()
e_est, score = estimate_elementary_charge(valid_q)

print(f"\nEstimated elementary charge (Method 2) ≈ {e_est:.3e} C (residual {score:.3e})")
print(f"Ratio to CODATA e: {e_est / E_CHARGE:.3f}")

# Append a summary row so the CSV mirrors Method 1’s output
summary = pd.DataFrame([{
    "droplet_number": "GCD_estimate",
    "ok_down": "",
    "ok_up": "",
    "N_down": "",
    "N_up": "",
    "R2_down": "",
    "R2_up": "",
    "v_down_mps_wls": "",
    "v_up_mps_wls": "",
    "v_down_stderr_mps_wls": "",
    "v_up_stderr_mps_wls": "",
    "V_rise_V": "",
    "V_rise_sigma_V": "",
    "radius": "",
    "effective_viscosity": "",
    "q_method2": e_est,
    "q_stderr_C_total": "",
    "n_e_estimate": e_est / E_CHARGE
}])
out = pd.concat([out, summary], ignore_index=True)
out.to_csv(OUT_CSV, index=False)
print(f"Saved updated {OUT_CSV} with GCD estimate.")
