# compute_q_method2_slip.py
import numpy as np
import pandas as pd

# ---------- Lab constants (SI) ----------
RHO_OIL  = 875.3        # kg/m^3
RHO_AIR  = 1.204        # kg/m^3
ETA      = 1.827e-5     # Pa·s
G        = 9.80         # m/s^2
D_PLATE  = 6.0e-3       # m  (plate separation)
BETA     = 1.061e-7     # m  (b/p from your lab sheet)
E_CHARGE = 1.602176634e-19  # C

VEL_CSV  = "droplet_velocities_final_V3.csv"
VOLT_CSV = "droplet_voltages.csv"
OUT_CSV  = "Q_method2_V3.csv"

DELTA_RHO = RHO_OIL - RHO_AIR

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
    r = np.sqrt(9.0 * ETA * vd / (2.0 * DELTA_RHO * G))
    eta_eff = ETA
    for _ in range(n_iter):
        r = max(r, 1e-12)
        eta_eff = ETA / (1.0 + BETA / r)
        r = np.sqrt(9.0 * eta_eff * vd / (2.0 * DELTA_RHO * G))
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

def q_unc_from_vstderr_method2(vd, vd_se, vu, vu_se, Vrise):
    """
    Uncertainty in q from velocity uncertainties only (finite differences),
    with slip re-evaluated each time via radius_iter_vd.
    """
    # Guard
    good = all(np.isfinite(x) for x in [vd, vd_se, vu, vu_se, Vrise]) and vd > 0 and Vrise > 0
    if not good or vd_se <= 0 or vu_se <= 0:
        return np.nan

    # Central q at (vd, vu)
    r0, _ = radius_iter_vd(vd)
    q0 = q_method2_from_v(r0, vd, vu, Vrise)

    # Perturb vd
    vd_p = vd + vd_se
    vd_m = max(vd - vd_se, 1e-12)
    r_p, _ = radius_iter_vd(vd_p)
    r_m, _ = radius_iter_vd(vd_m)
    q_vd_p = q_method2_from_v(r_p, vd_p, vu, Vrise)
    q_vd_m = q_method2_from_v(r_m, vd_m, vu, Vrise)
    dq_d_vd = 0.5 * abs(q_vd_p - q_vd_m)

    # Perturb vu (radius depends on vd only)
    q_vu_p = q_method2_from_v(r0, vd, vu + vu_se, Vrise)
    q_vu_m = q_method2_from_v(r0, vd, vu - vu_se, Vrise)
    dq_d_vu = 0.5 * abs(q_vu_p - q_vu_m)

    # Combine in quadrature
    return np.sqrt(dq_d_vd**2 + dq_d_vu**2)

def estimate_elementary_charge(q_values, e_min=0.5e-19, e_max=2.0e-19, n_steps=5000):
    """
    Same GCD finder as Method 1. Returns (e_best, residual_score).
    """
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

df = vels.merge(
    vols[["droplet_number", "voltage_rise"]],
    on="droplet_number",
    how="left"
)

# ---------- Compute r (from vd with slip), then q_method2 ----------
vd    = pd.to_numeric(df["v_down_mps_wls"], errors="coerce").to_numpy(float)
vu    = pd.to_numeric(df["v_up_mps_wls"],   errors="coerce").to_numpy(float)
vd_se = pd.to_numeric(df["v_down_stderr_wls"], errors="coerce").to_numpy(float)
vu_se = pd.to_numeric(df["v_up_stderr_wls"],   errors="coerce").to_numpy(float)
Vrise = pd.to_numeric(df["voltage_rise"], errors="coerce").to_numpy(float)

r_list, eta_eff_list, q_list, q_se_list = [], [], [], []

for v_d, v_u, se_d, se_u, Vr in zip(vd, vu, vd_se, vu_se, Vrise):
    r, eta_eff = radius_iter_vd(v_d)
    q = q_method2_from_v(r, v_d, v_u, Vr)
    q_se = q_unc_from_vstderr_method2(v_d, se_d, v_u, se_u, Vr)
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
    "r_m_slip": r_list,
    "eta_eff_Pa_s": eta_eff_list,
    "q_C_method2_slip": q_list,
    "q_stderr_C_from_v": q_se_list,
    "n_e_estimate": np.array(q_list) / E_CHARGE
})

# ---- (Optional) Filter to good fits only for sanity checks ----
# good_mask = (out["ok_down"] == True) & (out["ok_up"] == True)
# out = out[good_mask]

# ---- Rename to mirror your Method 1 renames ----
out = out.rename(columns={
    "r_m_slip": "radius",
    "eta_eff_Pa_s": "effective_viscosity",
    "q_C_method2_slip": "q_method2",
    "q_stderr_C_from_v": "q_unc_from_v"
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
    "radius": "",
    "effective_viscosity": "",
    "q_method2": e_est,
    "q_unc_from_v": "",
    "n_e_estimate": e_est / E_CHARGE
}])
out = pd.concat([out, summary], ignore_index=True)
out.to_csv(OUT_CSV, index=False)
print(f"Saved updated {OUT_CSV} with GCD estimate.")
