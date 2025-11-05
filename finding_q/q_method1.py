# compute_q_method1_slip.py
import numpy as np
import pandas as pd
import math

# ---------- Lab constants (SI) ----------
RHO_OIL = 875.3        # kg/m^3
RHO_AIR = 1.204        # kg/m^3
ETA     = 1.827e-5     # Pa·s (dynamic viscosity of air)
G       = 9.80         # m/s^2
D_PLATE = 6.0e-3       # m  (plate separation)
BETA    = 1.061e-7     # m  (b/p)
E_CHARGE = 1.602176634e-19  # C

# ---------- Uncertainties ----------
SIG_RHO_OIL = 0.44      # kg/m^3
SIG_RHO_AIR = 0.001     # kg/m^3
SIG_ETA     = 0.05e-5   # Pa·s   (example ~2.7%; set your value)
SIG_G       = 0.10      # m/s^2
SIG_D_PLATE = 0.05e-3   # m
SIG_BETA    = 0.0       # m  (set if you have it; else 0)
DEFAULT_SIG_VSTOP = 1.0 # V  (used if column 'voltage_stop_sigma' is absent)

# ---------- Files ----------
VEL_CSV  = "droplet_velocities_final_V3.csv"
VOLT_CSV = "droplet_voltages.csv"
OUT_CSV  = "Q_method1_V3.csv"

DELTA_RHO = RHO_OIL - RHO_AIR
SIG_DELTA_RHO = math.hypot(SIG_RHO_OIL, SIG_RHO_AIR)

# ---------- Slip-corrected radius from v_d ----------
def radius_iter_vd(vd, n_iter=3):
    """
    Slip-corrected radius from downward speed vd.
    Stokes (seed): r0 = sqrt(9*eta*vd/(2*Δρ*g))
    Cunningham: eta_eff = eta/(1 + BETA/r), iterate r.
    Returns (r, eta_eff).
    """
    if not np.isfinite(vd) or vd <= 0:
        return np.nan, np.nan
    r = math.sqrt(max(1e-30, 9.0 * ETA * vd / (2.0 * DELTA_RHO * G)))
    eta_eff = ETA
    for _ in range(n_iter):
        r = max(r, 1e-12)  # guard
        eta_eff = ETA / (1.0 + BETA / r)
        r = math.sqrt(max(1e-30, 9.0 * eta_eff * vd / (2.0 * DELTA_RHO * G)))
    return r, eta_eff

# ---------- q from stopping voltage ----------
def q_from_stop_voltage(r, Vstop):
    """ q = (4/3)π r^3 Δρ g * (d / V_stop) """
    if not (np.isfinite(r) and np.isfinite(Vstop)) or r <= 0 or Vstop <= 0:
        return np.nan
    return (4.0/3.0) * np.pi * (r**3) * DELTA_RHO * G * (D_PLATE / Vstop)

# ---------- Method-1 total uncertainty (FIXED propagation) ----------
def q_unc_method1_total(vd, vd_se, Vstop, Vstop_se, r, eta_eff):
    """
    Implements:
      (σ_q/q)^2 = (3 σ_r/r)^2 + (σ_Vstop/Vstop)^2 + (σ_d/d)^2,
    with
      (σ_r/r)^2 = (S_slip^2 / 4) * [ (σ_nueff/nu_eff)^2 + (σ_vd/vd)^2
                                   + (σ_g/g)^2 + (σ_Δρ/Δρ)^2 ].

    σ_nueff/nueff is from ETA and Cunningham parameter β only
    (r-dependence handled via S_slip).
    """
    if not (np.isfinite(vd) and vd > 0 and np.isfinite(vd_se) and vd_se > 0):
        return np.nan
    if not (np.isfinite(Vstop) and Vstop > 0 and np.isfinite(Vstop_se) and Vstop_se >= 0):
        return np.nan
    if not (np.isfinite(r) and r > 0 and np.isfinite(eta_eff) and eta_eff > 0):
        return np.nan

    # Slip sensitivity
    S_slip = (r + BETA) / (2.0 * r + BETA)

    # nu_eff = ETA / C, C = 1 + BETA/r; propagate ETA and β only:
    C = 1.0 + BETA / r
    sig_C = (SIG_BETA / r) if r > 0 else 0.0
    frac_sig_nueff = math.hypot(SIG_ETA / max(ETA, 1e-30), sig_C / max(C, 1e-30))

    # radius fractional uncertainty
    bracket = (
        (frac_sig_nueff)**2
        + (vd_se / max(vd, 1e-30))**2
        + (SIG_G / max(G, 1e-30))**2
        + (SIG_DELTA_RHO / max(DELTA_RHO, 1e-30))**2
    )
    frac_sig_r = 0.5 * S_slip * math.sqrt(bracket)

    # q fractional uncertainty
    frac_sig_q = math.sqrt(
        (3.0 * frac_sig_r)**2
        + (Vstop_se / Vstop)**2
        + (SIG_D_PLATE / max(D_PLATE, 1e-30))**2
    )
    # absolute σ_q
    q0 = q_from_stop_voltage(r, Vstop)
    return abs(frac_sig_q * q0)

# ---- Read inputs ----
vels = pd.read_csv(VEL_CSV)
vols = pd.read_csv(VOLT_CSV)
vols.columns = [c.strip() for c in vols.columns]

if "voltage_stop_sigma" in vols.columns:
    vols["voltage_stop_sigma"] = pd.to_numeric(vols["voltage_stop_sigma"], errors="coerce")
else:
    vols["voltage_stop_sigma"] = DEFAULT_SIG_VSTOP

df = vels.merge(
    vols[["droplet_number", "voltage_stop", "voltage_stop_sigma"]],
    on="droplet_number",
    how="left"
)

# ---- Compute r, q with slip correction + total σ_q ----
vd    = pd.to_numeric(df["v_down_mps_wls"], errors="coerce").to_numpy(float)
vd_se = pd.to_numeric(df.get("v_down_stderr_wls"), errors="coerce").to_numpy(float)
Vstop = pd.to_numeric(df["voltage_stop"], errors="coerce").to_numpy(float)
Vstop_se = pd.to_numeric(df["voltage_stop_sigma"], errors="coerce").to_numpy(float)

r_list, eta_eff_list, q_list, q_se_list = [], [], [], []

for v, vs, Vs, Vs_se in zip(vd, vd_se, Vstop, Vstop_se):
    r, eta_eff = radius_iter_vd(v)
    q = q_from_stop_voltage(r, Vs)
    q_se = q_unc_method1_total(v, vs, Vs, Vs_se, r, eta_eff)
    r_list.append(r)
    eta_eff_list.append(eta_eff)
    q_list.append(q)
    q_se_list.append(q_se)

out = pd.DataFrame({
    "droplet_number": df["droplet_number"].astype(int),
    "ok_down": df.get("ok_down", True),
    "N_down": df.get("N_down", np.nan),
    "R2_down": df.get("R2_down", np.nan),
    "v_down_mps_wls": vd,
    "v_down_stderr_mps_wls": vd_se,
    "V_stop_V": Vstop,
    "V_stop_sigma_V": Vstop_se,
    "r_m_slip": r_list,
    "eta_eff_Pa_s": eta_eff_list,
    "q_C_method1_slip": q_list,
    "q_stderr_C_total": q_se_list,  # <-- full uncertainty (this is the one to use)
    "n_e_estimate": np.array(q_list) / E_CHARGE
})

out = out.rename(columns={
    "r_m_slip": "radius",
    "eta_eff_Pa_s": "effective_viscosity"
})

out.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(out)} rows.")
print(out.head(12).to_string(index=False))

# ---------- Find Greatest Common Denominator of Charges ----------
def estimate_elementary_charge(q_values, e_min=0.5e-19, e_max=2.0e-19, n_steps=5000):
    """
    Rough GCD finder for quantized charges.
    q_values: iterable of positive finite charges (C)
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
    e_best = e_trials[best_idx]
    return e_best, residuals[best_idx]

# ---- Apply to computed charges ----
valid_mask = (out["ok_down"] == True) & np.isfinite(out["q_C_method1_slip"])
valid_q = out.loc[valid_mask, "q_C_method1_slip"].to_numpy()
e_est, score = estimate_elementary_charge(valid_q)

print(f"\nEstimated elementary charge ≈ {e_est:.3e} C (residual {score:.3e})")
print(f"Ratio to CODATA e: {e_est / E_CHARGE:.3f}")

# ---- Append result to CSV (optional summary row) ----
summary = pd.DataFrame([{
    "droplet_number": "GCD_estimate",
    "q_C_method1_slip": e_est,
    "n_e_estimate": e_est / E_CHARGE,
    "residual_score": score
}])
out = pd.concat([out, summary], ignore_index=True)
out.to_csv(OUT_CSV, index=False)
print(f"Saved updated {OUT_CSV} with GCD estimate.")
