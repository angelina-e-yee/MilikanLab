# compute_q_method1_slip.py
import numpy as np
import pandas as pd

# ---------- Lab constants (SI) ----------
RHO_OIL = 875.3        # kg/m^3
RHO_AIR = 1.204        # kg/m^3
ETA     = 1.827e-5     # Pa·s
G       = 9.80         # m/s^2
D_PLATE = 6.0e-3       # m  (plate separation)
BETA    = 1.061e-7     # m  (b/p from your lab sheet)
E_CHARGE = 1.602176634e-19  # C

VEL_CSV  = "droplet_velocities_final_V3.csv"
VOLT_CSV = "droplet_voltages.csv"
OUT_CSV  = "Q_method1_V3.csv"

DELTA_RHO = RHO_OIL - RHO_AIR

def radius_iter_vd(vd, n_iter=3):
    """
    Slip-corrected radius from downward speed vd.
    Stokes: r0 = sqrt(9*eta*vd/(2*Δρ*g))
    Cunningham: eta_eff = eta/(1 + BETA/r), iterate r.
    """
    if not np.isfinite(vd) or vd <= 0:
        return np.nan, np.nan
    r = np.sqrt(9.0 * ETA * vd / (2.0 * DELTA_RHO * G))
    eta_eff = ETA
    for _ in range(n_iter):
        r = max(r, 1e-12)  # guard
        eta_eff = ETA / (1.0 + BETA / r)
        r = np.sqrt(9.0 * eta_eff * vd / (2.0 * DELTA_RHO * G))
    return r, eta_eff

def q_from_stop_voltage(r, Vstop):
    """ q = (4/3)π r^3 Δρ g * (d / V_stop) """
    if not (np.isfinite(r) and np.isfinite(Vstop)) or r <= 0 or Vstop <= 0:
        return np.nan
    return (4.0/3.0) * np.pi * (r**3) * DELTA_RHO * G * (D_PLATE / Vstop)

def q_unc_from_vstderr(vd, vd_se, Vstop):
    """
    Propagate uncertainty from vd only via symmetric finite difference.
    """
    if not (np.isfinite(vd) and np.isfinite(vd_se)) or vd <= 0 or vd_se <= 0:
        return np.nan
    q0 = q_from_stop_voltage(*radius_iter_vd(vd))
    q_plus = q_from_stop_voltage(*radius_iter_vd(vd + vd_se))
    vd_minus = max(vd - vd_se, 1e-12)
    q_minus = q_from_stop_voltage(*radius_iter_vd(vd_minus))
    # include Vstop in all calls
    q0 = q_from_stop_voltage(radius_iter_vd(vd)[0], Vstop)
    q_plus = q_from_stop_voltage(radius_iter_vd(vd + vd_se)[0], Vstop)
    q_minus = q_from_stop_voltage(radius_iter_vd(vd_minus)[0], Vstop)
    return 0.5 * abs(q_plus - q_minus)

# ---- Read inputs ----
vels = pd.read_csv(VEL_CSV)
vols = pd.read_csv(VOLT_CSV)
vols.columns = [c.strip() for c in vols.columns]

df = vels.merge(
    vols[["droplet_number", "voltage_stop"]],
    on="droplet_number",
    how="left"
)

# ---- Compute r, q with slip correction ----
vd = pd.to_numeric(df["v_down_mps_wls"], errors="coerce").to_numpy(float)
vd_se = pd.to_numeric(df.get("v_down_stderr_wls"), errors="coerce").to_numpy(float)
Vstop = pd.to_numeric(df["voltage_stop"], errors="coerce").to_numpy(float)

r_list, eta_eff_list, q_list, q_se_list = [], [], [], []

for v, vs, Vs in zip(vd, vd_se, Vstop):
    r, eta_eff = radius_iter_vd(v)
    q = q_from_stop_voltage(r, Vs)
    q_se = q_unc_from_vstderr(v, vs, Vs)
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
    "r_m_slip": r_list,
    "eta_eff_Pa_s": eta_eff_list,
    "q_C_method1_slip": q_list,
    "q_stderr_C_from_v": q_se_list,
    "n_e_estimate": np.array(q_list) / E_CHARGE
})

out = out.rename(columns={
    "r_m_slip": "radius",
    "eta_eff_Pa_s": "effective_viscosity",
    "q_C_method2_slip": "q_method2",
    "q_stderr_C_from_v": "q_unc_from_v"
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


# ---- Apply to your computed charges ----
valid_q = out.loc[out["ok_down"] == True, "q_C_method1_slip"].to_numpy()
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
