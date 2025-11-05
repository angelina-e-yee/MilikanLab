
#
# Figure A: Percent change in radius due to the Cunningham slip correction
# x = r_uncorr (µm), y = 100 * (r_slip - r_uncorr) / r_uncorr  (%)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- CONSTANTS (same as your analysis scripts) -----------------
RHO_OIL = 875.3        # kg/m^3
RHO_AIR = 1.204        # kg/m^3
ETA     = 1.827e-5     # Pa·s
G       = 9.80         # m/s^2
BETA    = 1.061e-7     # m  (b/p)
DELTA_RHO = RHO_OIL - RHO_AIR

# ----------------- INPUT / OUTPUT PATHS --------------------------------------
VEL_CSV      = "droplet_velocities_final_V3.csv"        # required
SLIP_OUT_CSV = "droplet_charges_method1_slip.csv"       # optional (if present)
OUT_FIG      = "figureA_percent_change_radius.png"
OUT_SUMMARY  = "radius_correction_summary.csv"

# ----------------- HELPERS ---------------------------------------------------
def stokes_radius_from_vd(vd):
    """Uncorrected Stokes radius from downward terminal velocity vd."""
    return np.sqrt(9.0 * ETA * vd / (2.0 * DELTA_RHO * G))

def slip_radius_iter_from_vd(vd, n_iter=3):
    """Slip-corrected radius using iterative effective viscosity."""
    if not np.isfinite(vd) or vd <= 0:
        return np.nan
    r = stokes_radius_from_vd(vd)
    eta_eff = ETA
    for _ in range(n_iter):
        r = max(r, 1e-12)
        eta_eff = ETA / (1.0 + BETA / r)
        r = np.sqrt(9.0 * eta_eff * vd / (2.0 * DELTA_RHO * G))
    return r

# ----------------- LOAD DATA -------------------------------------------------
vel = pd.read_csv(VEL_CSV)
vel["ok_down"] = vel.get("ok_down", True).astype(bool)
vel["v_down_mps_wls"] = pd.to_numeric(vel["v_down_mps_wls"], errors="coerce")

# Only “good” downward fits
df = vel.loc[vel["ok_down"] & np.isfinite(vel["v_down_mps_wls"]), 
             ["droplet_number", "v_down_mps_wls"]].copy()

# Compute r_uncorr from vd
df["r_uncorr_m"] = stokes_radius_from_vd(df["v_down_mps_wls"])

# Get r_slip_m:
if os.path.exists(SLIP_OUT_CSV):
    slip = pd.read_csv(SLIP_OUT_CSV)
    # prefer exact join on droplet_number if available
    if "r_m_slip" in slip.columns:
        slip_small = slip[["droplet_number", "r_m_slip"]].dropna()
        df = df.merge(slip_small, on="droplet_number", how="left")
        df.rename(columns={"r_m_slip": "r_slip_m"}, inplace=True)

# If we still don’t have r_slip_m, compute it iteratively from vd
if "r_slip_m" not in df.columns:
    df["r_slip_m"] = df["v_down_mps_wls"].apply(slip_radius_iter_from_vd)

# Drop any non-finite radii
df = df[np.isfinite(df["r_uncorr_m"]) & np.isfinite(df["r_slip_m"])].copy()

# Prepare plotting variables
df["r_uncorr_um"] = df["r_uncorr_m"] * 1e6
df["pct_delta_r"] = 100.0 * (df["r_slip_m"] - df["r_uncorr_m"]) / df["r_uncorr_m"]

# Save a small summary CSV used for the figure (handy for the appendix)
df_out = df[["droplet_number", "r_uncorr_um", "pct_delta_r", "r_slip_m"]].copy()
df_out.rename(columns={
    "r_uncorr_um": "r_uncorr_um",
    "r_slip_m": "r_slip_m"
}, inplace=True)
df_out.to_csv(OUT_SUMMARY, index=False)

# ----------------- PLOT ------------------------------------------------------
plt.figure(figsize=(7, 4.5), dpi=150)
plt.scatter(df["r_uncorr_um"], df["pct_delta_r"], s=20, alpha=0.85)
plt.xlabel(r"Uncorrected Stokes radius $r_{\mathrm{uncorr}}$  ($\mu$m)")
plt.ylabel(r"Percent change in radius  $100\times\frac{r_{\mathrm{slip}}-r_{\mathrm{uncorr}}}{r_{\mathrm{uncorr}}}$  (%)")
plt.title("Cunningham Slip Effect vs. Radius")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_FIG)
print(f"Saved figure: {OUT_FIG}")
print(f"Saved summary CSV: {OUT_SUMMARY}")

# Quick console summary
if len(df) > 0:
    print("Range of percent change in r (min → max): "
          f"{df['pct_delta_r'].min():.2f}% → {df['pct_delta_r'].max():.2f}%")
    print("Median uncorrected radius:", f"{df['r_uncorr_um'].median():.2f} µm")
