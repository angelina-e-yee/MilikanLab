
"""
Final Millikan velocity computation:
- Robust steady-region detection (plateau trim + MAD derivative filter)
- Remove long 'stuck' runs (identical pixel values)
- Weighted Least Squares (WLS) with per-point σ_y from pixel + calibration
- Report: v (WLS), stderr_v (WLS), R2 (unweighted), reduced_chi2 (weighted), N
- Also output OLS velocity & stderr as a cross-check.

Input:  droplet_positions.csv  (first column = time (s); other columns like '1d','1u','2d',...)
Output: droplet_velocities_final.csv
"""

import re
import numpy as np
import pandas as pd

# ----------------- CONFIG -----------------
PX_PER_MM       = 520.0     # calibration C (px/mm)
SIG_C_PX_PER_MM = 1.0       # uncertainty in C (px/mm)
SIG_POS_PX      = 0.5       # per-frame position noise in pixels (tracking)
MIN_POINTS      = 12        # minimum points for a valid steady fit
MAD_K           = 3.5       # robust velocity filter strength
PLATEAU_PX      = 1.0       # |Δpx| < this → "no motion" for trimming ends
STUCK_THRESHOLD = 10        # drop runs of >= this many identical pixels
R2_MIN          = 0.98      # quality gate
# ------------------------------------------

def _px_to_m(x_px: np.ndarray) -> np.ndarray:
    """Convert pixels to meters using C (px/mm)."""
    # y_mm = y_px / C; convert to m
    return (x_px / PX_PER_MM) / 1000.0

def _sigma_y_m(y_px: np.ndarray) -> np.ndarray:
    """
    Per-point position uncertainty in meters from pixel noise & calibration error.
    For y_mm = y_px / C:
      σ_y(mm) = sqrt( (σ_px/C)^2 + (y_px*σ_C/C^2)^2 )
    Then convert to meters.
    """
    term1 = (SIG_POS_PX / PX_PER_MM)**2
    term2 = (y_px * SIG_C_PX_PER_MM / (PX_PER_MM**2))**2
    sigma_mm = np.sqrt(term1 + term2)
    return sigma_mm / 1000.0  # mm -> m

def remove_stuck_points(t, y_px, threshold=STUCK_THRESHOLD):
    """Drop long runs of identical pixel values (tracking stalls)."""
    t = np.asarray(t, float)
    y = np.asarray(y_px, float)
    keep = np.ones_like(y, dtype=bool)
    run = 0
    for i in range(1, len(y)):
        if np.isfinite(y[i]) and np.isfinite(y[i-1]) and (y[i] == y[i-1]):
            run += 1
        else:
            if run >= threshold - 1:
                keep[i-run-1:i-1] = False
            run = 0
    if run >= threshold - 1:
        keep[-run-1:-1] = False
    return t[keep], y[keep]

def robust_steady_slice(t, y_px):
    """
    Find the longest steady-velocity chunk:
      1) drop NaNs
      2) trim leading/trailing plateaus (|Δpx| < PLATEAU_PX)
      3) keep frames where gradient ~ median within MAD_K * 1.4826 * MAD
      4) return longest contiguous True run as a slice
    """
    t = np.asarray(t, float)
    y = pd.to_numeric(pd.Series(y_px), errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(y) < MIN_POINTS:
        return slice(0, 0)

    # Trim plateaus
    dy = np.diff(y, prepend=y[0])
    moving = np.abs(dy) >= PLATEAU_PX
    if not np.any(moving):
        return slice(0, 0)
    i0 = np.argmax(moving)
    i1 = len(moving) - 1 - np.argmax(moving[::-1])
    t, y = t[i0:i1+1], y[i0:i1+1]
    off = i0
    if len(y) < MIN_POINTS:
        return slice(0, 0)

    # Robust derivative filter
    v = np.gradient(y, t)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-12
    keep = np.abs(v - med) <= MAD_K * 1.4826 * mad
    if not np.any(keep):
        return slice(0, 0)

    # Longest contiguous True run
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, ok in enumerate(keep):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    if best_len < MIN_POINTS:
        return slice(0, 0)
    return slice(off + best_start, off + best_start + best_len)

def wls_fit(t, y_m, sigma_m):
    """
    Weighted least squares fit for y = m t + b.
    Returns: m, b, stderr_m, R2_unweighted, chi2_red, N
    """
    t = np.asarray(t, float)
    y = np.asarray(y_m, float)
    s = np.asarray(sigma_m, float)
    # Keep finite & positive-σ
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    t, y, s = t[mask], y[mask], s[mask]
    N = len(t)
    if N < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, N

    w = 1.0 / (s**2)
    W = np.sum(w)
    tbar = np.sum(w * t) / W
    ybar = np.sum(w * y) / W
    Sxx = np.sum(w * (t - tbar)**2)
    Sxy = np.sum(w * (t - tbar) * (y - ybar))
    if Sxx <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, N

    m = Sxy / Sxx
    b = ybar - m * tbar

    # Residuals & weighted variance
    yhat = m * t + b
    r = y - yhat

    dof = max(N - 2, 1)
    chi2 = np.sum((r / s)**2)
    chi2_red = chi2 / dof

    # Weighted estimate of variance of residuals
    s2 = chi2 / dof
    stderr_m = np.sqrt(s2 / Sxx)

    # R^2 (unweighted, purely diagnostic on the same window)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) or 1e-12
    R2 = 1.0 - ss_res / ss_tot

    return m, b, stderr_m, R2, chi2_red, N

def ols_fit(t, y_m):
    """Unweighted OLS slope stderr & R^2 as a cross-check."""
    t = np.asarray(t, float)
    y = np.asarray(y_m, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    N = len(t)
    if N < 2:
        return np.nan, np.nan, np.nan, N
    # Closed-form OLS
    m, b = np.polyfit(t, y, 1)
    yhat = m * t + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) or 1e-12
    R2 = 1.0 - ss_res / ss_tot
    # stderr of slope
    sy = np.sqrt(ss_res / max(N - 2, 1))
    Sxx = np.sum((t - np.mean(t))**2)
    stderr_m = sy / np.sqrt(Sxx) if Sxx > 0 else np.nan
    return m, stderr_m, R2, N

def analyze_positions(positions_csv="droplet_positions.csv", out_csv="droplet_velocities_final.csv"):
    df = pd.read_csv(positions_csv, na_values=["#NV", "#N/A", "NaN", "nan", "", " "])
    time_col = df.columns[0]
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    t_all = df[time_col].to_numpy(float)

    # Coerce position columns
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Group columns by droplet id and direction
    patt = re.compile(r"^\s*(\d+)\s*([udUD])\s*$")
    by_drop = {}
    for c in df.columns[1:]:
        m = patt.match(c)
        if not m:
            continue
        did = int(m.group(1))
        direc = m.group(2).lower()
        by_drop.setdefault(did, {})[direc] = c

    rows = []
    for did in sorted(by_drop.keys()):
        rec = {
            "droplet_number": did,
            # DOWN (WLS primary)
            "v_down_mps_wls": np.nan, "v_down_stderr_wls": np.nan,
            "R2_down": np.nan, "chi2red_down": np.nan, "N_down": 0,
            # UP (WLS primary)
            "v_up_mps_wls": np.nan, "v_up_stderr_wls": np.nan,
            "R2_up": np.nan, "chi2red_up": np.nan, "N_up": 0,
            # OLS (diagnostic)
            "v_down_mps_ols": np.nan, "v_down_stderr_ols": np.nan,
            "v_up_mps_ols": np.nan,   "v_up_stderr_ols": np.nan
        }

        # Helper to process one direction
        def process_direction(colname, rising=False):
            if colname not in df.columns:
                return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan)
            # Pre-clean: remove stuck points
            y_px_all = df[colname].to_numpy(float)
            t_clean, y_px_clean = remove_stuck_points(t_all, y_px_all, STUCK_THRESHOLD)

            # Find steady slice on cleaned series
            sl = robust_steady_slice(t_clean, y_px_clean)
            if sl.stop - sl.start < MIN_POINTS:
                return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan)

            tt = t_clean[sl]
            yy_px = y_px_clean[sl]
            yy_m  = _px_to_m(yy_px)
            sig_m = _sigma_y_m(yy_px)

            # WLS primary
            m_wls, b_wls, se_wls, R2, chi2r, N = wls_fit(tt, yy_m, sig_m)
            if np.isnan(m_wls):
                return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan)

            # OLS diag
            m_ols, se_ols, R2_ols, _ = ols_fit(tt, yy_m)

            # Report positive magnitude
            v_wls = abs(m_wls)
            v_ols = abs(m_ols) if np.isfinite(m_ols) else np.nan

            return (v_wls, se_wls, R2, chi2r, N, v_ols, se_ols)

        # DOWN
        if "d" in by_drop[did]:
            v, se, R2, chi2r, N, v_ols, se_ols = process_direction(by_drop[did]["d"], rising=False)
            rec.update({
                "v_down_mps_wls": v, "v_down_stderr_wls": se,
                "R2_down": R2, "chi2red_down": chi2r, "N_down": N,
                "v_down_mps_ols": v_ols, "v_down_stderr_ols": se_ols
            })

        # UP
        if "u" in by_drop[did]:
            v, se, R2, chi2r, N, v_ols, se_ols = process_direction(by_drop[did]["u"], rising=True)
            rec.update({
                "v_up_mps_wls": v, "v_up_stderr_wls": se,
                "R2_up": R2, "chi2red_up": chi2r, "N_up": N,
                "v_up_mps_ols": v_ols, "v_up_stderr_ols": se_ols
            })

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("droplet_number")
    # Quality flags based on R^2 only (diagnostic): you can add chi2 gates if you like (e.g., 0.3 < χ²_red < 3)
    out["ok_down"] = (out["N_down"] >= MIN_POINTS) & (out["R2_down"] >= R2_MIN)
    out["ok_up"]   = (out["N_up"]   >= MIN_POINTS) & (out["R2_up"]   >= R2_MIN)

    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(out)} rows.")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    analyze_positions("droplet_positions.csv", "droplet_velocities_final1.csv")
