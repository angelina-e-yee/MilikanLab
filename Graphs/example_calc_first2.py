#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import linregress
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

PX_PER_MM = 520.0   # calibration
MIN_POINTS = 12
R2_MIN = 0.98
MAD_K = 3.5

def px_to_m(y_px):
    return (y_px / PX_PER_MM) / 1000.0  # mm -> m

def robust_steady_slice(t, y):
    """Return slice(start, end) for the longest steady segment, or empty slice."""
    t = np.asarray(t, float)
    y = pd.to_numeric(pd.Series(y), errors='coerce').to_numpy()

    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(y) < MIN_POINTS:
        return slice(0,0)

    # Trim start/end plateaus (no motion) via difference threshold (in pixels)
    dy = np.diff(y, prepend=y[0])
    moving = np.abs(dy) >= 1.0  # >= 1 pixel change
    if not np.any(moving):
        return slice(0,0)
    i0 = np.argmax(moving)                # first True
    i1 = len(moving) - 1 - np.argmax(moving[::-1])  # last True
    t, y = t[i0:i1+1], y[i0:i1+1]
    offset = i0

    if len(y) < MIN_POINTS:
        return slice(0,0)

    # Robust derivative filter: keep where velocity ~ median within MAD_K
    v = np.gradient(y, t)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-12
    keep = np.abs(v - med) <= MAD_K * 1.4826 * mad
    if not np.any(keep):
        return slice(0,0)

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
        return slice(0,0)

    return slice(offset + best_start, offset + best_start + best_len)

def ols_fit(t, y):
    """Return (m, b, stderr_m, R2, N, sy, var_t) and the components used for pretty-printing."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    N = len(t)
    if N < 2:
        return np.nan, np.nan, np.nan, np.nan, N, np.nan, np.nan, {}

    # Closed-form OLS (for pretty-print)
    S_t = np.sum(t)
    S_y = np.sum(y)
    S_tt = np.sum(t*t)
    S_ty = np.sum(t*y)

    denom = N*S_tt - S_t**2
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan, N, np.nan, np.nan, {}

    m = (N*S_ty - S_t*S_y) / denom
    b = (S_y - m*S_t) / N
    yhat = m*t + b
    resid = y - yhat
    SS_res = np.sum(resid**2)
    SS_tot = np.sum((y - np.mean(y))**2) or 1e-12
    R2 = 1.0 - SS_res/SS_tot
    # Residual std (unbiased)
    sy = np.sqrt(SS_res / (N - 2)) if N > 2 else np.nan
    var_t = np.var(t, ddof=0)  # population variance

    if HAVE_SCIPY:
        # Cross-check stderr from scipy
        res = linregress(t, y)
        stderr_m = res.stderr
        # Use scipy's R^2 for consistency (should match)
        R2 = res.rvalue**2
    else:
        # Standard error of slope using classic formula:
        # stderr_m = sy / sqrt( sum( (t - mean(t))^2 ) )
        Sxx = np.sum((t - np.mean(t))**2)
        stderr_m = sy / np.sqrt(Sxx) if Sxx > 0 else np.nan

    parts = {
        "N": N, "S_t": S_t, "S_y": S_y, "S_tt": S_tt, "S_ty": S_ty,
        "denom": denom, "SS_res": SS_res, "SS_tot": SS_tot, "sy": sy, "var_t": var_t,
        "mean_t": np.mean(t), "Sxx": np.sum((t - np.mean(t))**2)
    }
    return m, b, stderr_m, R2, N, sy, var_t, parts

def analyze_first_two_columns(positions_csv="droplet_positions.csv"):
    df = pd.read_csv(positions_csv, na_values=["#NV","#N/A","NaN","nan",""," "])
    # Coerce all non-time columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    time_col = df.columns[0]
    t = df[time_col].to_numpy(dtype=float)

    # pick the first two data columns after time
    data_cols = [c for c in df.columns[1:]]
    if len(data_cols) == 0:
        print("No data columns found.")
        return
    take = data_cols[:2]

    results = []
    for idx, col in enumerate(take, start=1):
        y_px = df[col].to_numpy(dtype=float)
        sl = robust_steady_slice(t, y_px)
        if sl.stop - sl.start < MIN_POINTS:
            print(f"[{col}] No adequate steady segment found.")
            continue

        tt = t[sl]
        yy_m = px_to_m(y_px[sl])

        m, b, stderr_m, R2, N, sy, var_t, parts = ols_fit(tt, yy_m)

        # Pretty print formulas with numbers
        print("\n" + "="*80)
        print(f"Column: {col}")
        print(f"Steady segment indices: [{sl.start}:{sl.stop})  -> N = {N}")
        print("\nOLS slope (velocity) m using closed-form formula:")
        print("  m = [N * Σ(t_i y_i) - (Σ t_i)(Σ y_i)] / [N * Σ(t_i^2) - (Σ t_i)^2]")
        print(f"    = [{parts['N']} * {parts['S_ty']:.6e} - ({parts['S_t']:.6e})({parts['S_y']:.6e})] / "
              f"[{parts['N']} * {parts['S_tt']:.6e} - ({parts['S_t']:.6e})^2]")
        print(f"    = {m:.6e}  (m/s)")

        print("\nIntercept b:")
        print("  b = [Σ y_i - m Σ t_i] / N")
        print(f"    = [{parts['S_y']:.6e} - ({m:.6e})({parts['S_t']:.6e})] / {parts['N']} = {b:.6e}  (m)")

        print("\nGoodness of fit (R^2):")
        print("  R^2 = 1 - SS_res / SS_tot")
        print(f"    SS_res = Σ(y_i - ŷ_i)^2 = {parts['SS_res']:.6e}")
        print(f"    SS_tot = Σ(y_i - ȳ)^2   = {parts['SS_tot']:.6e}")
        print(f"  => R^2 = {R2:.6f}")

        print("\nStandard error of slope (uncertainty in velocity):")
        print("  σ_m = s_y / sqrt(Σ (t_i - t̄)^2)   (or equivalently s_y * sqrt(1 / (N Var(t))) for evenly spaced t)")
        print(f"    s_y (residual std) = {sy:.6e} (m)")
        print(f"    Σ (t_i - t̄)^2 = {parts['Sxx']:.6e} (s^2)")
        if parts['Sxx'] > 0:
            sigma_alt = sy / np.sqrt(parts['Sxx'])
        else:
            sigma_alt = np.nan
        print(f"  => σ_m (slope stderr) = {stderr_m:.6e} (m/s)   [alt calc: {sigma_alt:.6e}]")

        # Plot: raw data, steady region, fit
        fig = plt.figure(figsize=(7,4.5))
        plt.plot(t, px_to_m(y_px), marker='o', linestyle='None', ms=3, label='all data')
        plt.plot(tt, yy_m, marker='o', linestyle='None', ms=3, label='steady region')
        yhat = m*tt + b
        plt.plot(tt, yhat, linewidth=2, label='fit on steady region')
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.title(f"{col}: v = {m:.3e} m/s,  σ_v = {stderr_m:.1e} m/s,  R² = {R2:.3f}, N={N}")
        plt.legend()
        outpng = f"example_{col}_fit.png"
        plt.tight_layout()
        plt.savefig(outpng, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {outpng}")

        results.append({
            "column": col,
            "v_term_mps": abs(m),
            "stderr_mps": stderr_m,
            "R2": R2,
            "N": N
        })

    if results:
        out = pd.DataFrame(results)
        out_csv = "example_first2_results.csv"
        out.to_csv(out_csv, index=False)
        print("\nSummary written to:", out_csv)
        print(out.to_string(index=False))
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    csv_path = "droplet_positions.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    analyze_first_two_columns(csv_path)
