# plot_velocity_windows.py
# Visualize how steady windows are selected and how linear fits are done.
# Produces one PNG per droplet/direction in figs_velocity_windows/.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
POSITIONS_CSV   = "droplet_positions.csv"   # time in col 0; other cols like '1d','1u',...
FIGS_DIR        = "figs_velocity_windows"

# Calibration / noise model (match your velocity script)
PX_PER_MM       = 520.0        # px/mm
SIG_C_PX_PER_MM = 1.0          # px/mm
SIG_POS_PX      = 0.5          # px
QUANT_PX        = 1.0/np.sqrt(12.0)  # ~0.289 px

# Steady-window & robust fit params (match your velocity script)
MIN_POINTS      = 12
PLATEAU_PX      = 1.0
MAD_K           = 3.5
STUCK_THRESHOLD = 10

WINDOW_MIN      = 12
WINDOW_MAX      = 200
REFINE_RADIUS   = 6

HUBER_K         = 1.345
# ---------------------------


def _px_to_m(x_px):
    return (np.asarray(x_px, float) / PX_PER_MM) / 1000.0

def _sigma_y_m(y_px):
    y_px = np.asarray(y_px, float)
    sig_px = np.sqrt(SIG_POS_PX**2 + QUANT_PX**2)
    term1 = (sig_px / PX_PER_MM)**2
    term2 = (y_px * SIG_C_PX_PER_MM / (PX_PER_MM**2))**2
    sigma_mm = np.sqrt(term1 + term2)
    return sigma_mm / 1000.0

def remove_stuck_points(t, y_px, threshold=STUCK_THRESHOLD):
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
    t = np.asarray(t, float)
    y = pd.to_numeric(pd.Series(y_px), errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(y) < MIN_POINTS:
        return slice(0, 0)

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

    v = np.gradient(y, t)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-12
    keep = np.abs(v - med) <= MAD_K * 1.4826 * mad
    if not np.any(keep):
        return slice(0, 0)

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

def _huber_weights(res, scale, k=HUBER_K):
    t = np.abs(res) / (k * (scale + 1e-18))
    w = np.ones_like(res)
    mask = t > 1.0
    w[mask] = 1.0 / t[mask]
    return w

def wls_huber_fit(t, y_m, sigma_m, max_iter=10):
    t = np.asarray(t, float); y = np.asarray(y_m, float); s = np.asarray(sigma_m, float)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    t, y, s = t[mask], y[mask], s[mask]
    N = len(t)
    if N < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, N

    w = 1.0 / (s**2)
    for _ in range(max_iter):
        W = np.sum(w)
        tbar = np.sum(w * t) / W
        ybar = np.sum(w * y) / W
        Sxx = np.sum(w * (t - tbar)**2)
        Sxy = np.sum(w * (t - tbar) * (y - ybar))
        if Sxx <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, N
        m = Sxy / Sxx
        b = ybar - m * tbar
        yhat = m * t + b
        r = y - yhat

        med = np.median(r)
        mad = np.median(np.abs(r - med)) or 1e-12
        w = (1.0 / (s**2)) * _huber_weights(r, mad)

    yhat = m * t + b
    r = y - yhat
    dof = max(N - 2, 1)
    chi2 = np.sum((r / s)**2)
    chi2_red = chi2 / dof

    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) or 1e-12
    R2 = 1.0 - ss_res / ss_tot

    W = np.sum(w)
    tbar = np.sum(w * t) / W
    Sxx = np.sum(w * (t - tbar)**2)
    s2 = chi2_red * np.mean(s**2)
    stderr_m = np.sqrt(s2 / (Sxx + 1e-18))
    return m, b, stderr_m, R2, chi2_red, N

def refine_window(t, y_px, base_slice, min_len=WINDOW_MIN, max_len=WINDOW_MAX):
    i0 = max(0, base_slice.start); i1 = min(len(t), base_slice.stop)
    best = (np.inf, slice(i0, i1))
    for dl in range(-REFINE_RADIUS, REFINE_RADIUS + 1):
        for dr in range(-REFINE_RADIUS, REFINE_RADIUS + 1):
            a = max(0, i0 + dl)
            b = min(len(t), i1 + dr)
            if b - a < min_len or b - a > max_len:
                continue
            tt = t[a:b]
            yy_px = y_px[a:b]
            yy_m  = _px_to_m(yy_px)
            sig_m = _sigma_y_m(yy_px)
            m, _, _, _, chi2r, N = wls_huber_fit(tt, yy_m, sig_m)
            if not np.isfinite(m):
                continue
            score = abs(chi2r - 1.0) + 0.01 / np.sqrt(max(N,1))
            if score < best[0]:
                best = (score, slice(a, b))
    return best[1]

def plot_fit(t_clean, y_px_clean, sl, m, b, stats, droplet, direction, outdir=FIGS_DIR):
    os.makedirs(outdir, exist_ok=True)
    y_all_m = _px_to_m(y_px_clean)
    tt = t_clean[sl]
    yy_m = y_all_m[sl]
    yhat = m * tt + b

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(t_clean, y_all_m, color="0.6", lw=1, alpha=0.35, label="cleaned track")
    ax.plot(tt, yy_m, ".", ms=4, label="steady window")
    ax.plot(tt, yhat, lw=2, label="robust WLS fit")

    v, se, R2, chi2r, N = stats
    ax.set_title(f"Droplet {droplet}{direction}: steady window & linear fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(loc="best", framealpha=0.9)

    txt = (f"v = {abs(v):.3e} m/s ± {se:.1e}\n"
           f"R² = {R2:.4f},  χ²ν = {chi2r:.2f},  N = {N}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    fn = os.path.join(outdir, f"d{droplet}_{direction}_fit.png")
    fig.tight_layout()
    fig.savefig(fn, dpi=220)
    plt.close(fig)

def main():
    df = pd.read_csv(POSITIONS_CSV, na_values=["#NV", "#N/A", "NaN", "nan", "", " "])
    time_col = df.columns[0]
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    t_all = df[time_col].to_numpy(float)

    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    patt = re.compile(r"^\s*(\d+)\s*([udUD])\s*$")
    by_drop = {}
    for c in df.columns[1:]:
        m = patt.match(c)
        if not m:
            continue
        did = int(m.group(1))
        direc = m.group(2).lower()
        by_drop.setdefault(did, {})[direc] = c

    # ⬇️ Only process droplet 4, direction 'd'
    did = 4
    direction = "d"

    if did not in by_drop or direction not in by_drop[did]:
        print("Droplet 4d not found in data.")
        return

    col = by_drop[did][direction]
    y_px_all = df[col].to_numpy(float)
    t_clean, y_px_clean = remove_stuck_points(t_all, y_px_all, STUCK_THRESHOLD)

    sl0 = robust_steady_slice(t_clean, y_px_clean)
    if sl0.stop - sl0.start < MIN_POINTS:
        print("No valid steady slice for 4d.")
        return

    sl = refine_window(t_clean, y_px_clean, sl0, WINDOW_MIN, WINDOW_MAX)
    if sl.stop - sl.start < MIN_POINTS:
        print("No refined window for 4d.")
        return

    tt = t_clean[sl]
    yy_m = _px_to_m(y_px_clean[sl])
    sig_m = _sigma_y_m(y_px_clean[sl])

    m, b, se, R2, chi2r, N = wls_huber_fit(tt, yy_m, sig_m)
    if not np.isfinite(m):
        print("Invalid fit for 4d.")
        return

    v = abs(m)

    # ⬇️ Text label now uses “X²” instead of “χ²ν”
    os.makedirs(FIGS_DIR, exist_ok=True)
    y_all_m = _px_to_m(y_px_clean)
    yhat = m * tt + b

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(t_clean, y_all_m, color="0.6", lw=1, alpha=0.35, label="cleaned track")
    ax.plot(tt, yy_m, ".", ms=4, label="steady window")
    ax.plot(tt, yhat, lw=2, label="robust WLS fit")

    ax.set_title(f"Droplet {did}{direction}: steady window & linear fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(loc="best", framealpha=0.9)

    txt = (f"v = {v:.4e} m/s ± {se:.1e}\n"
           f"R² = {R2:.4f},  X² = {chi2r:.2f},  N = {N}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))

    fn = os.path.join(FIGS_DIR, "droplet4d_fit.png")
    fig.tight_layout()
    fig.savefig(fn, dpi=220)
    plt.close(fig)
    print(f"Saved figure: {fn}")


if __name__ == "__main__":
    main()
