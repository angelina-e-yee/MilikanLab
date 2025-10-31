import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PX_PER_MM = 520.0  # calibration (px/mm)

def px_to_m(px):
    return (px / PX_PER_MM) / 1000.0  # px -> mm -> m

def main(positions_csv="droplet_positions.csv", out_png="drops_1_10_20_samecolor.png"):
    df = pd.read_csv(positions_csv, na_values=["#NV", "#N/A", "NaN", "nan", "", " "])
    time_col = df.columns[0]
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    droplets = [1, 10, 20]
    directions = ["d", "u"]

    # Assign consistent colours
    cmap = plt.get_cmap("tab10")
    colors = {1: cmap(0), 10: cmap(1), 20: cmap(2)}

    plt.figure(figsize=(10, 6))
    for did in droplets:
        for d in directions:
            col = f"{did}{d}"
            if col not in df.columns:
                continue
            y_px = pd.to_numeric(df[col], errors="coerce").to_numpy()
            mask = np.isfinite(t) & np.isfinite(y_px)
            if mask.sum() == 0:
                continue
            y_m = px_to_m(y_px[mask])
            linestyle = "-" if d == "d" else "--"  # solid for fall, dashed for rise
            plt.plot(
                t[mask],
                y_m,
                linestyle=linestyle,
                color=colors[did],
                linewidth=2,
                label=f"{did}{d}"
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Droplets 1, 10, and 20 – Position vs Time")
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
