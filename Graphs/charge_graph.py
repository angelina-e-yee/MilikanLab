# plot_charge_vs_index_colored.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_charge_csv(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)

    # Keep only numeric droplet rows (drop the GCD summary etc.)
    df = df[pd.to_numeric(df.get("droplet_number", np.nan), errors="coerce").notna()].copy()
    df["droplet_number"] = df["droplet_number"].astype(int)

    # Match charge & uncertainty column names flexibly
    q_cols = ["q_method1", "q_method2", "q_C_method1_slip", "q_C_method2_slip", "q_C"]
    se_cols = ["q_unc_from_v", "q_stderr_C_from_v", "q_sigma", "sigma_q"]

    q_col = next((c for c in q_cols if c in df.columns), None)
    se_col = next((c for c in se_cols if c in df.columns), None)

    if q_col is None:
        raise ValueError(f"Could not find a charge column in {path}")
    if se_col is None:
        df["__sigma_q__"] = 0.0
        se_col = "__sigma_q__"

    ok_col = "ok_down" if "ok_down" in df.columns else None
    if ok_col:
        df = df[df[ok_col] == True]

    return df[["droplet_number", q_col, se_col]].rename(
        columns={q_col: "q", se_col: "sigma_q"}
    ).sort_values("droplet_number")

def main():
    files = [
        ("Method 1 (Down)", "q_method1_V3.csv", "#1f77b4"),
        ("Method 2 (Rise/Fall)", "q_method2_V3.csv", "#ff7f0e"),
    ]

    plt.figure(figsize=(10, 5))
    found = False

    for i, (label, path, color) in enumerate(files):
        df = load_charge_csv(path)
        if df is None or df.empty:
            continue
        found = True
        x = df["droplet_number"].to_numpy(float)
        y = df["q"].to_numpy(float)
        yerr = df["sigma_q"].to_numpy(float)

        plt.errorbar(
            x, y, yerr=yerr,
            fmt="o", ms=4, capsize=3,
            color=color, ecolor=color, alpha=0.85,
            label=label
        )

    if not found:
        raise SystemExit("No charge CSVs found in current directory.")

    plt.xlabel("Droplet index")
    plt.ylabel("Charge q (C)")
    plt.title("Measured Charge vs. Droplet Index")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("charge_vs_index_colored.png", dpi=300)
    plt.show()
    print("Saved figure: charge_vs_index_colored.png")

if __name__ == "__main__":
    main()
