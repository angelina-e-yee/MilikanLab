# plot_charge_histograms_methods.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_q(csv_path):
    df = pd.read_csv(csv_path)
    qcol = next(c for c in df.columns if c.lower().startswith("q_"))
    q = pd.to_numeric(df[qcol], errors="coerce").to_numpy()
    return q[np.isfinite(q) & (q > 0)]

# Load both datasets
q1 = load_q("q_method1_V3.csv")
q2 = load_q("q_method2_V3.csv")

# Convert to 1e-19 C for nicer axis scale
scale = 1e-19
q1_scaled = q1 / scale
q2_scaled = q2 / scale

# Shared bin edges for comparability
all_data = np.concatenate([q1_scaled, q2_scaled])
bins = np.linspace(np.min(all_data), np.max(all_data), 20)

plt.figure(figsize=(9,5))
plt.hist(q1_scaled, bins=bins, color="#4c78a8", edgecolor="black", alpha=0.6, label="Method 1 (Down)")
plt.hist(q2_scaled, bins=bins, color="#f58518", edgecolor="black", alpha=0.6, label="Method 2 (Rise/Fall)")

plt.xlabel(r"Charge ($\times 10^{-19}$ C)")
plt.ylabel("Frequency")
plt.title("Distribution of Measured Charges by Method")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("charge_hist_methods.png", dpi=300)
plt.show()
