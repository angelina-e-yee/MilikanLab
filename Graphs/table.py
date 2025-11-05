import pandas as pd
import numpy as np

# --- constants ---
e_true = 1.602e-19  # elementary charge (C)

def load_charges(csv):
    """Read charge data (q column) from CSV."""
    df = pd.read_csv(csv)
    qcol = next(c for c in df.columns if c.lower().startswith("q_"))
    q = pd.to_numeric(df[qcol], errors="coerce").dropna().to_numpy()
    q = q[q > 0]
    return q

def analyze_quantization(q, method_name):
    # find nearest integer multiple of e
    n = np.round(q / e_true)
    n[n == 0] = 1  # avoid divide by zero
    q_single = q / n  # reduce to "single-charge equivalent"
    
    # compute stats
    mean_q = np.mean(q)
    std_q = np.std(q)
    median_q = np.median(q)
    
    # count multiplicities
    unique, counts = np.unique(n, return_counts=True)
    multiplicities = dict(zip(unique.astype(int), counts))
    
    one_e_count = multiplicities.get(1, 0)
    total = len(q)
    share_1e = 100 * one_e_count / total if total else np.nan

    # assemble summary
    print(f"\n--- {method_name} ---")
    print(f"Mean q: ({mean_q:.3e} ± {std_q/np.sqrt(total):.3e}) C")
    print(f"Median q: {median_q:.3e} C")
    print(f"Multiplicity counts: ", end="")
    print(", ".join([f"{int(k)}e: {v}" for k, v in multiplicities.items()]))
    print(f"Share of 1e: {share_1e:.1f}% ({one_e_count}/{total} droplets)")

    return {
        "Method": method_name,
        "Mean": mean_q,
        "Mean_stderr": std_q/np.sqrt(total),
        "Median": median_q,
        "Counts": multiplicities,
        "Share_1e": share_1e
    }

# --- run on both methods ---
m1 = analyze_quantization(load_charges("q_method1_V3.csv"), "Method 1 (Down)")
m2 = analyze_quantization(load_charges("q_method2_V3.csv"), "Method 2 (Rise/Fall)")

# optional: export summary table
pd.DataFrame([m1, m2]).to_csv("charge_quantization_summary.csv", index=False)
