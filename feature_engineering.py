"""
feature_engineering.py
======================
Adds missing feature groups to node_features.csv before ML training.
Mhango, S.B. and Sambito, M. (2026)

This script fills the gaps between what dataset_generator.py already produces
and what the concept note specifies for the full feature set:

  Group 1 -- flow_diversion_fraction  (from weir/orifice geometry in SWMM inp)
  Group 3 -- mean_wastewater_flux     (Prior C -- 50 Monte Carlo runs)
             mean_contaminant_flux    (Prior D -- 50 Monte Carlo runs)
             contaminant_flux_std     (std across the same 50 runs)

Groups 2 and 4 are already present in node_features.csv from dataset_generator.py.

Usage:
  python feature_engineering.py
  python feature_engineering.py --node_features ./output/node_features.csv \\
                                --raw_scenarios  ./output/raw_scenarios.csv \\
                                --inp            Example8.inp \\
                                --output         ./output/node_features_full.csv

The output file is a drop-in replacement for node_features.csv.
Pass it to train_models.py via --node_features.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- flow_diversion_fraction
# ══════════════════════════════════════════════════════════════════════════════

def compute_flow_diversion_fraction(inp_file, node_list):
    """
    Computes the flow diversion fraction for each node.

    For regulator nodes (JI6, JI7, JI8, JI9) this is the fraction of combined
    flow that is passively diverted to the interceptor at design conditions,
    estimated from weir and orifice geometry in the SWMM input file using:

      Weir:    Q_weir   = C_w * L * H^1.5    (US customary, ft³/s)
      Orifice: Q_orifice = C_d * A * sqrt(2g*H)

    where H is taken as 0.5 × crest_height as a representative operating head.

    Nodes with no passive regulator (J-nodes, interceptor nodes downstream of
    the pump) receive fraction = 0.0.

    The concept note identifies this feature as the limiting factor for
    sensor placement at JI10–JI13 (Sambito et al. 2020, Section 4).

    Parameters
    ----------
    inp_file : str
        Path to SWMM .inp file.
    node_list : list of str
        All node IDs in network order.

    Returns
    -------
    dict : node_id -> flow_diversion_fraction
    """
    # Parse weirs and orifices
    weirs    = {}
    orifices = {}
    xsect    = {}
    section  = None

    with open(inp_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";;"):
                continue
            if s.startswith("["):
                section = s.strip("[]").upper()
                continue
            parts = s.split()
            if not parts:
                continue

            if section == "WEIRS" and len(parts) >= 6:
                # Name  From  To  Type  CrestHt  Qcoeff
                weirs[parts[0]] = {
                    "from":    parts[1],
                    "to":      parts[2],
                    "crest":   float(parts[4]),
                    "Cw":      float(parts[5]),
                }
            elif section == "ORIFICES" and len(parts) >= 6:
                # Name  From  To  Type  Offset  Qcoeff
                orifices[parts[0]] = {
                    "from": parts[1],
                    "to":   parts[2],
                    "Cd":   float(parts[5]),
                }
            elif section == "XSECTIONS" and len(parts) >= 3:
                xsect[parts[0]] = {
                    "shape": parts[1],
                    "geom1": float(parts[2]),
                    "geom2": float(parts[3]) if len(parts) > 3 else 0.0,
                }

    g_ft = 32.174   # ft/s²

    # Map from_node -> list of (weir, orifice) tuples at that node
    node_fracs = {n: 0.0 for n in node_list}

    # Group weirs by their from_node
    weirs_by_node = {}
    for wname, w in weirs.items():
        weirs_by_node.setdefault(w["from"], []).append((wname, w))

    # Group orifices by from_node
    orif_by_node = {}
    for oname, o in orifices.items():
        orif_by_node.setdefault(o["from"], []).append((oname, o))

    # Compute diversion fraction for each regulator node
    for node in node_list:
        node_weirs  = weirs_by_node.get(node, [])
        node_orifices = orif_by_node.get(node, [])

        if not node_weirs:
            continue    # no regulator at this node

        # Use minimum crest height as the reference; H = 0.5 * crest
        min_crest = min(w["crest"] for _, w in node_weirs)
        H = 0.5 * min_crest

        Q_overflow = 0.0    # to stream (loss from interceptor system)
        Q_intercept = 0.0   # to interceptor

        for wname, w in node_weirs:
            xs = xsect.get(wname, {"shape": "RECT_OPEN", "geom1": 1.0})
            L  = xs["geom1"]    # width (ft)
            Q_overflow += w["Cw"] * L * (H ** 1.5)

        for oname, o in node_orifices:
            xs = xsect.get(oname, {"shape": "CIRCULAR", "geom1": 0.5})
            D  = xs["geom1"]
            A  = 3.14159 * (D / 2) ** 2
            H_orif = H + min_crest    # total head above orifice
            Q_intercept += o["Cd"] * A * (2 * g_ft * H_orif) ** 0.5

        total = Q_overflow + Q_intercept
        frac  = (Q_intercept / total) if total > 0 else 0.0
        node_fracs[node] = round(frac, 6)

    return node_fracs


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 -- Monte Carlo Prior quantities (Prior C and Prior D)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mc_prior_features(raw_scenarios_df, node_list, n_mc=50, seed=42):
    """
    Approximates Prior C and Prior D quantities from the scenario dataset,
    following the methodology of Sambito et al. (2020, Section 4.2).

    In the original v1.0 framework, Prior C and D are derived from 50
    dedicated Monte Carlo runs. Here we approximate them from the existing
    scenario dataset by sampling n_mc scenarios uniformly (without replacement)
    to match the 50-run budget described in the concept note.

    Prior C -- mean_wastewater_flux:
        Mean total flow (m³/s) at the node across the sampled scenarios.
        In v1.0 this is the mean volume of wastewater passing through the node.
        Approximated here as mean_flow_m3s already in node_features.csv, but
        computed separately from the MC sample for consistency.

    Prior D -- mean_contaminant_flux:
        Mean peak concentration at the node across the sampled scenarios,
        weighted by detected=1. This captures mean contaminant mass flux
        through the node and is the most informative prior in v1.0.

    contaminant_flux_std:
        Standard deviation of peak_conc across the MC sample, capturing
        how consistently the node intercepts contamination regardless of
        source location.

    Parameters
    ----------
    raw_scenarios_df : pd.DataFrame
        Full raw_scenarios.csv output from dataset_generator.py.
    node_list : list of str
        All node IDs.
    n_mc : int
        Number of scenarios to sample for MC approximation (default 50).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with columns: node_id, mean_wastewater_flux,
                                mean_contaminant_flux, contaminant_flux_std
    """
    rng = np.random.RandomState(seed)
    all_scen_ids = raw_scenarios_df["scen_id"].unique()

    n_sample = min(n_mc, len(all_scen_ids))
    if n_sample < n_mc:
        print(f"  NOTE: Only {n_sample} scenarios available for MC prior "
              f"(requested {n_mc}). Run more scenarios for better approximation.")

    sampled_ids = rng.choice(all_scen_ids, size=n_sample, replace=False)
    mc_df = raw_scenarios_df[raw_scenarios_df["scen_id"].isin(sampled_ids)].copy()

    rows = []
    for node in node_list:
        node_rows = mc_df[mc_df["node_id"] == node]
        if len(node_rows) == 0:
            rows.append({
                "node_id":               node,
                "mean_wastewater_flux":  0.0,
                "mean_contaminant_flux": 0.0,
                "contaminant_flux_std":  0.0,
            })
            continue

        mean_flow  = node_rows["mean_flow_m3s"].mean()
        mean_conc  = node_rows["peak_conc"].mean()
        std_conc   = node_rows["peak_conc"].std()

        rows.append({
            "node_id":               node,
            "mean_wastewater_flux":  round(float(mean_flow), 8),
            "mean_contaminant_flux": round(float(mean_conc), 4),
            "contaminant_flux_std":  round(float(std_conc), 4) if not np.isnan(std_conc) else 0.0,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(node_features_path, raw_scenarios_path, inp_file, output_path):

    print("=" * 60)
    print("Feature Engineering -- Hybrid AI Sensor Placement")
    print("Mhango & Sambito (2026)")
    print("=" * 60)

    # ── Load existing features ─────────────────────────────────────────────────
    print("\n[1/4] Loading node_features.csv ...")
    nf = pd.read_csv(node_features_path).fillna(0)
    node_list = nf["node_id"].tolist()
    print(f"      Nodes: {len(nf)}  |  Columns: {list(nf.columns)}")

    print("\n[2/4] Loading raw_scenarios.csv ...")
    rs = pd.read_csv(raw_scenarios_path)
    print(f"      Rows: {len(rs):,}  |  Unique scenarios: {rs['scen_id'].nunique()}")

    print("\n[2.5/4] Recomputing dynamic features from raw_scenarios to prevent data leakage ...")
    grp = rs.groupby('node_id')
    node_agg = pd.DataFrame({
        'detection_freq':       grp['detected'].mean().round(4),
        'peak_conc_mean':       grp['peak_conc'].mean().round(4),
        'peak_conc_std':        grp['peak_conc'].std().round(4),
        'time_to_peak_mean':    grp['t_peak_min'].mean().round(2),
        'mean_flow_m3s':        grp['mean_flow_m3s'].mean().round(6),
        'n_scenarios_detected': grp['detected'].sum().astype(int),
    }).reset_index()
    
    # Drop old dynamic features and merge new ones
    dyn_cols = ['detection_freq', 'peak_conc_mean', 'peak_conc_std', 'time_to_peak_mean', 'mean_flow_m3s', 'n_scenarios_detected']
    nf = nf.drop(columns=[c for c in dyn_cols if c in nf.columns])
    nf = nf.merge(node_agg, on='node_id', how='left').fillna(0)


    # ── Group 1: flow_diversion_fraction ──────────────────────────────────────
    print("\n[3/4] Computing Group 1 -- flow_diversion_fraction ...")
    frac_dict = compute_flow_diversion_fraction(inp_file, node_list)

    n_nonzero = sum(1 for v in frac_dict.values() if v > 0)
    print(f"      Regulator nodes with non-zero fraction: {n_nonzero}")
    for node, frac in frac_dict.items():
        if frac > 0:
            print(f"        {node}: {frac:.4f}")

    nf["flow_diversion_fraction"] = nf["node_id"].map(frac_dict).fillna(0.0)

    # ── Group 3: MC Prior quantities ───────────────────────────────────────────
    print("\n[4/4] Computing Group 3 -- Monte Carlo Prior features (n=50 sample) ...")
    mc_features = compute_mc_prior_features(rs, node_list, n_mc=50, seed=42)

    nf = nf.merge(mc_features, on="node_id", how="left")

    # ── Check and report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE SET SUMMARY")
    print("=" * 60)

    feature_groups = {
        "Group 1 -- Static Topology": [
            "topo_depth", "n_upstream_nodes", "betweenness",
            "downstream_paths", "node_type_code", "flow_diversion_fraction",
        ],
        "Group 2 -- Prior Contamination Prob": [
            "is_high_risk", "prior_contam_prob",
        ],
        "Group 3 -- Bayesian Prior Quantities": [
            "mean_wastewater_flux", "mean_contaminant_flux", "contaminant_flux_std",
        ],
        "Group 4 -- Dynamic Simulation": [
            "peak_conc_mean", "peak_conc_std", "time_to_peak_mean",
            "mean_flow_m3s", "mean_vel_ms", "detection_freq",
        ],
    }

    total_features = 0
    for group, cols in feature_groups.items():
        present   = [c for c in cols if c in nf.columns]
        missing   = [c for c in cols if c not in nf.columns]
        total_features += len(present)
        status = "OK" if not missing else f"MISSING: {missing}"
        print(f"\n  {group}")
        print(f"    Features: {present}")
        print(f"    Status:   {status}")

    print(f"\n  Total ML input features: {total_features}")
    print(f"  Target:                  detection_freq")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    nf.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(nf)} rows, {len(nf.columns)} columns)")
    print("\nNext step: python train_models.py --node_features", output_path)

    return nf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add missing feature groups to node_features.csv"
    )
    parser.add_argument("--node_features", default="./output/node_features.csv")
    parser.add_argument("--raw_scenarios",  default="./output/raw_scenarios.csv")
    parser.add_argument("--model_path",     default="./dataset/Examples/Example8.inp")
    parser.add_argument("--output",         default="./output/node_features_full.csv")
    args = parser.parse_args()

    main(args.node_features, args.raw_scenarios, args.model_path, args.output)