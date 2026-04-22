"""
bdn_solver.py
=============
Bayesian Decision Network solver for optimal water quality sensor placement.
Mhango, S.B. and Sambito, M. (2026)

Implements the BDN framework of Sambito et al. (2020) with support for
ML-derived priors from train_models.py, enabling direct comparison between:

  Prior A -- Uniform
  Prior B -- Topology-based (1 / topo_depth)
  Prior C -- Wastewater flux (mean_wastewater_flux from 50 MC runs)
  Prior D -- Contaminant flux (mean_contaminant_flux, most informative v1.0 prior)
  ML priors -- XGBoost, LightGBM, MLP, GCN, GAT (from train_models.py)

Algorithm (greedy sequential sensor placement):
  1. Initialise source prior P(j) for all candidate source nodes j
  2. Build p_kj matrix from raw_scenarios: P(node k detects | source j)
  3. Greedy placement: for each sensor position s = 1, 2, ..., n_sensors:
       a. For each candidate observer node k not yet selected:
          Coverage(k) = sum_j P(j) * p_kj
       b. Place sensor at k* = argmax Coverage(k)
       c. Simulate a random contamination event from the scenario dataset
       d. Observe detection vector at placed sensors
       e. Update posterior: P(j) ∝ P(d | j) * P(j)
       f. Repeat Bayesian updates until convergence (max ΔP < epsilon)
  4. Evaluate: F1 (isolation likelihood) and F2 (detection reliability)
  5. Compare all prior types on: convergence speed, sensor placement, F1, F2

Inputs:
  raw_scenarios.csv          -- from dataset_generator.py
  node_features_full.csv     -- from feature_engineering.py
  priors/prior_*.csv         -- from train_models.py

Outputs (written to --output_dir, default ./bdn_output/):
  results/comparison_table.csv    -- all priors vs all metrics
  results/sensor_placements.csv   -- recommended sensors per prior
  results/convergence.csv         -- iterations to convergence per prior

Usage:
  python bdn_solver.py
  python bdn_solver.py --raw_scenarios ./output/raw_scenarios.csv \\
                       --node_features ./output/node_features_full.csv \\
                       --priors_dir    ./ml_output/priors \\
                       --n_sensors     3 \\
                       --output_dir    ./bdn_output
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED    = 42
EPSILON = 1e-4    # posterior convergence threshold
MAX_ITER = 200    # maximum Bayesian update iterations per sensor
rng = np.random.RandomState(SEED)

# Nodes excluded from sensor placement (not physical candidates)
EXCLUDE_SENSORS = {"O1", "O2", "Well"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(raw_scenarios_path, node_features_path):
    """
    Loads and validates the two input files.
    Returns node list, candidate source list, candidate sensor list,
    and the raw scenario DataFrame.
    """
    rs = pd.read_csv(raw_scenarios_path)
    rs["peak_conc"] = rs["peak_conc"].fillna(0)
    rs["t_peak_min"] = rs["t_peak_min"].fillna(-1)

    nf = pd.read_csv(node_features_path).fillna(0)
    node_list = nf["node_id"].tolist()

    if "node_type" in nf.columns:
        candidate_sensors = [
            n for n in node_list
            if n not in EXCLUDE_SENSORS
            and nf.loc[nf["node_id"] == n, "node_type"].values[0] in ("J", "JI", "aux")
        ]
    else:
        # Fallback: use node_type_code (0=J, 1=JI, 3=outfall)
        candidate_sensors = [
            n for n in node_list
            if n not in EXCLUDE_SENSORS
            and nf.loc[nf["node_id"] == n, "node_type_code"].values[0] in (0, 1)
        ] if "node_type_code" in nf.columns else [
            n for n in node_list if n not in EXCLUDE_SENSORS
        ]

    candidate_sources = rs["src_node"].unique().tolist()

    print(f"      Nodes total:       {len(node_list)}")
    print(f"      Candidate sensors: {len(candidate_sensors)}")
    print(f"      Candidate sources: {len(candidate_sources)}")
    print(f"      Scenario rows:     {len(rs):,}")
    print(f"      Unique scenarios:  {rs['scen_id'].nunique()}")

    return rs, nf, node_list, candidate_sources, candidate_sensors


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — p_kj DETECTION PROBABILITY MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_pkj_matrix(raw_df, candidate_sensors, candidate_sources):
    """
    Builds the detection probability matrix p_kj.

    p_kj = P(node k detects contamination | contamination source is j)
         = fraction of scenarios with source=j in which node k detected

    Returns
    -------
    p_kj : np.ndarray, shape (n_sensors, n_sources)
    sensor_order : list of str  (row order)
    source_order : list of str  (column order)
    """
    pivot = (
        raw_df.groupby(["node_id", "src_node"])["detected"]
        .mean()
        .unstack(fill_value=0.0)
    )

    # Align to our canonical orderings; fill missing with 0
    sensor_order = [s for s in candidate_sensors if s in pivot.index]
    source_order = [s for s in candidate_sources if s in pivot.columns]

    p_kj = pivot.reindex(index=sensor_order, columns=source_order, fill_value=0.0).values

    # Clip to [0, 1] and add small epsilon to avoid log(0) in likelihood
    p_kj = np.clip(p_kj, 1e-6, 1.0 - 1e-6)

    return p_kj, sensor_order, source_order


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRIOR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_v1_priors(nf, candidate_sources, source_order):
    """
    Constructs the four v1.0 baseline prior distributions over source nodes.
    Returns a dict: prior_name -> np.array of shape (n_sources,).
    All priors are normalised to sum to 1.
    """
    def normalise(v):
        v = np.array(v, dtype=float)
        v = np.clip(v, 0, None)
        return v / v.sum() if v.sum() > 0 else np.ones(len(v)) / len(v)

    # Index source nodes in the canonical source_order
    nf_src = nf.set_index("node_id")

    def src_feat(feat):
        return np.array([
            nf_src.loc[s, feat] if s in nf_src.index else 0.0
            for s in source_order
        ])

    priors = {}

    # Prior A: uniform
    priors["Prior_A_Uniform"] = normalise(np.ones(len(source_order)))

    # Prior B: topological -- weight ∝ 1/(topo_depth+1)
    depths = src_feat("topo_depth")
    priors["Prior_B_Topo"] = normalise(1.0 / (depths + 1.0))

    # Prior C: mean wastewater flux (proportional to mean_flow_m3s or
    # mean_wastewater_flux if available from feature_engineering.py)
    flux_col = "mean_wastewater_flux" if "mean_wastewater_flux" in nf.columns else "mean_flow_m3s"
    priors["Prior_C_Flux"] = normalise(src_feat(flux_col) + 1e-9)

    # Prior D: mean contaminant flux (most informative v1.0 prior)
    if "mean_contaminant_flux" in nf.columns:
        priors["Prior_D_ContamFlux"] = normalise(src_feat("mean_contaminant_flux") + 1e-9)
    else:
        # Approximate: prior_contam_prob × mean_flow_m3s
        priors["Prior_D_ContamFlux"] = normalise(
            src_feat("prior_contam_prob") * (src_feat("mean_flow_m3s") + 1e-9)
        )

    return priors


def load_ml_priors(priors_dir, candidate_sources, source_order):
    """
    Loads ML-derived priors from train_models.py output.
    Only source nodes in source_order are kept; others are zeroed.
    Returns dict: prior_name -> np.array of shape (n_sources,).
    """
    ml_priors = {}
    if not os.path.isdir(priors_dir):
        print(f"      No priors directory found at {priors_dir} -- skipping ML priors")
        return ml_priors

    for fname in sorted(os.listdir(priors_dir)):
        if not fname.endswith(".csv"):
            continue
        label = fname.replace("prior_", "ML_").replace(".csv", "").upper()
        df = pd.read_csv(os.path.join(priors_dir, fname))

        if "node_id" not in df.columns or "prior_prob" not in df.columns:
            continue

        prob_map = dict(zip(df["node_id"], df["prior_prob"]))
        probs = np.array([prob_map.get(s, 0.0) for s in source_order])
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(source_order)) / len(source_order)

        ml_priors[label] = probs
        print(f"        Loaded: {label}")

    return ml_priors


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BAYESIAN UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_update(prior, detection_vector, p_kj_placed, epsilon=EPSILON, max_iter=MAX_ITER):
    """
    Iterates Bayesian updates until the posterior converges.

    Parameters
    ----------
    prior : np.ndarray (n_sources,)
        Current prior over contamination sources.
    detection_vector : np.ndarray (n_placed_sensors,)
        Binary detection result at each placed sensor (1=detected, 0=not).
    p_kj_placed : np.ndarray (n_placed_sensors, n_sources)
        Rows of p_kj for the currently placed sensors.
    epsilon : float
        Convergence threshold: max|P_new - P_old| < epsilon.
    max_iter : int
        Hard iteration limit.

    Returns
    -------
    posterior : np.ndarray (n_sources,)
        Converged posterior distribution.
    n_iter : int
        Number of Bayesian updates performed.
    """
    posterior = prior.copy()

    for n_iter in range(1, max_iter + 1):
        # Likelihood P(d | source=j) for each source j
        # = product over placed sensors k of p_kj^d_k * (1-p_kj)^(1-d_k)
        d = detection_vector[:, np.newaxis]           # (n_sensors, 1)
        p = p_kj_placed                                # (n_sensors, n_sources)

        log_likelihood = np.sum(
            d * np.log(p) + (1 - d) * np.log(1 - p),
            axis=0
        )
        likelihood = np.exp(log_likelihood - log_likelihood.max())  # normalise for stability

        unnorm = likelihood * posterior
        total  = unnorm.sum()
        new_posterior = unnorm / total if total > 0 else posterior.copy()

        delta = np.max(np.abs(new_posterior - posterior))
        posterior = new_posterior

        if delta < epsilon:
            break

    return posterior, n_iter


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GREEDY SENSOR PLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def greedy_sensor_placement(
    prior, p_kj, sensor_order, source_order,
    raw_df, n_sensors, verbose=True
):
    """
    Greedy sequential sensor placement with Bayesian updating.

    For each sensor position:
      1. Compute expected coverage for each unplaced sensor node
      2. Place the sensor that maximises expected coverage
      3. Sample a random scenario from raw_df and observe detection
      4. Update posterior via Bayes
      5. Repeat step 3-4 until posterior converges
      6. Record convergence iteration count

    Parameters
    ----------
    prior : np.ndarray (n_sources,)
    p_kj  : np.ndarray (n_sensors, n_sources)
    sensor_order : list of str
    source_order : list of str
    raw_df : pd.DataFrame
    n_sensors : int
        Number of sensors to place.
    verbose : bool

    Returns
    -------
    dict with keys:
      placed_sensors   : list of str
      convergence_iters: list of int (one per sensor placement)
      final_posterior  : np.ndarray (n_sources,)
    """
    sensor_idx = {s: i for i, s in enumerate(sensor_order)}
    posterior  = prior.copy()
    placed     = []
    placed_rows = []   # row indices in p_kj for placed sensors
    iters_list = []

    # Unique scenarios available for sampling
    scen_ids = raw_df["scen_id"].unique()

    for step in range(n_sensors):
        # ── Step 1: select best sensor ─────────────────────────────────────
        candidates = [s for s in sensor_order if s not in placed]
        best_node  = None
        best_cover = -1.0

        for cand in candidates:
            k = sensor_idx[cand]
            coverage = float(np.dot(p_kj[k, :], posterior))
            if coverage > best_cover:
                best_cover = coverage
                best_node  = cand

        placed.append(best_node)
        placed_rows.append(sensor_idx[best_node])
        p_kj_placed = p_kj[placed_rows, :]   # (n_placed, n_sources)

        if verbose:
            print(f"      Sensor {step+1}: {best_node:<10}  "
                  f"expected_coverage={best_cover:.4f}")

        # ── Steps 2-4: Bayesian updating until convergence ──────────────────
        total_iters = 0
        n_updates   = 0

        while total_iters < MAX_ITER:
            # Sample a random scenario
            scen_id = rng.choice(scen_ids)
            scen_rows = raw_df[raw_df["scen_id"] == scen_id]

            # Build detection vector for placed sensors
            det_vector = np.array([
                int(scen_rows.loc[scen_rows["node_id"] == s, "detected"].values[0])
                if s in scen_rows["node_id"].values else 0
                for s in placed
            ], dtype=float)

            old_posterior = posterior.copy()
            posterior, n_iter = bayesian_update(
                posterior, det_vector, p_kj_placed
            )
            total_iters += n_iter
            n_updates   += 1

            # Check global convergence (stable posterior across updates)
            if np.max(np.abs(posterior - old_posterior)) < EPSILON:
                break

        iters_list.append(n_updates)
        if verbose:
            top_source = source_order[np.argmax(posterior)]
            print(f"                  Bayesian updates: {n_updates:3d}  "
                  f"top_source={top_source}  P={posterior.max():.4f}")

    return {
        "placed_sensors":    placed,
        "convergence_iters": iters_list,
        "final_posterior":   posterior,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION METRICS (F1 and F2)
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1_f2(placed_sensors, prior, p_kj, sensor_order, source_order):
    """
    Computes F1 (isolation likelihood) and F2 (detection reliability).

    F2 -- Detection reliability:
        P(at least one placed sensor detects | contamination event)
        = sum_j P(j) * [1 - prod_{s in S}(1 - p_sj)]

    F1 -- Isolation likelihood:
        P(exactly one placed sensor detects | contamination event)
        Approximated as: for each source j, P(j) * P(exactly one detection | j)
        = sum_j P(j) * sum_{s in S} [ p_sj * prod_{t != s}(1 - p_tj) ]

    Parameters
    ----------
    placed_sensors : list of str
    prior : np.ndarray (n_sources,) -- the prior at placement time
    p_kj  : np.ndarray (n_sensors, n_sources)
    sensor_order, source_order : list of str

    Returns
    -------
    F1, F2 : float
    """
    sensor_idx = {s: i for i, s in enumerate(sensor_order)}
    placed_idx = [sensor_idx[s] for s in placed_sensors if s in sensor_idx]

    if not placed_idx:
        return 0.0, 0.0

    p_placed = p_kj[placed_idx, :]   # (n_placed, n_sources)
    n_placed, n_src = p_placed.shape

    # F2: P(at least one detects)
    prob_none = np.prod(1.0 - p_placed, axis=0)   # (n_sources,)
    f2 = float(np.dot(prior, 1.0 - prob_none))

    # F1: P(exactly one detects)
    f1_per_src = np.zeros(n_src)
    for k in range(n_placed):
        # sensor k detects, all others don't
        p_k  = p_placed[k, :]
        others_miss = np.ones(n_src)
        for m in range(n_placed):
            if m != k:
                others_miss *= (1.0 - p_placed[m, :])
        f1_per_src += p_k * others_miss

    f1 = float(np.dot(prior, f1_per_src))

    return round(f1, 4), round(f2, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(raw_scenarios_path, node_features_path, priors_dir,
         n_sensors, output_dir):

    for sub in ["results"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    print("=" * 60)
    print("Bayesian Decision Network Solver")
    print("Hybrid AI Sensor Placement -- Mhango & Sambito (2026)")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data ...")
    rs, nf, node_list, candidate_sources, candidate_sensors = load_data(
        raw_scenarios_path, node_features_path
    )

    # ── Build p_kj ─────────────────────────────────────────────────────────────
    print("\n[2/5] Building detection probability matrix p_kj ...")
    p_kj, sensor_order, source_order = build_pkj_matrix(
        rs, candidate_sensors, candidate_sources
    )
    print(f"      p_kj shape: {p_kj.shape} "
          f"({len(sensor_order)} sensors × {len(source_order)} sources)")
    print(f"      Mean p_kj: {p_kj.mean():.4f}  |  Max: {p_kj.max():.4f}")

    # ── Build priors ────────────────────────────────────────────────────────────
    print("\n[3/5] Building prior distributions ...")
    print("    v1.0 baseline priors:")
    v1_priors = build_v1_priors(nf, candidate_sources, source_order)
    for name in v1_priors:
        print(f"      {name}")

    print("    ML priors:")
    ml_priors = load_ml_priors(priors_dir, candidate_sources, source_order)
    all_priors = {**v1_priors, **ml_priors}

    # ── Run BDN for each prior ─────────────────────────────────────────────────
    print(f"\n[4/5] Running BDN sensor placement (n_sensors={n_sensors}) ...")
    all_results = []

    for prior_name, prior in all_priors.items():
        print(f"\n  Prior: {prior_name}")

        for current_eta in range(1, n_sensors + 1):
            import time
            start_time = time.time()
            result = greedy_sensor_placement(
                prior      = prior,
                p_kj       = p_kj,
                sensor_order = sensor_order,
                source_order = source_order,
                raw_df     = rs,
                n_sensors  = current_eta,
                verbose    = False,
            )
            inference_time = time.time() - start_time

            f1, f2 = compute_f1_f2(
                result["placed_sensors"],
                prior, p_kj, sensor_order, source_order
            )

            total_iters = sum(result["convergence_iters"])

            all_results.append({
                "prior":            prior_name,
                "eta":              current_eta,
                "sensors_placed":   ", ".join(result["placed_sensors"]),
                "convergence_iters": ", ".join(str(x) for x in result["convergence_iters"]),
                "total_updates":    total_iters,
                "F1":               f1,
                "F2":               f2,
                "inference_time_s": round(inference_time, 2),
            })

        print(f"    Sensors (n={n_sensors}): {result['placed_sensors']}")
        print(f"    F1 (isolation):     {f1:.4f}")
        print(f"    F2 (detection):     {f2:.4f}")
        print(f"    Total updates:      {total_iters}")

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n[5/5] Saving results ...")
    results_df = pd.DataFrame(all_results)

    # Comparison table
    comp_path = os.path.join(output_dir, "results", "comparison_table.csv")
    results_df.to_csv(comp_path, index=False)
    print(f"      Saved: {comp_path}")

    # Summary printout
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Prior':<26} {'eta':>3} {'Sensors':<22} {'Updates':>7} {'F1':>7} {'F2':>7}")
    print("-" * 80)
    # Only print the max eta for the summary to save space
    max_eta_df = results_df[results_df["eta"] == n_sensors]
    for _, row in max_eta_df.sort_values("F1", ascending=False).iterrows():
        sensors_short = row["sensors_placed"][:22]
        print(f"{str(row['prior']):<26} {int(row['eta']):>3} {sensors_short:<22} "
              f"{int(row['total_updates']):>7} "
              f"{float(row['F1']):>7.4f} {float(row['F2']):>7.4f}")
    print("=" * 80)

    print(f"\nNote: F1 = isolation likelihood | F2 = detection reliability")
    print(f"      Higher is better for both. An ML prior with fewer updates")
    print(f"      than Prior_D_ContamFlux confirms the hybrid AI hypothesis.")
    print(f"\nAll results written to: {os.path.abspath(output_dir)}/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BDN solver for sensor placement (Mhango & Sambito 2026)"
    )
    parser.add_argument("--raw_scenarios",  default="./output/raw_scenarios.csv")
    parser.add_argument("--node_features",  default="./output/node_features_full.csv")
    parser.add_argument("--priors_dir",     default="./ml_output/priors")
    parser.add_argument("--n_sensors",      type=int, default=3)
    parser.add_argument("--output_dir",     default="./bdn_output")
    args = parser.parse_args()

    main(
        raw_scenarios_path = args.raw_scenarios,
        node_features_path = args.node_features,
        priors_dir         = args.priors_dir,
        n_sensors          = args.n_sensors,
        output_dir         = args.output_dir,
    )