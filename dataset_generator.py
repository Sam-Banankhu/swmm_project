"""
dataset_generator.py
====================
Generates the ML training dataset for the Hybrid AI sensor placement project.
Sambito & Mhango (2026)

What this script does:
  1. Loads the SWMM Example 8 network
  2. Builds static topology features for every node using NetworkX
  3. Runs N contamination scenarios through SWMM, varying:
       - source node (non-uniform probability, J4/J10/JI18 at 2x)
       - contaminant mass  (random, 0.01 to 0.5 kg equivalent)
       - injection duration (random, 0.25 to 3.0 hours)
       - injection start time (random, within first 6 hours)
  4. For each (scenario, node) pair records:
       - peak concentration
       - time to peak
       - binary detection flag (threshold = 5 mg/L)
  5. Computes per-node summary features across all scenarios
  6. Saves two files:
       - raw_scenarios.csv     (one row per scenario x node)
       - node_features.csv     (one row per node, aggregated features)

Usage:
  python dataset_generator.py --n_scenarios 100 --output_dir ./output
  python dataset_generator.py --n_scenarios 5000 --output_dir ./output

Requirements:
  pip install pyswmm swmm-toolkit networkx pandas numpy
"""

import os
import sys
import argparse
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from swmm.toolkit.solver import (
    swmm_open, swmm_close, swmm_start, swmm_step, swmm_end
)
from swmm.toolkit import solver as slv

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

INP_FILE       = 'Example8.inp'       # path to your SWMM input file
THRESHOLD      = 5.0                  # mg/L detection threshold
CARRIER_FLOW   = 0.01                 # cfs - small carrier flow to transport pollutant

# Contamination parameter ranges (from Sambito et al. 2020)
MASS_MIN       = 0.01                 # kg
MASS_MAX       = 0.50                 # kg
DURATION_MIN   = 0.25                 # hours
DURATION_MAX   = 3.0                  # hours
START_MIN      = 0.0                  # hours from sim start
START_MAX      = 6.0                  # hours from sim start

# Non-uniform source node probabilities (v1.0 Figure 2)
# J4, J10, JI18 have double contamination probability
HIGH_RISK_NODES = {'J4', 'J10', 'JI18'}

# Nodes to EXCLUDE as injection sources (outfalls, storage, non-physical)
EXCLUDE_SOURCE  = {'O1', 'O2', 'Well'}

# ── Step 1: Read node list from SWMM file ─────────────────────────────────────

def get_nodes_and_links(inp_file):
    """
    Parse the SWMM input file to get:
      - list of all node IDs
      - list of (from_node, to_node) for all conduits
      - node type mapping (J=sewer, JI=interceptor, Aux=special, outfall, storage)
    """
    nodes    = []
    links    = []
    outfalls = []
    storage  = []

    section = None
    with open(inp_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;'):
                continue
            if line.startswith('['):
                section = line.strip('[]').upper()
                continue

            parts = line.split()
            if not parts:
                continue

            if section == 'JUNCTIONS':
                nodes.append(parts[0])
            elif section == 'OUTFALLS':
                outfalls.append(parts[0])
                nodes.append(parts[0])
            elif section == 'STORAGE':
                storage.append(parts[0])
                nodes.append(parts[0])
            elif section == 'CONDUITS':
                if len(parts) >= 3:
                    links.append((parts[1], parts[2], parts[0]))  # from, to, name

    def node_type(n):
        if n in outfalls:   return 'outfall'
        if n in storage:    return 'storage'
        if n == 'Aux3':     return 'aux'
        if n.startswith('JI'): return 'JI'
        if n.startswith('J'):  return 'J'
        return 'other'

    node_types = {n: node_type(n) for n in nodes}
    return nodes, links, node_types


# ── Step 2: Build topology features using NetworkX ───────────────────────────

def build_topology_features(nodes, links, node_types):
    """
    Builds a directed graph of the sewer network and computes
    static topology features for each node.

    Features computed:
      - topo_depth          : shortest path distance to nearest outfall
      - n_upstream_nodes    : number of nodes that can reach this node
      - betweenness         : betweenness centrality (normalised)
      - downstream_paths    : number of distinct downstream routes to any outfall
      - node_type_code      : 0=J, 1=JI, 2=Aux, 3=outfall, 4=storage, 5=other
      - is_high_risk        : 1 if node is J4, J10, or JI18 (v1.0 Figure 2)
      - prior_contam_prob   : contamination source probability (v1.0 Figure 2)
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for from_n, to_n, _ in links:
        if from_n in G and to_n in G:
            G.add_edge(from_n, to_n)

    outfalls = [n for n, t in node_types.items() if t == 'outfall']

    # Shortest path to nearest outfall
    def topo_depth(node):
        best = 999
        for o in outfalls:
            try:
                d = nx.shortest_path_length(G, node, o)
                best = min(best, d)
            except nx.NetworkXNoPath:
                pass
        return best

    # Upstream node count (nodes that can reach this one)
    def upstream_count(node):
        try:
            return len(nx.ancestors(G, node))
        except:
            return 0

    # Number of downstream paths to any outfall
    def downstream_path_count(node):
        count = 0
        for o in outfalls:
            try:
                paths = list(nx.all_simple_paths(G, node, o, cutoff=20))
                count += len(paths)
            except:
                pass
        return count

    # Betweenness centrality
    bc = nx.betweenness_centrality(G, normalized=True)

    # Node type code
    type_map = {'J': 0, 'JI': 1, 'aux': 2, 'outfall': 3, 'storage': 4, 'other': 5}

    # Prior contamination probability (from v1.0 Figure 2)
    # High-risk nodes ~0.06, others ~0.03
    n_nodes    = len(nodes)
    n_high     = len(HIGH_RISK_NODES)
    # Normalised so all probs sum to 1
    base_prob  = 1.0 / (n_nodes + n_high)   # high-risk nodes count double
    high_prob  = 2.0 * base_prob

    rows = []
    for n in nodes:
        rows.append({
            'node_id':            n,
            'topo_depth':         topo_depth(n),
            'n_upstream_nodes':   upstream_count(n),
            'betweenness':        round(bc.get(n, 0), 6),
            'downstream_paths':   downstream_path_count(n),
            'node_type':          node_types[n],
            'node_type_code':     type_map.get(node_types[n], 5),
            'is_high_risk':       1 if n in HIGH_RISK_NODES else 0,
            'prior_contam_prob':  high_prob if n in HIGH_RISK_NODES else base_prob,
        })

    return pd.DataFrame(rows).set_index('node_id')


# ── Step 3: Build a modified SWMM inp for one scenario ───────────────────────

def format_time(hours):
    """Convert decimal hours to HH:MM, clamped to 11:59"""
    hours = min(max(hours, 0.0), 11.99)
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"

def build_scenario_inp(base_inp, tmp_inp, source_node,
                       concentration_mg_l, duration_hrs, start_hrs):
    """
    Writes a temporary SWMM input file with one contamination event injected
    at source_node starting at start_hrs, lasting duration_hrs hours,
    at the given concentration (mg/L).

    A small carrier flow (CARRIER_FLOW cfs) is also injected alongside the
    concentration so that SWMM's CONCEN inflow type has flow to carry it.
    """
    end_hrs     = start_hrs + duration_hrs
    pre_hrs     = max(0.0, start_hrs - (1.0/60.0))   # 1 min before start
    post_hrs    = end_hrs + (5.0/60.0)                # 5 min after end

    def ts_points(value_on):
        return [
            ("00:00",              0.0),
            (format_time(pre_hrs), 0.0),
            (format_time(start_hrs), value_on),
            (format_time(end_hrs),   value_on),
            (format_time(post_hrs),  0.0),
            ("12:00",              0.0),
        ]

    with open(base_inp) as f:
        lines = f.readlines()

    new_lines      = []
    ts_done        = False
    inflow_done    = False
    ts_name_flow   = f'CarrierFlow_{source_node}'
    ts_name_conc   = f'ContamConc_{source_node}'

    for line in lines:
        new_lines.append(line)

        # Append timeseries after the last Rain timeseries line
        if 'Rain_023in' in line and '12:00' in line and not ts_done:
            for t, v in ts_points(CARRIER_FLOW):
                new_lines.append(f'{ts_name_flow:<28}{t:<12}{v}\n')
            for t, v in ts_points(concentration_mg_l):
                new_lines.append(f'{ts_name_conc:<28}{t:<12}{v}\n')
            ts_done = True

        # Add inflow lines after the last dry weather inflow line
        if 'J12              FLOW' in line and '0.0125' in line and not inflow_done:
            new_lines.append(
                f'{source_node:<17}FLOW             {ts_name_flow:<17}DIRECT   1.0      1.0\n'
            )
            new_lines.append(
                f'{source_node:<17}CONTAM           {ts_name_conc:<17}CONCEN   1.0      1.0\n'
            )
            inflow_done = True

    with open(tmp_inp, 'w') as f:
        f.writelines(new_lines)


# ── Step 4: Run one scenario and extract results ──────────────────────────────

def run_scenario(tmp_inp, node_ids):
    """
    Runs SWMM on tmp_inp and returns a dict:
      node_id -> {peak_conc, time_to_peak_min, detected}
    """
    rpt   = tmp_inp.replace('.inp', '.rpt')
    out_f = tmp_inp.replace('.inp', '.out')

    results = {}

    try:
        swmm_open(tmp_inp, rpt, out_f)
        swmm_start(True)

        # We record concentration at every routing step (15 seconds)
        # time_to_peak is in minutes
        node_peaks     = {nid: 0.0   for nid in node_ids}
        node_peak_step = {nid: 0     for nid in node_ids}
        step_count = 0

        while True:
            t = swmm_step()
            if t == 0:
                break
            step_count += 1
            for i, nid in enumerate(node_ids):
                c = slv.node_get_pollutant(i, 0)[0]
                if c > node_peaks[nid]:
                    node_peaks[nid]     = c
                    node_peak_step[nid] = step_count

        swmm_end()
        swmm_close()

        # Convert step number to minutes (15s per step)
        for nid in node_ids:
            peak = node_peaks[nid]
            t_peak = (node_peak_step[nid] * 15.0) / 60.0 if peak > 0 else None
            results[nid] = {
                'peak_conc':       round(peak, 4),
                'time_to_peak_min': round(t_peak, 2) if t_peak else None,
                'detected':         1 if peak >= THRESHOLD else 0,
            }

    except Exception as e:
        print(f"    WARNING: simulation failed - {e}")
        for nid in node_ids:
            results[nid] = {'peak_conc': 0.0, 'time_to_peak_min': None, 'detected': 0}
    finally:
        for f in [rpt, out_f]:
            try: os.remove(f)
            except: pass

    return results


# ── Step 5: Build source node sampling weights ────────────────────────────────

def build_sampling_weights(candidate_nodes):
    """
    Returns a list of (node, weight) pairs.
    High-risk nodes (J4, J10, JI18) get weight 2, others get weight 1.
    """
    weights = []
    for n in candidate_nodes:
        w = 2 if n in HIGH_RISK_NODES else 1
        weights.append(w)
    total = sum(weights)
    probs = [w / total for w in weights]
    return candidate_nodes, probs


# ── Step 6: Convert mass (kg) to concentration (mg/L) for injection ──────────

def mass_to_concentration(mass_kg, duration_hrs, carrier_flow_cfs):
    """
    Converts an injection mass (kg) and duration into a concentration (mg/L)
    given the carrier flow rate.

    mass_kg      = total pollutant mass to inject
    duration_hrs = injection duration in hours
    carrier_cfs  = carrier flow in cubic feet per second

    mass_kg -> grams -> milligrams
    volume  = carrier_flow_cfs * duration_hrs * 3600 seconds * 28.317 L/ft3
    conc    = mass_mg / volume_L
    """
    mass_mg    = mass_kg * 1e6
    vol_liters = carrier_flow_cfs * (duration_hrs * 3600) * 28.317
    conc       = mass_mg / vol_liters if vol_liters > 0 else 0
    return round(conc, 2)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(inp_file, n_scenarios, output_dir, seed=42):

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SWMM Dataset Generator")
    print("Hybrid AI Sensor Placement -- Mhango & Sambito (2026)")
    print("=" * 60)

    # ── Load network ──
    print("\n[1/5] Reading network from SWMM input file...")
    all_nodes, links, node_types = get_nodes_and_links(inp_file)
    print(f"      Nodes found : {len(all_nodes)}")
    print(f"      Links found : {len(links)}")

    # Candidate source nodes (exclude outfalls, storage)
    candidate_sources = [
        n for n in all_nodes
        if n not in EXCLUDE_SOURCE and node_types[n] not in ('outfall', 'storage')
    ]
    print(f"      Candidate injection nodes: {len(candidate_sources)}")

    # ── Build topology features ──
    print("\n[2/5] Computing topology features...")
    topo_df = build_topology_features(all_nodes, links, node_types)
    print(f"      Features computed for {len(topo_df)} nodes")
    print(f"      Columns: {list(topo_df.columns)}")

    # ── Set up sampling ──
    src_nodes, src_probs = build_sampling_weights(candidate_sources)

    # ── Run scenarios ──
    print(f"\n[3/5] Running {n_scenarios} contamination scenarios...")
    print(f"      Source: {inp_file}")
    print(f"      Detection threshold: {THRESHOLD} mg/L")
    print(f"      Injection duration: U[{DURATION_MIN}, {DURATION_MAX}] hours")
    print(f"      Injection mass:     U[{MASS_MIN}, {MASS_MAX}] kg")
    print()

    raw_rows   = []
    tmp_inp    = os.path.join(os.path.dirname(inp_file), '_scenario_tmp.inp')

    for scen_i in range(n_scenarios):

        # Sample scenario parameters
        source_node   = np.random.choice(src_nodes, p=src_probs)
        mass_kg       = random.uniform(MASS_MIN, MASS_MAX)
        duration_hrs  = random.uniform(DURATION_MIN, DURATION_MAX)
        start_hrs     = random.uniform(START_MIN, START_MAX)
        concentration = mass_to_concentration(mass_kg, duration_hrs, CARRIER_FLOW)

        if (scen_i + 1) % 10 == 0 or scen_i == 0:
            print(f"  Scenario {scen_i+1:>5}/{n_scenarios}  "
                  f"source={source_node:<8}  "
                  f"mass={mass_kg:.3f}kg  "
                  f"dur={duration_hrs:.2f}h  "
                  f"start={start_hrs:.2f}h  "
                  f"conc={concentration:.1f}mg/L")

        # Build temp inp file
        build_scenario_inp(
            inp_file, tmp_inp, source_node,
            concentration, duration_hrs, start_hrs
        )

        # Run simulation
        node_results = run_scenario(tmp_inp, all_nodes)

        # Store one row per (scenario, node)
        for node_id in all_nodes:
            r = node_results[node_id]
            raw_rows.append({
                'scen_id':          scen_i + 1,
                'src_node':         source_node,
                'mass_kg':          round(mass_kg, 4),
                'duration_hrs':     round(duration_hrs, 3),
                'start_hrs':        round(start_hrs, 3),
                'conc_injected':    concentration,
                'node_id':          node_id,
                'peak_conc':        r['peak_conc'],
                'time_to_peak_min': r['time_to_peak_min'],
                'detected':         r['detected'],
            })

    # Clean up temp file
    try: os.remove(tmp_inp)
    except: pass

    print(f"\n  Done. Total rows: {len(raw_rows)}")

    # ── Save raw scenarios ──
    print("\n[4/5] Saving raw scenario dataset...")
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(output_dir, 'raw_scenarios.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"      Saved: {raw_path}  ({len(raw_df)} rows)")

    # ── Aggregate node features ──
    print("\n[5/5] Aggregating node-level features...")

    # Per-node dynamic features aggregated across all scenarios
    grp = raw_df.groupby('node_id')

    node_agg = pd.DataFrame({
        'detection_freq':       grp['detected'].mean(),
        'peak_conc_mean':       grp['peak_conc'].mean().round(4),
        'peak_conc_std':        grp['peak_conc'].std().round(4),
        'time_to_peak_mean':    grp['time_to_peak_min'].mean().round(2),
        'n_scenarios_detected': grp['detected'].sum().astype(int),
    })

    # Mean flow velocity per node -- approximated from SWMM link data
    # (populated from simulation; here we use a placeholder derived from
    #  the conduit roughness and geometry -- to be refined once SWMM is running)
    node_agg['mean_flow_velocity'] = 0.0   # placeholder -- fill from SWMM output

    # Merge with topology features
    final_df = topo_df.join(node_agg, how='left')
    final_df = final_df.reset_index().rename(columns={'index': 'node_id'})

    feat_path = os.path.join(output_dir, 'node_features.csv')
    final_df.to_csv(feat_path, index=False)
    print(f"      Saved: {feat_path}  ({len(final_df)} rows, {len(final_df.columns)} columns)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Scenarios run       : {n_scenarios}")
    print(f"  Nodes in network    : {len(all_nodes)}")
    print(f"  Total dataset rows  : {len(raw_df)}")
    print(f"  Detection rate      : {raw_df['detected'].mean():.1%}")
    print(f"\n  Top 5 nodes by detection frequency:")
    top5 = (raw_df.groupby('node_id')['detected']
              .mean()
              .sort_values(ascending=False)
              .head(5))
    for nid, freq in top5.items():
        print(f"    {nid:<12}  {freq:.3f}")

    print(f"\n  Output files:")
    print(f"    {raw_path}")
    print(f"    {feat_path}")
    print("\nDone.")

    return raw_df, final_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SWMM contamination scenario dataset for ML training'
    )
    parser.add_argument('--inp',        default='Example8.inp',
                        help='Path to SWMM input file (default: Example8.inp)')
    parser.add_argument('--n_scenarios',type=int, default=100,
                        help='Number of scenarios to run (default: 100, use 5000+ for full dataset)')
    parser.add_argument('--output_dir', default='./output',
                        help='Directory to save CSV files (default: ./output)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    main(
        inp_file    = args.inp,
        n_scenarios = args.n_scenarios,
        output_dir  = args.output_dir,
        seed        = args.seed,
    )
