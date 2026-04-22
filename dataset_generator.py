"""
dataset_generator.py
====================
Generates the ML training dataset for the Hybrid AI sensor placement project.
Sambito & Mhango (2026)

raw_scenarios.csv columns (matches concept note exactly):
  scen_id        scenario number
  src_node       injection node
  node_id        observed node
  dist_src       pipe-segment distance from src_node to node_id
  topo_depth     shortest directed path from node_id to nearest outfall
  peak_conc      peak concentration at node_id (mg/L)
  t_peak_min     time from injection start to peak (minutes); blank if none
  mean_flow_m3s  mean total inflow at node_id during scenario (m3/s)
  detected       1 if peak_conc >= 5 mg/L, else 0
  mass_kg / duration_hrs / start_hrs / conc_injected  (scenario parameters)

Usage:
  python dataset_generator.py --n_scenarios 100 --output_dir ./output
  python dataset_generator.py --n_scenarios 5000 --output_dir ./output

Requirements:
  pip install pyswmm swmm-toolkit networkx pandas numpy
"""

import os, argparse, random, warnings, multiprocessing as mp
import numpy as np
import pandas as pd
import networkx as nx
from swmm.toolkit.solver import swmm_open, swmm_close, swmm_start, swmm_step, swmm_end
from swmm.toolkit import solver as slv
from config import config
from cache import cached, cache

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
THRESHOLD       = config.get('dataset.threshold', 5.0)        # mg/L
CARRIER_FLOW    = config.get('dataset.carrier_flow', 0.01)    # cfs
CFS_TO_M3S      = 0.028317
MASS_MIN, MASS_MAX = config.get('dataset.mass_range', [0.01, 0.50])    # kg
DURATION_MIN, DURATION_MAX = config.get('dataset.duration_range', [0.25, 3.0])  # hours
START_MIN, START_MAX = config.get('dataset.start_range', [0.0, 6.0])    # hours
HIGH_RISK_NODES = set(config.get('dataset.high_risk_nodes', ['J4', 'J10', 'JI18']))
EXCLUDE_SOURCE  = set(config.get('dataset.exclude_sources', ['O1', 'O2', 'Well']))


# ── 1. Parse network ───────────────────────────────────────────────────────────
@cached()
def parse_network(inp_file):
    nodes, outfalls, storage, links = [], [], [], []
    section = None
    with open(inp_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(';;'): continue
            if s.startswith('['):
                section = s.strip('[]').upper(); continue
            parts = s.split()
            if not parts: continue
            if section == 'JUNCTIONS':  nodes.append(parts[0])
            elif section == 'OUTFALLS': outfalls.append(parts[0]); nodes.append(parts[0])
            elif section == 'STORAGE':  storage.append(parts[0]);  nodes.append(parts[0])
            elif section in ('CONDUITS','PUMPS','WEIRS','ORIFICES'):
                if len(parts) >= 3: links.append((parts[1], parts[2]))

    def ntype(n):
        if n in outfalls:      return 'outfall'
        if n in storage:       return 'storage'
        if n == 'Aux3':        return 'aux'
        if n.startswith('JI'): return 'JI'
        if n.startswith('J'):  return 'J'
        return 'other'

    node_types = {n: ntype(n) for n in nodes}

    G_dir = nx.DiGraph(); G_dir.add_nodes_from(nodes)
    G_und = nx.Graph();   G_und.add_nodes_from(nodes)
    for f, t in links:
        if f in G_dir and t in G_dir:
            G_dir.add_edge(f, t)
            G_und.add_edge(f, t)

    return nodes, outfalls, node_types, G_dir, G_und


# ── 2. Topology distances ──────────────────────────────────────────────────────
def compute_topo_depth(nodes, outfalls, G_dir):
    depth = {}
    for n in nodes:
        best = 999
        for o in outfalls:
            try: best = min(best, nx.shortest_path_length(G_dir, n, o))
            except: pass
        depth[n] = best
    return depth

def compute_dist_matrix(nodes, G_und):
    return dict(nx.all_pairs_shortest_path_length(G_und))


# ── 3. Static feature table ────────────────────────────────────────────────────
def build_topology_features(nodes, outfalls, node_types, G_dir, topo_depth):
    bc       = nx.betweenness_centrality(G_dir, normalized=True)
    type_map = {'J':0, 'JI':1, 'aux':2, 'outfall':3, 'storage':4, 'other':5}
    n_high   = len(HIGH_RISK_NODES)
    base_p   = 1.0 / (len(nodes) + n_high)
    high_p   = 2.0 * base_p

    def down_paths(n):
        c = 0
        for o in outfalls:
            try: c += len(list(nx.all_simple_paths(G_dir, n, o, cutoff=20)))
            except: pass
        return c

    rows = []
    for n in nodes:
        rows.append({
            'node_id':          n,
            'topo_depth':       topo_depth[n],
            'n_upstream_nodes': len(nx.ancestors(G_dir, n)),
            'betweenness':      round(bc.get(n, 0), 6),
            'downstream_paths': down_paths(n),
            'node_type':        node_types[n],
            'node_type_code':   type_map.get(node_types[n], 5),
            'is_high_risk':     1 if n in HIGH_RISK_NODES else 0,
            'prior_contam_prob': high_p if n in HIGH_RISK_NODES else base_p,
        })
    return pd.DataFrame(rows).set_index('node_id')


# ── 4. Scenario .inp builder ───────────────────────────────────────────────────
def format_time(hours):
    hours = min(max(hours, 0.0), 11.99)
    h = int(hours); m = int(round((hours - h) * 60))
    if m == 60: h += 1; m = 0
    return f"{h:02d}:{m:02d}"

def build_scenario_inp(base_inp, tmp_inp, src, conc_mg_l, dur_hrs, start_hrs):
    end_hrs  = start_hrs + dur_hrs
    pre_hrs  = max(0.0, start_hrs - 1.0/60.0)
    post_hrs = end_hrs + 5.0/60.0

    def ts(val):
        return [
            ("00:00",                0.0),
            (format_time(pre_hrs),   0.0),
            (format_time(start_hrs), val),
            (format_time(end_hrs),   val),
            (format_time(post_hrs),  0.0),
            ("12:00",                0.0),
        ]

    ts_flow = f'CarrierFlow_{src}'
    ts_conc = f'ContamConc_{src}'

    with open(base_inp) as f:
        content = f.read()

    # Remove invalid QUALITY keyword that causes ERROR 205
    content = content.replace('QUALITY    ALL\n', '').replace('QUALITY ALL\n', '')

    # 1. Ensure [POLLUTANTS] section is present
    if '[POLLUTANTS]' not in content:
        pb = ('[POLLUTANTS]\n'
              ';;Name  Units  Crain  Cgw    Crdii  Kdecay  SnowOnly\n'
              'CONTAM  MG/L   0.0    0.0    0.0    0.0     NO\n\n')
        if '[INFLOWS]' in content:
            content = content.replace('[INFLOWS]', pb + '[INFLOWS]')
        elif '[JUNCTIONS]' in content:
            content = content.replace('[JUNCTIONS]', pb + '[JUNCTIONS]')
        else:
            content += "\n" + pb

    # 2. Ensure [TIMESERIES] section is present and inject series
    ts_block = f';; Injection Timeseries for {src}\n'
    for t, v in ts(CARRIER_FLOW):
        ts_block += f'{ts_flow:<28}{t:<12}{v}\n'
    for t, v in ts(conc_mg_l):
        ts_block += f'{ts_conc:<28}{t:<12}{v}\n'

    if '[TIMESERIES]' in content:
        parts = content.split('[TIMESERIES]')
        post_ts = parts[1].split('[', 1)
        if len(post_ts) > 1:
            content = parts[0] + '[TIMESERIES]' + post_ts[0] + ts_block + '[' + post_ts[1]
        else:
            content = parts[0] + '[TIMESERIES]' + parts[1] + "\n" + ts_block
    else:
        content += f"\n[TIMESERIES]\n{ts_block}\n"

    # 3. Ensure [INFLOWS] section is present and inject entries
    inflow_entry = (f'{src:<17}FLOW   {ts_flow:<20}DIRECT  1.0  1.0\n'
                    f'{src:<17}CONTAM {ts_conc:<20}CONCEN  1.0  1.0\n')
    
    if '[INFLOWS]' in content:
        parts = content.split('[INFLOWS]')
        post_in = parts[1].split('[', 1)
        if len(post_in) > 1:
            content = parts[0] + '[INFLOWS]' + post_in[0] + inflow_entry + '[' + post_in[1]
        else:
            content = parts[0] + '[INFLOWS]' + parts[1] + "\n" + inflow_entry
    else:
        content += f"\n[INFLOWS]\n{inflow_entry}\n"

    with open(tmp_inp, 'w') as f:
        f.write(content)


# ── 5. Run one scenario ────────────────────────────────────────────────────────
def run_scenario(tmp_inp, node_ids):
    """
    Run SWMM using the low-level swmm.toolkit API.
    Collects per-node: peak_conc, t_peak_min, mean_flow_m3s, detected.
    node_get_pollutant(i, 0)[0]  -> CONTAM concentration
    node_get_result(i, 0)        -> total inflow (cfs)
    """
    rpt, outf = tmp_inp.replace('.inp','.rpt'), tmp_inp.replace('.inp','.out')
    results   = {}

    try:
        swmm_open(tmp_inp, rpt, outf)
        swmm_start(True)

        n_count   = slv.project_get_count(slv.swmm_NODE)
        ids_order = [slv.project_get_id(slv.swmm_NODE, i) for i in range(n_count)]

        peaks      = {nid: 0.0 for nid in ids_order}
        peak_step  = {nid: 0   for nid in ids_order}
        inflow_sum = {nid: 0.0 for nid in ids_order}
        step_n = 0

        while True:
            t = swmm_step()
            if t == 0: break
            step_n += 1
            for i, nid in enumerate(ids_order):
                # concentration
                c = slv.node_get_pollutant(i, 0)[0]
                if c > peaks[nid]:
                    peaks[nid]     = c
                    peak_step[nid] = step_n
                # inflow (cfs)
                inflow_sum[nid] += abs(slv.node_get_result(i, 0))

        swmm_end()
        swmm_close()

        for nid in node_ids:
            peak  = peaks.get(nid, 0.0)
            pstep = peak_step.get(nid, 0)
            mflow = round((inflow_sum.get(nid, 0.0) / max(step_n, 1)) * CFS_TO_M3S, 6)
            results[nid] = {
                'peak_conc':     round(peak, 4),
                't_peak_min':    round(pstep * 15.0 / 60.0, 2) if peak > 0 else None,
                'mean_flow_m3s': mflow,
                'detected':      1 if peak >= THRESHOLD else 0,
            }

    except Exception as e:
        print(f"    WARNING: simulation failed -- {e}")
        for nid in node_ids:
            results[nid] = {
                'peak_conc': 0.0, 't_peak_min': None,
                'mean_flow_m3s': 0.0, 'detected': 0,
            }
    finally:
        for f in [rpt, outf]:
            try: os.remove(f)
            except: pass

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────
def sampling_weights(candidates):
    w = [2 if n in HIGH_RISK_NODES else 1 for n in candidates]
    t = sum(w)
    return candidates, [x / t for x in w]

def mass_to_conc(mass_kg, dur_hrs):
    vol = CARRIER_FLOW * (dur_hrs * 3600) * 28.317
    return round((mass_kg * 1e6) / max(vol, 1e-9), 2)


def run_single_scenario(args):
    """Run a single scenario (for parallel processing)."""
    i, src, mass_kg, dur_hrs, start_hr, conc, inp_file, nodes, dist_matrix, topo_depth = args

    tmp_inp = f'_scenario_tmp_{os.getpid()}_{i}.inp'  # Unique temp file per process
    tmp_inp = os.path.join(os.path.dirname(inp_file), tmp_inp)

    build_scenario_inp(inp_file, tmp_inp, src, conc, dur_hrs, start_hr)
    res = run_scenario(tmp_inp, nodes)

    try:
        os.remove(tmp_inp)
    except:
        pass

    # Build rows for this scenario
    rows = []
    for nid in nodes:
        r = res[nid]
        rows.append({
            'scen_id':       i + 1,
            'src_node':      src,
            'mass_kg':       round(mass_kg, 4),
            'duration_hrs':  round(dur_hrs, 3),
            'start_hrs':     round(start_hr, 3),
            'conc_injected': conc,
            'node_id':       nid,
            'dist_src':      dist_matrix.get(src, {}).get(nid, 999),
            'topo_depth':    topo_depth[nid],
            'peak_conc':     r['peak_conc'],
            't_peak_min':    r['t_peak_min'],
            'mean_flow_m3s': r['mean_flow_m3s'],
            'detected':      r['detected'],
        })

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main(inp_file, n_scenarios, output_dir, seed=42):
    random.seed(seed); np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SWMM Dataset Generator  --  Mhango & Sambito (2026)")
    print("=" * 60)

    # Parse
    print("\n[1/5] Parsing network...")
    nodes, outfalls, node_types, G_dir, G_und = parse_network(inp_file)
    candidates = [n for n in nodes
                  if n not in EXCLUDE_SOURCE
                  and node_types[n] not in ('outfall', 'storage')]
    print(f"      Nodes: {len(nodes)}  |  Candidate sources: {len(candidates)}")

    # Distances
    print("\n[2/5] Computing topology distances...")
    topo_depth  = compute_topo_depth(nodes, outfalls, G_dir)
    dist_matrix = compute_dist_matrix(nodes, G_und)
    topo_df     = build_topology_features(nodes, outfalls, node_types, G_dir, topo_depth)
    valid_depths = [v for v in topo_depth.values() if v < 999]
    print(f"      topo_depth range: {min(valid_depths)} to {max(valid_depths)}")

    # Scenarios
    src_nodes, src_probs = sampling_weights(candidates)

    print(f"\n[3/5] Running {n_scenarios} scenarios...")

    # Check if parallel processing is enabled
    parallel_enabled = config.get('dataset.parallel.enabled', False)
    n_workers = config.get('dataset.parallel.n_workers')
    chunk_size = config.get('dataset.parallel.chunk_size', 10)

    if parallel_enabled and n_scenarios > 1:
        print(f"      Using parallel processing with {n_workers or 'all'} workers...")

        # Prepare scenario arguments
        scenario_args = []
        for i in range(n_scenarios):
            src = np.random.choice(src_nodes, p=src_probs)
            mass_kg = random.uniform(MASS_MIN, MASS_MAX)
            dur_hrs = random.uniform(DURATION_MIN, DURATION_MAX)
            start_hr = random.uniform(START_MIN, START_MAX)
            conc = mass_to_conc(mass_kg, dur_hrs)

            scenario_args.append((
                i, src, mass_kg, dur_hrs, start_hr, conc,
                inp_file, nodes, dist_matrix, topo_depth
            ))

        # Run scenarios in parallel
        with mp.Pool(processes=n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(run_single_scenario, scenario_args, chunksize=chunk_size)):
                results.extend(result)
                if (i + 1) % max(1, n_scenarios // 10) == 0:
                    print(f"      Completed {i+1}/{n_scenarios} scenarios")

        raw_rows = results
    else:
        # Sequential processing (original logic)
        tmp_inp = os.path.join(os.path.dirname(inp_file), '_scenario_tmp.inp')
        raw_rows = []

        for i in range(n_scenarios):
            src = np.random.choice(src_nodes, p=src_probs)
            mass_kg = random.uniform(MASS_MIN, MASS_MAX)
            dur_hrs = random.uniform(DURATION_MIN, DURATION_MAX)
            start_hr = random.uniform(START_MIN, START_MAX)
            conc = mass_to_conc(mass_kg, dur_hrs)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Scenario {i+1:>5}/{n_scenarios}  "
                      f"src={src:<8} mass={mass_kg:.3f}kg  "
                      f"dur={dur_hrs:.2f}h  conc={conc:.1f}mg/L")

            build_scenario_inp(inp_file, tmp_inp, src, conc, dur_hrs, start_hr)
            res = run_scenario(tmp_inp, nodes)

            for nid in nodes:
                r = res[nid]
                raw_rows.append({
                    'scen_id':       i + 1,
                    'src_node':      src,
                    'mass_kg':       round(mass_kg, 4),
                    'duration_hrs':  round(dur_hrs, 3),
                    'start_hrs':     round(start_hr, 3),
                    'conc_injected': conc,
                    'node_id':       nid,
                    'dist_src':      dist_matrix.get(src, {}).get(nid, 999),
                    'topo_depth':    topo_depth[nid],
                    'peak_conc':     r['peak_conc'],
                    't_peak_min':    r['t_peak_min'],
                    'mean_flow_m3s': r['mean_flow_m3s'],
                    'detected':      r['detected'],
                })

        try:
            os.remove(tmp_inp)
        except:
            pass

    print(f"\n  Done. Total rows: {len(raw_rows):,}")

    # Save raw
    print("\n[4/5] Saving raw_scenarios.csv...")
    raw_df   = pd.DataFrame(raw_rows)
    raw_path = os.path.join(output_dir, 'raw_scenarios.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"      {raw_path}  ({len(raw_df):,} rows)")

    # Build node_features
    print("\n[5/5] Building node_features.csv...")
    grp = raw_df.groupby('node_id')
    node_agg = pd.DataFrame({
        'detection_freq':       grp['detected'].mean().round(4),
        'peak_conc_mean':       grp['peak_conc'].mean().round(4),
        'peak_conc_std':        grp['peak_conc'].std().round(4),
        'time_to_peak_mean':    grp['t_peak_min'].mean().round(2),
        'mean_flow_m3s':        grp['mean_flow_m3s'].mean().round(6),
        'n_scenarios_detected': grp['detected'].sum().astype(int),
    })
    final_df  = topo_df.join(node_agg, how='left').reset_index()
    feat_path = os.path.join(output_dir, 'node_features.csv')
    final_df.to_csv(feat_path, index=False)
    print(f"      {feat_path}  ({len(final_df.columns)} columns)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Scenarios      : {n_scenarios}")
    print(f"  Total rows     : {len(raw_df):,}")
    print(f"  Detection rate : {raw_df['detected'].mean():.1%}")
    print(f"\n  Top 5 nodes by detection frequency:")
    top5 = raw_df.groupby('node_id')['detected'].mean().sort_values(ascending=False).head(5)
    for nid, freq in top5.items():
        td = topo_depth.get(nid, '?')
        mf = raw_df[raw_df['node_id']==nid]['mean_flow_m3s'].mean()
        print(f"    {nid:<10}  freq={freq:.3f}  topo_depth={td}  mean_flow={mf:.5f} m3/s")

    print(f"\n  Sample rows (concept note format, detected=1 only):")
    cols = ['scen_id','src_node','node_id','dist_src','topo_depth',
            'peak_conc','t_peak_min','mean_flow_m3s','detected']
    print(raw_df[raw_df['detected']==1][cols].head(8).to_string(index=False))

    return raw_df, final_df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',         default='./dataset/Examples/Example8.inp', help='Path to SWMM input file')
    p.add_argument('--n_scenarios', type=int, default=100, help='Number of scenarios to simulate')
    p.add_argument('--output_dir',  default='./output', help='Directory for output CSV files')
    p.add_argument('--seed',        type=int, default=42, help='Random seed for reproducibility')
    a = p.parse_args()
    main(a.model_path, a.n_scenarios, a.output_dir, a.seed)
