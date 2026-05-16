import os
import sys
import argparse
import random
import yaml
import uuid
import time
import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pyswmm import Simulation, Nodes, Links

# ── 1. Constants ──────────────────────────────────────────────────────────────
CFS_TO_M3S = 0.0283168
G_TOPOLOGY = None  # Global graph for workers

def load_config(config_path="config/default.yaml"):
    if not os.path.exists(config_path): return None
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def format_time(hrs):
    h = int(hrs)
    m = int((hrs - h) * 60)
    s = int(((hrs - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ── 2. Topology ───────────────────────────────────────────────────────────────
def build_topology_features(inp_path, high_risk_nodes=None):
    import networkx as nx
    G = nx.DiGraph()
    outfalls = []
    high_risk_nodes = set(high_risk_nodes or [])
    
    with open(inp_path) as f:
        section = None
        for line in f:
            s = line.strip().upper()
            if not s or s.startswith(';'): continue
            if s.startswith('['): section = s.strip('[]'); continue
            parts = line.split()
            if section in ['CONDUITS', 'PUMPS', 'WEIRS', 'ORIFICES'] and len(parts) >= 3:
                G.add_edge(parts[1], parts[2])
            if section == 'OUTFALLS' and len(parts) >= 1:
                outfalls.append(parts[0])
    
    print(f"      Parsed {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"      Found {len(outfalls)} outfalls: {outfalls}")

    # Topology features
    topo_depth = {}
    betweenness = nx.betweenness_centrality(G)
    
    nodes = list(G.nodes())
    for node in nodes:
        min_d = 999
        for out in outfalls:
            try:
                d = nx.shortest_path_length(G, node, out)
                if d < min_d: min_d = d
            except: pass
        topo_depth[node] = min_d if min_d != 999 else 0

    rows = []
    print(f"      Computing features for {len(nodes)} nodes...")
    for i, node in enumerate(nodes):
        if i % 10 == 0: print(f"        Processing node {i}/{len(nodes)}: {node}")
        n_up = len(nx.ancestors(G, node))
        # Downstream path count: number of distinct routes to any outfall
        n_down_paths = 0
        for out in outfalls:
            try:
                paths = list(nx.all_simple_paths(G, node, out))
                n_down_paths += len(paths)
            except: pass
            
        nt = 1 if node.startswith('JI') else (3 if node in outfalls else 0)
        rows.append({
            'node_id': node, 
            'topo_depth': topo_depth[node], 
            'n_upstream_nodes': n_up, # Matched name to train_models.py
            'betweenness': round(betweenness[node], 6),
            'downstream_paths': n_down_paths,
            'node_type_code': nt, 
            'is_high_risk': 1 if node in high_risk_nodes else 0,
            'prior_contam_prob': 2.0 if node in high_risk_nodes else 1.0
        })
    df = pd.DataFrame(rows).set_index('node_id')
    df['prior_contam_prob'] /= df['prior_contam_prob'].sum()
    return df, topo_depth

# ── 3. Scenario Worker ────────────────────────────────────────────────────────
def worker_run_scenario(args):
    (scen_id, base_inp, src, mass_kg, duration_hrs, start_offset_hrs, carrier_flow, node_ids, threshold) = args
    global G_TOPOLOGY
    
    unique_id = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_inp = os.path.join(tmp_dir, f"_scen_{unique_id}.inp")
    
    try:
        vol_m3 = carrier_flow * (duration_hrs * 3600)
        conc_mg_l = (mass_kg * 1e6) / vol_m3
        ts_flow = f'F_{src}'
        ts_conc = f'C_{src}'
        
        with open(base_inp) as f: content = f.read()
        
        # Strictly 12-hour simulation
        total_sim_hrs = 12.0
        start_dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
        end_dt = start_dt + datetime.timedelta(hours=total_sim_hrs)
        end_date_str = end_dt.strftime("%m/%d/%Y")
        end_time_str = end_dt.strftime("%H:%M:%S")

        # Timeseries
        ts_data = f"{ts_flow} 0:00 0\n{ts_flow} {format_time(start_offset_hrs)} {carrier_flow}\n" \
                  f"{ts_flow} {format_time(start_offset_hrs+duration_hrs)} {carrier_flow}\n" \
                  f"{ts_flow} {format_time(start_offset_hrs+duration_hrs+0.01)} 0\n" \
                  f"{ts_conc} 0:00 0\n{ts_conc} {format_time(start_offset_hrs)} {conc_mg_l}\n" \
                  f"{ts_conc} {format_time(start_offset_hrs+duration_hrs)} {conc_mg_l}\n" \
                  f"{ts_conc} {format_time(start_offset_hrs+duration_hrs+0.01)} 0\n"
        
        if '[TIMESERIES]' in content:
            content = content.replace('[TIMESERIES]', f'[TIMESERIES]\n{ts_data}')
        else:
            content += f"\n[TIMESERIES]\n{ts_data}\n"

        inflow = f"{src} FLOW {ts_flow} DIRECT 1.0 1.0\n{src} tracer {ts_conc} CONCEN 1.0 1.0\n"
        if '[INFLOWS]' in content:
            content = content.replace('[INFLOWS]', f'[INFLOWS]\n{inflow}')
        else:
            content += f"\n[INFLOWS]\n{inflow}\n"

        with open(tmp_inp, 'w') as f: f.write(content)

        results = []
        import networkx as nx
        with Simulation(tmp_inp) as sim:
            nodes_obj = Nodes(sim)
            peaks = {nid: 0.0 for nid in node_ids}
            p_step = {nid: 0 for nid in node_ids}
            i_sum = {nid: 0.0 for nid in node_ids}
            steps = 0
            
            for _ in sim:
                steps += 1
                for nid in node_ids:
                    node = nodes_obj[nid]
                    c = node.pollut_quality['tracer']
                    if c > peaks[nid]: peaks[nid] = c; p_step[nid] = steps
                    i_sum[nid] += abs(node.total_inflow)

            for nid in node_ids:
                # Calculate dist_src (pipe-segment distance from source)
                dist = -1
                if G_TOPOLOGY:
                    try:
                        dist = nx.shortest_path_length(G_TOPOLOGY, src, nid)
                    except: pass

                results.append({
                    'scen_id': scen_id, 'src_node': src, 'node_id': nid,
                    'dist_src': dist,
                    'peak_conc': round(peaks[nid], 4),
                    't_peak_min': p_step[nid] if peaks[nid] > 0 else None,
                    'mean_flow_m3s': round((i_sum[nid] / max(steps, 1)) * CFS_TO_M3S, 6),
                    'detected': 1 if peaks[nid] >= threshold else 0
                })
        # Results are collected, now return outside the Simulation context
        pass 
    except Exception as e:
        return None
    finally:
        # Retry loop for Windows file handle release
        for _ in range(5):
            try:
                if os.path.exists(tmp_inp): 
                    os.remove(tmp_inp)
                break
            except PermissionError:
                time.sleep(0.1)
                
    return results

# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  default=config['dataset']['model_path'] if config else './dataset/Examples/Example8.inp')
    p.add_argument('--n_scenarios', type=int, default=config['dataset']['n_scenarios'] if config else 100)
    p.add_argument('--output_dir',  default=config['dataset']['output_dir'] if config else './output')
    p.add_argument('--workers',     type=int, default=config['dataset']['parallel'].get('n_workers') if config and config['dataset']['parallel'].get('n_workers') else os.cpu_count())
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    random.seed(42); np.random.seed(42)

    topo_df, topo_depth = build_topology_features(a.model_path, config['dataset'].get('high_risk_nodes', []))
    
    # Set global graph for workers
    import networkx as nx
    G = nx.DiGraph()
    with open(a.model_path) as f:
        section = None
        for line in f:
            s = line.strip().upper()
            if not s or s.startswith(';'): continue
            if s.startswith('['): section = s.strip('[]'); continue
            parts = line.split()
            if section in ['CONDUITS', 'PUMPS', 'WEIRS', 'ORIFICES'] and len(parts) >= 3:
                G.add_edge(parts[1], parts[2])
    global G_TOPOLOGY
    G_TOPOLOGY = G

    node_ids = list(topo_df.index)
    exclude = config['dataset'].get('exclude_sources', [])
    candidates = [n for n in node_ids if n not in exclude]
    
    probs = [topo_df.loc[n, 'prior_contam_prob'] for n in candidates]
    probs = np.array(probs) / sum(probs)

    tasks = []
    for i in range(a.n_scenarios):
        src = np.random.choice(candidates, p=probs)
        mass = round(random.uniform(*config['dataset'].get('mass_range', [0.01, 0.5])), 3)
        dur = round(random.uniform(*config['dataset'].get('duration_range', [0.25, 3.0])), 2)
        start = round(random.uniform(*config['dataset'].get('start_range', [0.0, 6.0])), 2)
        tasks.append((f"{i+1:04d}", a.model_path, src, mass, dur, start, 
                      config['dataset'].get('carrier_flow', 0.005), node_ids, 
                      config['dataset'].get('threshold', 5.0)))

    print(f"Generating {a.n_scenarios} scenarios using {a.workers} workers...")
    sys.stdout.flush()
    
    all_results = []
    start_time = time.time()
    
    try:
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as executor:
                for idx, res in enumerate(executor.map(worker_run_scenario, tasks)):
                    if res: all_results.extend(res)
                    c = idx + 1
                    if c % max(1, a.n_scenarios // 100) == 0 or c == a.n_scenarios:
                        elapsed = time.time() - start_time
                        rate = c / elapsed
                        rem_sec = (a.n_scenarios - c) / rate if rate > 0 else 0
                        print(f"  Progress: {c}/{a.n_scenarios} | Rate: {rate:.2f} scen/s", end='\r')
        else:
            for idx, task in enumerate(tasks):
                res = worker_run_scenario(task)
                if res: all_results.extend(res)
                c = idx + 1
                elapsed = time.time() - start_time
                rate = c / elapsed
                print(f"  Progress: {c}/{a.n_scenarios} | Rate: {rate:.2f} scen/s", end='\r')
    except KeyboardInterrupt:
        print("\n\nProcess cancelled by user. Saving partial results...")
    
    print() # Newline after progress bar
    if not all_results:
        print("No results generated.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(a.output_dir, 'raw_scenarios.csv'), index=False)
    
    # Aggregation
    grp = df.groupby('node_id')
    node_agg = pd.DataFrame({
        'detection_freq': grp['detected'].mean().round(4),
        'peak_conc_mean': grp['peak_conc'].mean().round(4),
        'peak_conc_std':  grp['peak_conc'].std().round(4),
        'time_to_peak_mean': grp['t_peak_min'].mean().round(2),
        'mean_flow_m3s':  grp['mean_flow_m3s'].mean().round(6),
        'mean_dist_src':  grp['dist_src'].mean().round(2),
    })
    final_df = topo_df.join(node_agg, how='left').reset_index()
    final_df.to_csv(os.path.join(a.output_dir, 'node_features.csv'), index=False)
    print(f"\nDone. Features saved to {a.output_dir}/node_features.csv")

if __name__ == '__main__':
    main()
