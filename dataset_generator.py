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
            if section == 'CONDUITS' and len(parts) >= 3: G.add_edge(parts[1], parts[2])
            if section == 'OUTFALLS' and len(parts) >= 1: outfalls.append(parts[0])

    topo_depth = {}
    nodes = list(G.nodes())
    for node in nodes:
        min_d = 999
        for out in outfalls:
            try:
                d = nx.shortest_path_length(G, node, out); 
                if d < min_d: min_d = d
            except: pass
        topo_depth[node] = min_d if min_d != 999 else 0

    rows = []
    for node in nodes:
        n_up = len(nx.ancestors(G, node))
        nt = 1 if node.startswith('JI') else (3 if node in outfalls else 0)
        rows.append({
            'node_id': node, 'topo_depth': topo_depth[node], 'n_upstream': n_up,
            'node_type_code': nt, 'is_high_risk': 1 if node in high_risk_nodes else 0,
            'prior_contam_prob': 2.0 if node in high_risk_nodes else 1.0
        })
    df = pd.DataFrame(rows).set_index('node_id')
    df['prior_contam_prob'] /= df['prior_contam_prob'].sum()
    return df, topo_depth

# ── 3. Scenario Worker ────────────────────────────────────────────────────────
def worker_run_scenario(args):
    (scen_id, base_inp, src, mass_kg, duration_hrs, start_offset_hrs, carrier_flow, node_ids, threshold) = args
    
    unique_id = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    tmp_inp = os.path.join(tmp_dir, f"_scen_{unique_id}.inp")
    tmp_rpt = os.path.join(tmp_dir, f"_scen_{unique_id}.rpt")
    tmp_out = os.path.join(tmp_dir, f"_scen_{unique_id}.out")
    
    try:
        # 1. Build Inp
        vol_m3 = carrier_flow * (duration_hrs * 3600)
        conc_mg_l = (mass_kg * 1e6) / vol_m3
        
        ts_flow = f'F_{src}'
        ts_conc = f'C_{src}'
        
        with open(base_inp) as f: content = f.read()
        
        # Path fixes
        base_dir = os.path.dirname(os.path.abspath(base_inp))
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.strip() and not line.startswith('[') and not line.startswith(';'):
                parts = line.split()
                if 'FILE' in [p.upper() for p in parts]:
                    f_idx = [p.upper() for p in parts].index('FILE')
                    if f_idx + 1 < len(parts):
                        fn_raw = parts[f_idx+1]
                        fn = fn_raw.strip('"')
                        if not os.path.isabs(fn):
                            line = line.replace(fn_raw, f'"{os.path.join(base_dir, fn)}"')
            new_lines.append(line)
        content = '\n'.join(new_lines)

        # Optimize simulation window to exactly match injection + wash-out buffer (3 hours)
        buffer_hrs = 3.0
        total_sim_hrs = start_offset_hrs + duration_hrs + buffer_hrs
        
        start_dt = datetime.datetime(1968, 1, 1, 0, 0, 0)
        end_dt = start_dt + datetime.timedelta(hours=total_sim_hrs)
        end_date_str = end_dt.strftime("%m/%d/%Y")
        end_time_str = end_dt.strftime("%H:%M:%S")

        if '[OPTIONS]' in content:
            opt_parts = content.split('[OPTIONS]', 1)
            rest = opt_parts[1].split('[', 1)
            opts = rest[0].split('\n')
            new_opts = []
            for o in opts:
                if 'START_DATE' in o.upper():   new_opts.append("START_DATE           01/01/1968")
                elif 'START_TIME' in o.upper(): new_opts.append("START_TIME           00:00:00")
                elif 'END_DATE' in o.upper():   new_opts.append(f"END_DATE             {end_date_str}")
                elif 'END_TIME' in o.upper():   new_opts.append(f"END_TIME             {end_time_str}")
                else: new_opts.append(o)
            content = opt_parts[0] + '[OPTIONS]' + '\n'.join(new_opts) + (('[' + rest[1]) if len(rest)>1 else '')

        # Pollutants
        if '[POLLUTANTS]' not in content:
            pb = "[POLLUTANTS]\nCONTAM MG/L 0 0 0 0 NO * 0 0 0\n\n"
            content = content.replace('[JUNCTIONS]', pb + '[JUNCTIONS]')

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

        # Inflows
        inflow = f"{src} FLOW {ts_flow} DIRECT 1.0 1.0\n{src} CONTAM {ts_conc} CONCEN 1.0 1.0\n"
        if '[INFLOWS]' in content:
            content = content.replace('[INFLOWS]', f'[INFLOWS]\n{inflow}')
        else:
            content += f"\n[INFLOWS]\n{inflow}\n"

        with open(tmp_inp, 'w') as f: f.write(content)

        # 2. Run
        results = []
        with Simulation(tmp_inp) as sim:
            nodes_obj = Nodes(sim)
            links_obj = Links(sim)
            n2l = {nid: [l.linkid for l in links_obj if l.inlet_node == nid or l.outlet_node == nid] for nid in node_ids}
            
            peaks = {nid: 0.0 for nid in node_ids}
            p_step = {nid: 0 for nid in node_ids}
            i_sum = {nid: 0.0 for nid in node_ids}
            v_sum = {nid: 0.0 for nid in node_ids}
            steps = 0
            
            for _ in sim:
                steps += 1
                for nid in node_ids:
                    node = nodes_obj[nid]
                    c = node.pollut_quality['CONTAM']
                    if c > peaks[nid]: peaks[nid] = c; p_step[nid] = steps
                    i_sum[nid] += abs(node.total_inflow)
                    v_node = 0.0
                    if n2l[nid]:
                        for lid in n2l[nid]:
                            l = links_obj[lid]
                            try:
                                a = getattr(l, 'ups_xsection_area', 0.0) or getattr(l, 'ds_xsection_area', 0.0)
                                if a > 0.001: v_node += abs(l.flow) / a
                            except: pass
                        v_sum[nid] += (v_node / len(n2l[nid]))

            for nid in node_ids:
                results.append({
                    'scen_id': scen_id, 'src_node': src, 'node_id': nid,
                    'peak_conc': round(peaks[nid], 4),
                    't_peak_min': p_step[nid] if peaks[nid] > 0 else None,
                    'mean_flow_m3s': round((i_sum[nid] / max(steps, 1)) * CFS_TO_M3S, 6),
                    'mean_vel_ms': round((v_sum[nid] / max(steps, 1)) * 0.3048, 6),
                    'detected': 1 if peaks[nid] >= threshold else 0
                })
        
        return results
    except Exception as e:
        # print(f"Error in worker {scen_id}: {e}")
        return None
    finally:
        for f in [tmp_inp, tmp_rpt, tmp_out]:
            try: 
                if os.path.exists(f): os.remove(f)
            except: pass

# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  default=config['dataset']['model_path'] if config else './dataset/Examples/Example8.inp')
    p.add_argument('--n_scenarios', type=int, default=config['dataset']['n_scenarios'] if config else 100)
    p.add_argument('--output_dir',  default=config['dataset']['output_dir'] if config else './output')
    p.add_argument('--workers',     type=int, default=os.cpu_count())
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    random.seed(42); np.random.seed(42)

    topo_df, topo_depth = build_topology_features(a.model_path, config['dataset'].get('high_risk_nodes', []))
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
    print("Press Ctrl+C to safely cancel and save progress.\n")
    
    all_results = []
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as executor:
            for idx, res in enumerate(executor.map(worker_run_scenario, tasks)):
                if res: all_results.extend(res)
                
                # Timer and ETA calculation
                c = idx + 1
                if c % max(1, a.n_scenarios // 100) == 0 or c == a.n_scenarios:
                    elapsed = time.time() - start_time
                    rate = c / elapsed
                    rem_sec = (a.n_scenarios - c) / rate if rate > 0 else 0
                    
                    el_str = str(datetime.timedelta(seconds=int(elapsed)))
                    rem_str = str(datetime.timedelta(seconds=int(rem_sec)))
                    
                    print(f"  Progress: {c}/{a.n_scenarios} | "
                          f"Elapsed: {el_str} | ETA: {rem_str} | Rate: {rate:.2f} scen/s", end='\r')
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
        'mean_vel_ms':   grp['mean_vel_ms'].mean().round(6),
    })
    final_df = topo_df.join(node_agg, how='left').reset_index()
    final_df.to_csv(os.path.join(a.output_dir, 'node_features.csv'), index=False)
    print(f"\nDone. Features saved to {a.output_dir}/node_features.csv")

if __name__ == '__main__':
    main()
