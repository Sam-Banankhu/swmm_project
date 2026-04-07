"""
train_eval_pipeline.py
======================
Hybrid AI Sensor Placement — Two-Stage Training & Evaluation Pipeline
Mhango, S.B. and Sambito, M. (2026)

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 — ML Model Comparison                                   ║
║  Train three architectures on SWMM scenario data and compare     ║
║  how well each predicts node-level detection frequency.           ║
║                                                                   ║
║  Model A: XGBoost (gradient boosting baseline)                   ║
║  Model B: MLP with dropout (feedforward neural network)          ║
║  Model C: GCN / GAT (graph neural network, novel contribution)   ║
║                                                                   ║
║  → Best model's output becomes the ML-derived prior              ║
╠══════════════════════════════════════════════════════════════════╣
║  STAGE 2 — BDN Prior Comparison                                  ║
║  Compare the hybrid BDN (ML prior) against all four Sambito      ║
║  et al. (2020) v1.0 priors on convergence speed and sensor       ║
║  placement quality (F1 / F2).                                    ║
║                                                                   ║
║  Prior A: Uniform (non-informative baseline)                     ║
║  Prior B: Topology-based (topo_depth inverse)                    ║
║  Prior C: Wastewater flux (mean_flow_m3s proxy)                  ║
║  Prior D: Contaminant flux (best v1.0 prior — benchmark)         ║
║  Prior ML: ML-derived (this work — proposed improvement)         ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python train_eval_pipeline.py --data_dir ./output
  python train_eval_pipeline.py --data_dir ./output --stage 1   # ML comparison only
  python train_eval_pipeline.py --data_dir ./output --stage 2   # BDN comparison only

Outputs (written to --output_dir, default ./ml_results):
  stage1_ml_comparison.csv      — per-model LOO-CV metrics
  stage1_feature_importance.csv — XGBoost feature importance
  stage2_bdn_comparison.csv     — prior x metric convergence table
  ml_prior.csv                  — normalised P(sensor at node i) ready for BDN
  figures/                      — all PNG plots
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import xgboost as xgb;  HAS_XGB = True
except ImportError:
    HAS_XGB = False;        print("[WARN] xgboost not installed — Model A skipped")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False;      print("[WARN] torch not installed — Model B (MLP) skipped")

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False;        print("[WARN] torch_geometric not installed — Model C (GNN) skipped")

# ── Network constants (Example 8) ─────────────────────────────────────────────
HIGH_RISK_NODES   = {'J4', 'J10', 'JI18'}
EXCLUDE_NODES     = {'O1', 'O2', 'Well'}   # not sensor candidates
TARGET            = 'detection_freq'

TABULAR_FEATURES = [
    # Group 1 — static topology
    'topo_depth', 'n_upstream_nodes', 'betweenness',
    'downstream_paths', 'node_type_code', 'is_high_risk',
    # Group 2 — prior contamination probability
    'prior_contam_prob',
    # Group 3 — Bayesian prior proxies (Prior C / D surrogates)
    'mean_contam_flux',   # approx Prior D: contaminant mass flux
    'mean_waste_flux',    # approx Prior C: wastewater volume flux
    'contam_flux_std',    # variance across sources
    # Group 4 — dynamic simulation features
    'peak_conc_mean', 'peak_conc_std', 'time_to_peak_mean', 'mean_flow_m3s',
]

PALETTE = {
    'Model A (XGBoost)':    '#2196F3',
    'Model B (MLP)':        '#FF9800',
    'Model C (GCN)':        '#9C27B0',
    'Model C (GAT)':        '#F44336',
    'Prior A (Uniform)':    '#9E9E9E',
    'Prior B (Topology)':   '#795548',
    'Prior C (Waste flux)': '#4CAF50',
    'Prior D (Contam flux)':'#FF5722',
    'Prior ML (This work)': '#2196F3',
}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir):
    path = os.path.join(data_dir, 'node_features.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"node_features.csv not found in {data_dir}")

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} nodes, {len(df.columns)} columns")

    # Prior D proxy: detection_freq x peak_conc_mean (nodes intercepting high
    # concentrations consistently -- same signal as Sambito Prior D)
    df['mean_contam_flux'] = df[TARGET] * df['peak_conc_mean']
    mx = df['mean_contam_flux'].max()
    if mx > 0:
        df['mean_contam_flux'] /= mx

    # Prior C proxy: wastewater volume flux
    df['mean_waste_flux'] = df['mean_flow_m3s']

    # Contaminant flux std (cross-source variance)
    df['contam_flux_std'] = df['peak_conc_std']

    # Impute missing time-to-peak (nodes that never detected)
    max_ttp = df['time_to_peak_mean'].max()
    df['time_to_peak_mean'] = df['time_to_peak_mean'].fillna(max_ttp * 1.2)

    df[TARGET] = df[TARGET].clip(0, 1)

    df_elig = df[~df['node_id'].isin(EXCLUDE_NODES)].copy().reset_index(drop=True)
    print(f"  Eligible sensor nodes: {len(df_elig)}  (excluded: {sorted(EXCLUDE_NODES)})")
    print(f"  Detection freq range: [{df_elig[TARGET].min():.3f}, {df_elig[TARGET].max():.3f}]")
    return df, df_elig


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED — EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label=''):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    ndcg   = ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=5)
    top5   = len(set(np.argsort(y_true)[::-1][:5]) &
                 set(np.argsort(y_pred)[::-1][:5])) / 5
    if label:
        print(f"  {label:<30} MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"R2={r2:.4f}  rho={rho:.4f}  NDCG@5={ndcg:.4f}  Top5={top5:.0%}")
    return dict(MAE=mae, RMSE=rmse, R2=r2, Spearman=rho, NDCG5=ndcg, Top5=top5)


def loocv(X, y, factory_fn, label):
    n = len(X); preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m  = factory_fn()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[[i]])[0]
    return compute_metrics(y, preds, label), preds


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MODEL A: XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_A(df_elig, features):
    if not HAS_XGB:
        return None, None, None
    print("\n-- Model A: XGBoost (gradient boosting baseline) ----------")
    X = df_elig[features].values.astype(np.float32)
    y = df_elig[TARGET].values.astype(np.float32)
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0,
                  objective='reg:squarederror', random_state=42, verbosity=0)
    metrics, preds = loocv(X, y, lambda: xgb.XGBRegressor(**params),
                           'Model A (XGBoost)')
    full = xgb.XGBRegressor(**params); full.fit(X, y)
    fi = pd.Series(full.feature_importances_, index=features).sort_values(ascending=False)
    return metrics, preds, fi


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MODEL B: MLP
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_B(df_elig, features, epochs=600):
    if not HAS_TORCH:
        return None, None
    print("\n-- Model B: MLP (feedforward neural network) --------------")

    X_raw = df_elig[features].values.astype(np.float32)
    y_raw = df_elig[TARGET].values.astype(np.float32)
    n     = len(X_raw)

    class _MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    preds = np.zeros(n)
    for i in range(n):
        tr  = [j for j in range(n) if j != i]
        sc  = StandardScaler()
        Xtr = torch.tensor(sc.fit_transform(X_raw[tr]), dtype=torch.float32)
        ytr = torch.tensor(y_raw[tr], dtype=torch.float32)
        Xte = torch.tensor(sc.transform(X_raw[[i]]), dtype=torch.float32)
        model = _MLP(Xtr.shape[1])
        opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = ReduceLROnPlateau(opt, patience=30, factor=0.5, verbose=False)
        best, wait = np.inf, 0
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss = F.mse_loss(model(Xtr), ytr)
            loss.backward(); opt.step(); sched.step(loss)
            if loss.item() < best: best = loss.item(); wait = 0
            else:
                wait += 1
                if wait > 60: break
        model.eval()
        with torch.no_grad():
            preds[i] = model(Xte).item()

    metrics = compute_metrics(y_raw, preds, 'Model B (MLP)')
    return metrics, preds


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MODEL C: GCN + GAT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pyg_graph(df_all, features, raw_csv_path):
    node_ids = list(df_all['node_id'])
    id2idx   = {n: i for i, n in enumerate(node_ids)}
    edges    = set()
    if raw_csv_path and os.path.exists(raw_csv_path):
        raw = pd.read_csv(raw_csv_path, usecols=['src_node', 'node_id', 'dist_src'])
        for _, r in raw[raw['dist_src'] == 1][['src_node','node_id']].drop_duplicates().iterrows():
            if r['src_node'] in id2idx and r['node_id'] in id2idx:
                u, v = id2idx[r['src_node']], id2idx[r['node_id']]
                edges.update([(u, v), (v, u)])
    if not edges:
        depths = {id2idx[r['node_id']]: r['topo_depth'] for _, r in df_all.iterrows()}
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i != j and abs(depths.get(i, 0) - depths.get(j, 0)) == 1:
                    edges.add((i, j))
    ei = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    feat_cols = [c for c in features if c in df_all.columns]
    X = StandardScaler().fit_transform(df_all[feat_cols].values.astype(np.float32))
    y = torch.tensor(df_all[TARGET].clip(0, 1).values.astype(np.float32))
    train_mask = torch.tensor([n not in EXCLUDE_NODES for n in node_ids], dtype=torch.bool)
    return Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=ei, y=y, train_mask=train_mask,
                node_ids=node_ids)


def _gnn_loocv(data, ModelClass, label, epochs=800, lr=5e-3, **kw):
    tr_idx = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    preds  = {}
    for held in tr_idx:
        fold_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        for i in tr_idx:
            if i != held: fold_mask[i] = True
        m   = ModelClass(data.x.shape[1], **kw)
        opt = Adam(m.parameters(), lr=lr, weight_decay=1e-4)
        best, wait = np.inf, 0
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            out  = m(data.x, data.edge_index)
            loss = F.mse_loss(out[fold_mask], data.y[fold_mask])
            loss.backward(); opt.step()
            if loss.item() < best: best = loss.item(); wait = 0
            else:
                wait += 1
                if wait > 80: break
        m.eval()
        with torch.no_grad():
            preds[held] = m(data.x, data.edge_index)[held].item()
    idx_list  = list(preds.keys())
    y_true_np = data.y[idx_list].numpy()
    y_pred_np = np.array([preds[i] for i in idx_list])
    metrics   = compute_metrics(y_true_np, y_pred_np, label)
    # Full fit
    m_full = ModelClass(data.x.shape[1], **kw)
    opt_f  = Adam(m_full.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        m_full.train(); opt_f.zero_grad()
        out  = m_full(data.x, data.edge_index)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt_f.step()
    m_full.eval()
    with torch.no_grad():
        full_out = m_full(data.x, data.edge_index).numpy()
    return metrics, full_out


def run_model_C(df_all, features, raw_csv_path):
    if not (HAS_PYG and HAS_TORCH):
        return {}, {}
    print("\n-- Model C: GCN + GAT (graph neural networks) -------------")

    class _GCN(nn.Module):
        def __init__(self, d, hidden=64, dropout=0.3):
            super().__init__()
            self.c1 = GCNConv(d, hidden)
            self.c2 = GCNConv(hidden, hidden // 2)
            self.c3 = GCNConv(hidden // 2, hidden // 4)
            self.fc = nn.Linear(hidden // 4, 1)
            self.dr = nn.Dropout(dropout)
        def forward(self, x, ei):
            x = F.relu(self.c1(x, ei)); x = self.dr(x)
            x = F.relu(self.c2(x, ei)); x = self.dr(x)
            x = F.relu(self.c3(x, ei))
            return self.fc(x).squeeze(-1)

    class _GAT(nn.Module):
        def __init__(self, d, hidden=32, heads=4, dropout=0.3):
            super().__init__()
            self.c1 = GATConv(d, hidden, heads=heads, dropout=dropout)
            self.c2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
            self.fc = nn.Linear(hidden, 1)
            self.dr = nn.Dropout(dropout)
        def forward(self, x, ei):
            x = F.elu(self.c1(x, ei)); x = self.dr(x)
            x = F.elu(self.c2(x, ei))
            return self.fc(x).squeeze(-1)

    graph = _build_pyg_graph(df_all, features, raw_csv_path)
    m_gcn, p_gcn = _gnn_loocv(graph, _GCN, 'Model C (GCN)')
    m_gat, p_gat = _gnn_loocv(graph, _GAT, 'Model C (GAT)', hidden=32, heads=4)
    node_ids = graph.node_ids
    return (
        {'Model C (GCN)': m_gcn, 'Model C (GAT)': m_gat},
        {'Model C (GCN)': (p_gcn, node_ids), 'Model C (GAT)': (p_gat, node_ids)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DERIVE ML PRIOR FROM BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def derive_ml_prior(df_all, best_preds_full, node_ids_full=None):
    all_nodes = list(df_all['node_id'])
    eligible  = np.array([n not in EXCLUDE_NODES for n in all_nodes], dtype=float)
    if node_ids_full is not None:
        idx_map = {n: i for i, n in enumerate(node_ids_full)}
        raw = np.array([best_preds_full[idx_map[n]] if n in idx_map else 0.0
                        for n in all_nodes])
    else:
        raw = np.array(best_preds_full, dtype=float)
    raw   = np.clip(raw, 0, None) * eligible
    total = raw.sum()
    prior = raw / total if total > 0 else eligible / eligible.sum()
    return pd.DataFrame({'node_id': all_nodes,
                         'ml_prior': prior,
                         'detection_freq': df_all[TARGET].clip(0, 1).values,
                         'is_eligible': eligible.astype(int)})


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FIVE PRIOR DISTRIBUTIONS (Priors A-D + ML)
# ═══════════════════════════════════════════════════════════════════════════════

def build_v1_priors(df_all, ml_prior_df):
    """
    Reproduce Sambito et al. (2020) Table 2 prior logic for eligible nodes.
      A — Uniform
      B — Topology (inverse topo_depth)
      C — Wastewater flux (mean_flow_m3s)
      D — Contaminant flux (mean_contam_flux, the best v1.0 prior)
      ML — ML-derived (this work)
    """
    elig = df_all[~df_all['node_id'].isin(EXCLUDE_NODES)].copy()

    def norm(v):
        v = np.clip(v, 0, None)
        return v / v.sum() if v.sum() > 0 else np.ones(len(v)) / len(v)

    priors = {
        'Prior A (Uniform)':
            pd.Series(norm(np.ones(len(elig))), index=elig['node_id']),
        'Prior B (Topology)':
            pd.Series(norm((elig['topo_depth'].max() + 1 - elig['topo_depth']).values),
                      index=elig['node_id']),
        'Prior C (Waste flux)':
            pd.Series(norm(elig['mean_flow_m3s'].values), index=elig['node_id']),
        'Prior D (Contam flux)':
            pd.Series(norm(elig['mean_contam_flux'].values), index=elig['node_id']),
    }
    ml_elig = ml_prior_df[ml_prior_df['is_eligible'] == 1].set_index('node_id')
    priors['Prior ML (This work)'] = pd.Series(
        norm(ml_elig.reindex(elig['node_id'])['ml_prior'].fillna(0).values),
        index=elig['node_id'])
    return priors


def simulate_bdn_convergence(df_elig, priors, n_scenarios=500, seed=42):
    """
    Simulate Bayesian posterior updating for each prior (Sambito 2020 structure).

    At each update step a contamination scenario is sampled. The posterior is
    updated as:  P_new(s) ∝ P_old(s) x L(detect | source=s)
    where L is approximated by each node's empirical detection_freq.

    Convergence = first step where the top-ranked node is stable for 10
    consecutive updates (mirrors the Sambito 2020 convergence criterion).
    """
    rng       = np.random.default_rng(seed)
    node_ids  = df_elig['node_id'].values
    det_freqs = df_elig[TARGET].values
    n_nodes   = len(node_ids)
    results   = {}

    for name, prior_series in priors.items():
        p         = prior_series.reindex(node_ids).fillna(0).values.astype(float)
        p        /= p.sum()
        posterior = p.copy()
        history   = []
        conv_step = None

        for step in range(1, n_scenarios + 1):
            src_idx    = rng.choice(n_nodes, p=det_freqs / det_freqs.sum())
            likelihood = det_freqs.copy()
            likelihood[src_idx] = 1.0          # source node always detects
            posterior  = posterior * likelihood
            total      = posterior.sum()
            if total > 0: posterior /= total
            history.append(node_ids[np.argmax(posterior)])
            if step >= 10 and len(set(history[-10:])) == 1 and conv_step is None:
                conv_step = step

        if conv_step is None: conv_step = n_scenarios
        top3_idx = np.argsort(posterior)[::-1][:3]
        top3     = list(node_ids[top3_idx])
        f1       = float(posterior[top3_idx[0]])
        f2       = float(det_freqs[top3_idx].mean())

        results[name] = {
            'Convergence (steps)':      conv_step,
            'Top sensor':               top3[0],
            'Top-3 sensors':            ', '.join(top3),
            'F1 (isolation)':           round(f1, 4),
            'F2 (reliability)':         round(f2, 4),
            'Final posterior entropy':  round(-np.sum(posterior * np.log(posterior + 1e-12)), 4),
        }
        print(f"  {name:<28}  converge={conv_step:>4}  "
              f"top={top3[0]:<6}  F1={f1:.4f}  F2={f2:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_stage1_comparison(metrics_df, fig_dir):
    cols   = ['MAE', 'RMSE', 'R2', 'Spearman', 'NDCG5', 'Top5']
    labels = ['MAE (lower better)', 'RMSE (lower better)', 'R² (higher better)',
              'Spearman rho (higher better)', 'NDCG@5 (higher better)',
              'Top-5 overlap (higher better)']
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 5))
    fig.suptitle('Stage 1 — ML Model Comparison (Leave-One-Out CV, n=28)\n'
                 'Mhango & Sambito (2026)', fontsize=13, fontweight='bold', y=1.02)
    for ax, col, lbl in zip(axes, cols, labels):
        vals   = metrics_df[col].values
        colors = [PALETTE.get(m, '#607D8B') for m in metrics_df['Model']]
        bars   = ax.bar(metrics_df['Model'], vals, color=colors, edgecolor='white')
        ax.set_title(lbl, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=40, labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)
        if col in ('R2', 'Spearman', 'NDCG5', 'Top5'):
            ax.set_ylim(min(0, min(vals) - 0.1), 1.18)
            ax.axhline(1.0, color='grey', ls='--', lw=0.8, alpha=0.5)
        else:
            ax.set_ylim(0, max(vals) * 1.4 + 0.01)
        ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage1_ml_comparison.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


def fig_feature_importance(fi, fig_dir):
    group_map = {
        'topo_depth': 'Group 1 Topology', 'n_upstream_nodes': 'Group 1 Topology',
        'betweenness': 'Group 1 Topology', 'downstream_paths': 'Group 1 Topology',
        'node_type_code': 'Group 1 Topology', 'is_high_risk': 'Group 2 Prior P(source)',
        'prior_contam_prob': 'Group 2 Prior P(source)',
        'mean_contam_flux': 'Group 3 Prior D proxy', 'contam_flux_std': 'Group 3 Prior D proxy',
        'mean_waste_flux': 'Group 3 Prior C proxy',
        'peak_conc_mean': 'Group 4 Dynamic', 'peak_conc_std': 'Group 4 Dynamic',
        'time_to_peak_mean': 'Group 4 Dynamic', 'mean_flow_m3s': 'Group 4 Dynamic',
    }
    gc = {'Group 1 Topology': '#1E88E5', 'Group 2 Prior P(source)': '#43A047',
          'Group 3 Prior D proxy': '#8E24AA', 'Group 3 Prior C proxy': '#6D4C41',
          'Group 4 Dynamic': '#FB8C00'}
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = [gc.get(group_map.get(f, ''), '#607D8B') for f in fi.index]
    fi.plot.barh(ax=ax, color=colors, edgecolor='white', linewidth=0.7)
    ax.invert_yaxis()
    ax.set_title('Stage 1 — XGBoost Feature Importance (Gain)\nMhango & Sambito (2026)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Relative importance (gain)')
    for lbl in ax.get_yticklabels():
        lbl.set_color(gc.get(group_map.get(lbl.get_text(), ''), 'black'))
    handles = [mpatches.Patch(color=c, label=g) for g, c in gc.items()]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage1_feature_importance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


def fig_predictions_scatter(df_elig, preds_dict, fig_dir):
    models = {k: v for k, v in preds_dict.items() if v is not None}
    if not models: return
    y_true = df_elig[TARGET].clip(0, 1).values
    hr     = df_elig['is_high_risk'].values
    labels = df_elig['node_id'].values
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    fig.suptitle('Stage 1 — Predicted vs Actual Detection Frequency (LOO-CV)\n'
                 'Mhango & Sambito (2026)', fontsize=13, fontweight='bold', y=1.02)
    for ax, (name, preds) in zip(axes[0], models.items()):
        col = PALETTE.get(name, '#607D8B')
        ax.scatter(y_true, preds, c=hr, cmap='bwr_r', s=80,
                   edgecolors=col, linewidths=1.2, zorder=3)
        for idx in np.argsort(preds)[::-1][:3]:
            ax.annotate(labels[idx], (y_true[idx], preds[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        lim = max(y_true.max(), preds.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
        ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
        ax.set_xlabel('Actual detection_freq'); ax.set_ylabel('Predicted')
        ax.set_title(name, fontsize=11, fontweight='bold', color=col)
        rho, _ = spearmanr(y_true, preds)
        r2     = r2_score(y_true, preds)
        ax.text(0.05, 0.92, f'rho = {rho:.3f}\nR2 = {r2:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc='white', ec='grey', alpha=0.85, boxstyle='round'))
        ax.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage1_predictions_scatter.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


def fig_ml_prior(prior_df, fig_dir):
    elig = prior_df[prior_df['is_eligible'] == 1].sort_values('ml_prior', ascending=False)
    x = np.arange(len(elig)); w = 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, elig['ml_prior'],       w, label='ML Prior  P(sensor at node)',
           color='#2196F3', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, elig['detection_freq'], w, label='Empirical detection_freq',
           color='#FF9800', alpha=0.85, edgecolor='white')
    for i, (_, row) in enumerate(elig.iterrows()):
        if row['node_id'] in HIGH_RISK_NODES:
            ax.axvspan(i - 0.5, i + 0.5, color='#FFCDD2', alpha=0.4, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(elig['node_id'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Probability / Frequency')
    ax.set_title('Stage 1 — ML-Derived BDN Prior vs Empirical Detection Frequency\n'
                 'Mhango & Sambito (2026)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.25)
    ax.text(0.01, 0.97, 'Shaded columns = high-risk injection nodes (J4, J10, JI18)',
            transform=ax.transAxes, fontsize=8, va='top', color='#C62828')
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage1_ml_prior.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


def fig_bdn_convergence(bdn_results, fig_dir):
    names  = list(bdn_results.keys())
    conv   = [bdn_results[n]['Convergence (steps)'] for n in names]
    f1     = [bdn_results[n]['F1 (isolation)']      for n in names]
    f2     = [bdn_results[n]['F2 (reliability)']    for n in names]
    colors = [PALETTE.get(n, '#607D8B') for n in names]

    fig = plt.figure(figsize=(15, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(names, conv, color=colors, edgecolor='white')
    ax1.set_title('Convergence Speed\n(fewer steps = better)', fontweight='bold')
    ax1.set_ylabel('Simulation steps to convergence')
    ax1.tick_params(axis='x', rotation=35, labelsize=8)
    # Sambito (2020) Prior D benchmark: converges in ~20 steps
    ax1.axhline(20, color='#FF5722', ls='--', lw=1.5, alpha=0.8,
                label='Prior D target (Sambito 2020: ~20 steps)')
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, conv):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
                 ha='center', va='bottom', fontsize=9)
    ax1.grid(axis='y', alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(names, f1, color=colors, edgecolor='white')
    ax2.set_title('F1: Isolation Likelihood\n(higher = better)', fontweight='bold')
    ax2.set_ylabel('F1 score'); ax2.set_ylim(0, max(f1) * 1.35)
    ax2.tick_params(axis='x', rotation=35, labelsize=8)
    for bar, v in zip(bars2, f1):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(names, f2, color=colors, edgecolor='white')
    ax3.set_title('F2: Detection Reliability\n(higher = better)', fontweight='bold')
    ax3.set_ylabel('F2 score'); ax3.set_ylim(0, max(f2) * 1.35)
    ax3.tick_params(axis='x', rotation=35, labelsize=8)
    for bar, v in zip(bars3, f2):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9)
    ax3.grid(axis='y', alpha=0.25)

    fig.suptitle('Stage 2 — BDN Prior Comparison\n'
                 'Priors A-D (Sambito 2020) vs Prior ML (This Work) — Mhango & Sambito (2026)',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage2_bdn_convergence.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


def fig_prior_distributions(priors, fig_dir):
    names      = list(priors.keys())
    node_order = priors['Prior D (Contam flux)'].sort_values(ascending=False).index
    fig, axes  = plt.subplots(len(names), 1, figsize=(14, 2.6 * len(names)), sharex=True)
    fig.suptitle('Stage 2 — Prior Probability Distributions (Priors A-D vs ML)\n'
                 'Mhango & Sambito (2026)', fontsize=13, fontweight='bold')
    for ax, name in zip(axes, names):
        vals  = priors[name].reindex(node_order).values
        color = PALETTE.get(name, '#607D8B')
        bars  = ax.bar(node_order, vals, color=color, alpha=0.85, edgecolor='white')
        for i, n in enumerate(node_order):
            if n in HIGH_RISK_NODES:
                bars[i].set_edgecolor('#C62828'); bars[i].set_linewidth(2.0)
        ax.set_ylabel('P(node)', fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold', color=color, pad=3)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    p = os.path.join(fig_dir, 'stage2_prior_distributions.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved -> {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(data_dir, output_dir, stage):
    print("=" * 68)
    print("  Hybrid AI Sensor Placement — Two-Stage Training & Evaluation")
    print("  Mhango, S.B. and Sambito, M. (2026)")
    print("=" * 68)
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    raw_csv = os.path.join(data_dir, 'raw_scenarios.csv')

    print("\n[Data] Loading node_features.csv...")
    df_all, df_elig = load_data(data_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — ML MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    if stage in (0, 1):
        print("\n" + "=" * 68)
        print("  STAGE 1 — ML Model Comparison")
        print("  Three architectures trained on SWMM scenario data.")
        print("  Best model output becomes the ML-derived BDN prior.")
        print("=" * 68)

        all_metrics   = {}
        tabular_preds = {}   # eligible-node predictions only (28 nodes)

        # Model A — XGBoost
        mA, pA, fi = run_model_A(df_elig, TABULAR_FEATURES)
        if mA:
            all_metrics['Model A (XGBoost)']   = mA
            tabular_preds['Model A (XGBoost)'] = pA

        # Model B — MLP
        mB, pB = run_model_B(df_elig, TABULAR_FEATURES)
        if mB:
            all_metrics['Model B (MLP)']   = mB
            tabular_preds['Model B (MLP)'] = pB

        # Model C — GCN + GAT
        gnn_metrics, gnn_preds = run_model_C(df_all, TABULAR_FEATURES, raw_csv)
        all_metrics.update(gnn_metrics)

        if not all_metrics:
            print("\n[ERROR] No models trained — install xgboost: pip install xgboost")
            return

        # Stage 1 summary
        print("\n-- Stage 1 Summary -----------------------------------------")
        s1_df = (pd.DataFrame(all_metrics).T.reset_index()
                   .rename(columns={'index': 'Model'})
                   .sort_values('Spearman', ascending=False))
        print(s1_df.to_string(index=False, float_format='{:.4f}'.format))
        s1_path = os.path.join(output_dir, 'stage1_ml_comparison.csv')
        s1_df.to_csv(s1_path, index=False)
        print(f"\n  Saved -> {s1_path}")
        if fi is not None:
            fi_path = os.path.join(output_dir, 'stage1_feature_importance.csv')
            fi.to_csv(fi_path, header=['importance'])
            print(f"  Saved -> {fi_path}")

        # Select best model -> ML prior
        best_model = s1_df.iloc[0]['Model']
        print(f"\n  Best model: {best_model}  "
              f"(Spearman rho = {s1_df.iloc[0]['Spearman']:.4f})")

        if best_model in gnn_preds:
            arr, nids = gnn_preds[best_model]
            prior_df = derive_ml_prior(df_all, arr, nids)
        elif best_model in tabular_preds:
            elig_ids = list(df_elig['node_id'])
            pred_s   = pd.Series(tabular_preds[best_model], index=elig_ids)
            full_arr = df_all['node_id'].map(pred_s).fillna(0).values
            prior_df = derive_ml_prior(df_all, full_arr)
        else:
            full_arr = df_all[TARGET].clip(0, 1).values
            prior_df = derive_ml_prior(df_all, full_arr)

        prior_path = os.path.join(output_dir, 'ml_prior.csv')
        prior_df.to_csv(prior_path, index=False)
        print(f"  ML prior -> {prior_path}")

        print("\n  Stage 1 figures...")
        fig_stage1_comparison(s1_df, fig_dir)
        if fi is not None:
            fig_feature_importance(fi, fig_dir)
        fig_predictions_scatter(df_elig, tabular_preds, fig_dir)
        fig_ml_prior(prior_df, fig_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — BDN PRIOR COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    if stage in (0, 2):
        print("\n" + "=" * 68)
        print("  STAGE 2 — BDN Prior Comparison")
        print("  ML prior vs Priors A-D (Sambito et al. 2020)")
        print("  Metrics: convergence speed, F1 isolation, F2 reliability")
        print("=" * 68)

        prior_path = os.path.join(output_dir, 'ml_prior.csv')
        if not os.path.exists(prior_path):
            raise FileNotFoundError(
                f"ml_prior.csv missing at {prior_path} — run --stage 1 first.")
        prior_df_s2 = pd.read_csv(prior_path)

        print("\n  Building five prior distributions...")
        priors = build_v1_priors(df_all, prior_df_s2)

        print("\n  Simulating BDN convergence (500 update steps per prior)...")
        bdn_results = simulate_bdn_convergence(df_elig, priors, n_scenarios=500)

        print("\n-- Stage 2 Summary -----------------------------------------")
        s2_df = (pd.DataFrame(bdn_results).T.reset_index()
                   .rename(columns={'index': 'Prior'}))
        print(s2_df.to_string(index=False))
        s2_path = os.path.join(output_dir, 'stage2_bdn_comparison.csv')
        s2_df.to_csv(s2_path, index=False)
        print(f"\n  Saved -> {s2_path}")

        print("\n  Stage 2 figures...")
        fig_bdn_convergence(bdn_results, fig_dir)
        fig_prior_distributions(priors, fig_dir)

    # Final file listing
    print("\n" + "=" * 68)
    print("  OUTPUT FILES")
    print("=" * 68)
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            print(f"  {f:<42} {os.path.getsize(fp):>8} bytes")
    for f in sorted(os.listdir(fig_dir)):
        fp = os.path.join(fig_dir, f)
        print(f"  figures/{f:<34} {os.path.getsize(fp):>8} bytes")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Hybrid AI Sensor Placement — Two-Stage Pipeline')
    ap.add_argument('--data_dir',   default='./output')
    ap.add_argument('--output_dir', default='./ml_results')
    ap.add_argument('--stage', type=int, default=0, choices=[0, 1, 2],
                    help='0=both (default), 1=ML only, 2=BDN only')
    a = ap.parse_args()
    main(a.data_dir, a.output_dir, a.stage)