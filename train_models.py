"""
train_models.py
===============
ML training pipeline for the Hybrid AI Sensor Placement project.
Mhango, S.B. and Sambito, M. (2026)

Three models are trained, matching the concept note architecture:

  Option A -- XGBoost / LightGBM (gradient boosting, tabular baseline)
  Option B -- MLP with dropout (feedforward neural network)
  Option C -- GCN / GAT (graph neural network, requires torch-geometric)

Each model predicts detection_freq at every node (a regression target in [0,1]).
The output vector is normalised to sum to 1 and used as the ML-derived prior
P(sensor at node i) for the BDN solver, replacing the Monte Carlo Prior D.

Inputs:
  node_features.csv   -- one row per node, produced by dataset_generator.py
  raw_scenarios.csv   -- one row per (scenario, node), same source
  Example8.inp        -- SWMM input file (for graph edges in Option C)

Outputs (written to --output_dir, default ./ml_output/):
  models/xgb_model.json        XGBoost model
  models/lgbm_model.txt        LightGBM model
  models/mlp_model.pt          MLP weights (PyTorch)
  models/gcn_model.pt          GCN weights  (if torch-geometric available)
  models/gat_model.pt          GAT weights  (if torch-geometric available)
  priors/prior_xgb.csv         ML prior from XGBoost
  priors/prior_lgbm.csv        ML prior from LightGBM
  priors/prior_mlp.csv         ML prior from MLP
  priors/prior_gcn.csv         ML prior from GCN
  priors/prior_gat.csv         ML prior from GAT
  evaluation/metrics.csv       CV metrics for all models
  evaluation/feature_importance.csv   XGBoost / LightGBM importances

Usage:
  python train_models.py
  python train_models.py --node_features ./output/node_features.csv \\
                         --raw_scenarios  ./output/raw_scenarios.csv \\
                         --inp            Example8.inp \\
                         --output_dir     ./ml_output
  python train_models.py --skip_gnn       # skip GNN if torch-geometric unavailable

Requirements (core):
  pip install xgboost lightgbm scikit-learn pandas numpy torch

Requirements (GNN, optional):
  pip install torch-geometric
"""

import os
import sys
import argparse
import warnings
import json
import numpy as np
import pandas as pd

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

# ONNX export
try:
    import onnxruntime
    import onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnxruntime onnxmltools skl2onnx")

from config import config
from cache import cached

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = config.get('ml.cv.random_state', 42)
np.random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
# Columns used from node_features.csv as model input features
# (excludes target columns and string identifiers)
STATIC_FEATURES = [
    "topo_depth",
    "n_upstream_nodes",
    "betweenness",
    "downstream_paths",
    "node_type_code",
    "is_high_risk",
    "prior_contam_prob",
    "flow_diversion_fraction",
]

DYNAMIC_FEATURES = [
    "peak_conc_mean",
    "peak_conc_std",
    "time_to_peak_mean",
    "mean_flow_m3s",
    "mean_vel_ms",
    "mean_wastewater_flux",
    "mean_contaminant_flux",
    "contaminant_flux_std",
]

ALL_NODE_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

# Target: detection_freq is the proportion of scenarios in which detected=1
TARGET = "detection_freq"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING AND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def load_node_features(path):
    """
    Load node_features.csv (one row per node, 31 rows for Example 8).
    Fills missing dynamic features with 0 (nodes never detected have NaN
    for time_to_peak_mean).
    """
    df = pd.read_csv(path)
    df = df.fillna(0)

    # Exclude outfalls/storage from training — they are not sensor candidates.
    # Support both: legacy 'node_type' text column AND new 'node_type_code' int column.
    # node_type_code: 0 = J (junction), 1 = JI (interceptor), 3 = outfall/storage
    if "node_type" in df.columns:
        df["is_candidate"] = df["node_type"].isin(["J", "JI", "aux"]).astype(int)
    elif "node_type_code" in df.columns:
        df["is_candidate"] = df["node_type_code"].isin([0, 1]).astype(int)
    else:
        # No type info available — treat all nodes as candidates
        df["is_candidate"] = 1

    return df


def load_raw_scenarios(path):
    """
    Load raw_scenarios.csv (one row per scenario × node).
    Used to compute scenario-level statistics and for row-level cross-checks.
    """
    df = pd.read_csv(path)
    df["t_peak_min"] = df["t_peak_min"].fillna(-1)   # -1 = not detected
    return df


def build_edge_index_and_features(inp_file, node_list):
    """
    Parse Example8.inp to extract:
      - edge_index: (2, num_edges) tensor of [src, dst] node indices
      - edge_attr:  (num_edges, 4) tensor of [length, roughness, diameter, shape_code]

    All link types are included: conduits, pumps, weirs, orifices.
    Shape codes: CIRCULAR=0, TRAPEZOIDAL=1, RECT_OPEN=2, other=3.
    """
    node_idx = {n: i for i, n in enumerate(node_list)}

    # Parse conduit geometry
    conduits = {}
    xsections = {}
    section = None

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
            if section == "CONDUITS" and len(parts) >= 5:
                conduits[parts[0]] = {
                    "from": parts[1], "to": parts[2],
                    "length": float(parts[3]), "roughness": float(parts[4])
                }
            elif section == "XSECTIONS" and len(parts) >= 3:
                shape_map = {"CIRCULAR": 0, "TRAPEZOIDAL": 1, "RECT_OPEN": 2}
                xsections[parts[0]] = {
                    "shape_code": shape_map.get(parts[1], 3),
                    "geom1": float(parts[2]) if len(parts) > 2 else 0.0,
                }
            # For pumps, weirs, orifices — add as edges with default attributes
            elif section in ("PUMPS", "WEIRS", "ORIFICES") and len(parts) >= 3:
                conduits[parts[0]] = {
                    "from": parts[1], "to": parts[2],
                    "length": 10.0, "roughness": 0.016
                }

    src_idx, dst_idx, attrs = [], [], []

    for name, info in conduits.items():
        f_node, t_node = info["from"], info["to"]
        if f_node not in node_idx or t_node not in node_idx:
            continue
        xs = xsections.get(name, {"shape_code": 3, "geom1": 0.5})
        src_idx.append(node_idx[f_node])
        dst_idx.append(node_idx[t_node])
        attrs.append([
            info["length"],
            info["roughness"],
            xs["geom1"],          # diameter or height
            float(xs["shape_code"])
        ])

    return [src_idx, dst_idx], attrs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CROSS-VALIDATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def leave_one_out_cv(model_fn, X, y, node_ids):
    """
    Leave-One-Out cross-validation over nodes.
    For each node, train on the remaining 30 nodes and predict on the held-out one.
    Returns arrays of true and predicted detection frequencies.

    LOO is appropriate here because n=31 nodes is too small for a
    held-out test split to be meaningful, and we want every node to
    appear exactly once in the test set.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    preds = np.zeros(len(y))
    for i in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[i] = False
        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        preds[i] = model.predict(X[[i]])[0]

    mae  = mean_absolute_error(y, preds)
    r2   = r2_score(y, preds)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    rank_corr = np.corrcoef(
        pd.Series(y).rank().values,
        pd.Series(preds).rank().values
    )[0, 1]

    return {
        "mae":       round(mae,  4),
        "rmse":      round(rmse, 4),
        "r2":        round(r2,   4),
        "rank_corr": round(rank_corr, 4),
        "preds":     preds,
    }


def normalise_to_prior(preds, node_ids, candidate_mask):
    """
    Converts raw model predictions to a prior probability vector.
    - Clamps predictions to [0, 1]
    - Sets prior=0 for non-candidate nodes (outfalls, storage)
    - Normalises candidate priors to sum to 1
    Returns a DataFrame with columns [node_id, raw_pred, prior_prob]
    """
    preds = np.clip(preds, 0, 1)
    preds_masked = preds * candidate_mask
    total = preds_masked.sum()
    if total > 0:
        prior = preds_masked / total
    else:
        # Fallback: uniform over candidates
        prior = candidate_mask / candidate_mask.sum()

    return pd.DataFrame({
        "node_id":   node_ids,
        "raw_pred":  preds.round(6),
        "prior_prob": prior.round(6),
    })


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OPTION A: GRADIENT BOOSTING (XGBoost + LightGBM)
# ══════════════════════════════════════════════════════════════════════════════

def train_gradient_boosting(X, y, node_ids, candidate_mask, output_dir):
    """
    Trains XGBoost and LightGBM regressors with LOO cross-validation.
    Both models predict detection_freq at each node.
    """
    try:
        import xgboost as xgb
        import lightgbm as lgbm
    except ImportError:
        print("  [SKIP] xgboost or lightgbm not installed. Run: pip install xgboost lightgbm")
        return {}

    from sklearn.preprocessing import StandardScaler
    import time

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("  Training XGBoost ...")

    xgb_params = config.get('ml.models.xgboost.params', {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': SEED,
        'verbosity': 0,
    })

    def xgb_fn():
        return xgb.XGBRegressor(**xgb_params)

    # MLflow tracking for XGBoost
    if MLFLOW_AVAILABLE and config.get('ml.tracking.enabled', True):
        with mlflow.start_run(run_name="XGBoost_Training"):
            mlflow.log_params(xgb_params)
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("cv_type", "Leave-One-Out")

            xgb_cv = leave_one_out_cv(xgb_fn, X_scaled, y, node_ids)

            # Log metrics
            mlflow.log_metric("mae", xgb_cv['mae'])
            mlflow.log_metric("rmse", xgb_cv['rmse'])
            mlflow.log_metric("r2", xgb_cv['r2'])
            mlflow.log_metric("rank_corr", xgb_cv['rank_corr'])

            print(f"    LOO MAE={xgb_cv['mae']:.4f}  RMSE={xgb_cv['rmse']:.4f}  "
                  f"R²={xgb_cv['r2']:.4f}  RankCorr={xgb_cv['rank_corr']:.4f}")

            # Full-data fit for saving and prior extraction
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_scaled, y)
            model_path = os.path.join(output_dir, "models", "xgb_model.json")
            xgb_model.save_model(model_path)
            print(f"    Saved: {model_path}")

            # Log model
            mlflow.xgboost.log_model(xgb_model, "model")

            # ONNX export
            if ONNX_AVAILABLE:
                try:
                    onnx_model = convert_sklearn(xgb_model, initial_types=[('input', FloatTensorType([None, X.shape[1]]))])
                    onnx_path = os.path.join(output_dir, "models", "xgb_model.onnx")
                    onnxmltools.utils.save_model(onnx_model, onnx_path)
                    mlflow.log_artifact(onnx_path, "onnx_model")
                    print(f"    ONNX exported: {onnx_path}")
                except Exception as e:
                    print(f"    ONNX export failed: {e}")

            prior_xgb = normalise_to_prior(xgb_cv["preds"], node_ids, candidate_mask)
            prior_path = os.path.join(output_dir, "priors", "prior_xgb.csv")
            prior_xgb.to_csv(prior_path, index=False)
            print(f"    Prior saved: {prior_path}")

            # Feature importance
            feat_imp_xgb = pd.DataFrame({
                "feature": ALL_NODE_FEATURES,
                "importance_xgb": xgb_model.feature_importances_,
            }).sort_values("importance_xgb", ascending=False)

            results["xgb"] = {
                "model": "XGBoost",
                **{k: v for k, v in xgb_cv.items() if k != "preds"},
            }
    else:
        # Original logic without MLflow
        start_time = time.time()
        xgb_cv = leave_one_out_cv(xgb_fn, X_scaled, y, node_ids)
        print(f"    LOO MAE={xgb_cv['mae']:.4f}  RMSE={xgb_cv['rmse']:.4f}  "
              f"R²={xgb_cv['r2']:.4f}  RankCorr={xgb_cv['rank_corr']:.4f}")

        # Full-data fit for saving and prior extraction
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_scaled, y)
        train_time_xgb = time.time() - start_time
        model_path = os.path.join(output_dir, "models", "xgb_model.json")
        xgb_model.save_model(model_path)
        print(f"    Saved: {model_path}")

        prior_xgb = normalise_to_prior(xgb_cv["preds"], node_ids, candidate_mask)
        prior_path = os.path.join(output_dir, "priors", "prior_xgb.csv")
        prior_xgb.to_csv(prior_path, index=False)
        print(f"    Prior saved: {prior_path}")

        # Feature importance
        feat_imp_xgb = pd.DataFrame({
            "feature": ALL_NODE_FEATURES,
            "importance_xgb": xgb_model.feature_importances_,
        }).sort_values("importance_xgb", ascending=False)

        results["xgb"] = {
            "model": "XGBoost",
            "train_time_s": round(train_time_xgb, 2),
            **{k: v for k, v in xgb_cv.items() if k != "preds"},
        }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("  Training LightGBM ...")

    lgbm_params = dict(
        n_estimators=200,
        num_leaves=8,          # must be small when n_samples is small (default 31 causes no splits)
        min_child_samples=1,   # allow leaves with a single sample for n=31
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
        force_col_wise=True,
    )

    def lgbm_fn():
        return lgbm.LGBMRegressor(**lgbm_params)

    start_time = time.time()
    lgbm_cv = leave_one_out_cv(lgbm_fn, X_scaled, y, node_ids)
    print(f"    LOO MAE={lgbm_cv['mae']:.4f}  RMSE={lgbm_cv['rmse']:.4f}  "
          f"R²={lgbm_cv['r2']:.4f}  RankCorr={lgbm_cv['rank_corr']:.4f}")

    lgbm_model = lgbm.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_scaled, y)
    train_time_lgbm = time.time() - start_time
    model_path = os.path.join(output_dir, "models", "lgbm_model.txt")
    lgbm_model.booster_.save_model(model_path)
    print(f"    Saved: {model_path}")

    prior_lgbm = normalise_to_prior(lgbm_cv["preds"], node_ids, candidate_mask)
    prior_path = os.path.join(output_dir, "priors", "prior_lgbm.csv")
    prior_lgbm.to_csv(prior_path, index=False)
    print(f"    Prior saved: {prior_path}")

    feat_imp_lgbm = pd.DataFrame({
        "feature": ALL_NODE_FEATURES,
        "importance_lgbm": lgbm_model.feature_importances_,
    })

    # Save combined feature importances
    feat_imp = feat_imp_xgb.merge(feat_imp_lgbm, on="feature")
    imp_path = os.path.join(output_dir, "evaluation", "feature_importance.csv")
    feat_imp.to_csv(imp_path, index=False)
    print(f"    Feature importances saved: {imp_path}")

    results["lgbm"] = {
        "model": "LightGBM",
        "train_time_s": round(train_time_lgbm, 2),
        **{k: v for k, v in lgbm_cv.items() if k != "preds"},
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTION B: MLP WITH DROPOUT
# ══════════════════════════════════════════════════════════════════════════════

def train_mlp(X, y, node_ids, candidate_mask, output_dir):
    """
    Trains a 3-layer MLP with dropout using PyTorch.
    LOO cross-validation is approximated via K-fold (k=5) given the small
    dataset size and the overhead of training a neural net per fold.

    Architecture (concept note Option B):
      Linear(n_feat -> 64) → ReLU → Dropout(0.3)
      Linear(64 -> 32)     → ReLU → Dropout(0.3)
      Linear(32 -> 1)      → Sigmoid (output in [0,1])
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, r2_score
        import time
    except (ImportError, OSError):
        print("  [SKIP] PyTorch not installed or not loadable. Run: pip install torch")
        return {}

    torch.manual_seed(SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    y_arr = y.astype(np.float32)

    class SensorMLP(nn.Module):
        def __init__(self, n_feat):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_one(X_tr, y_tr, n_epochs=300):
        model = SensorMLP(X_tr.shape[1])
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X_tr)
        yt = torch.tensor(y_tr)
        model.train()
        for _ in range(n_epochs):
            opt.zero_grad()
            pred = model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        return model

    print("  Training MLP (5-fold CV) ...")
    kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds_all = np.zeros(len(y_arr))

    start_time = time.time()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
        model = train_one(X_scaled[tr_idx], y_arr[tr_idx])
        model.eval()
        with torch.no_grad():
            preds_all[val_idx] = model(torch.tensor(X_scaled[val_idx])).numpy()

    mae  = mean_absolute_error(y_arr, preds_all)
    r2   = r2_score(y_arr, preds_all)
    rmse = float(np.sqrt(np.mean((y_arr - preds_all) ** 2)))
    rank_corr = float(np.corrcoef(
        pd.Series(y_arr).rank().values,
        pd.Series(preds_all).rank().values
    )[0, 1])

    print(f"    5-Fold MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"R²={r2:.4f}  RankCorr={rank_corr:.4f}")

    # Full-data fit for saving
    full_model = train_one(X_scaled, y_arr, n_epochs=500)
    train_time_mlp = time.time() - start_time
    model_path = os.path.join(output_dir, "models", "mlp_model.pt")
    torch.save({
        "state_dict": full_model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": ALL_NODE_FEATURES,
    }, model_path)
    print(f"    Saved: {model_path}")

    prior_mlp = normalise_to_prior(preds_all, node_ids, candidate_mask)
    prior_path = os.path.join(output_dir, "priors", "prior_mlp.csv")
    prior_mlp.to_csv(prior_path, index=False)
    print(f"    Prior saved: {prior_path}")

    return {
        "mlp": {
            "model": "MLP",
            "train_time_s": round(train_time_mlp, 2),
            "mae":       round(float(mae),       4),
            "rmse":      round(float(rmse),      4),
            "r2":        round(float(r2),        4),
            "rank_corr": round(float(rank_corr), 4),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OPTION C: GRAPH NEURAL NETWORKS (GCN + GAT)
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(X, y, node_ids, candidate_mask, edge_index, edge_attr, output_dir):
    """
    Trains GCN and GAT models using PyTorch Geometric.

    Graph structure:
      Nodes  : 31 network junctions (+ outfalls + storage)
      Edges  : 35 directed links (conduits, pumps, weirs, orifices)
      Node X : ALL_NODE_FEATURES (11 features)
      Edge X : [length, roughness, diameter, shape_code] (4 features)

    Since Example 8 has only 31 nodes, the GNN operates on the full graph
    at once (no mini-batching). Cross-validation uses a random 80/20
    node split repeated 10 times (Monte Carlo CV), following the concept
    note's acknowledgement that Example 8 is too small for standard splits.

    Architecture:
      GCN: 3 × GCNConv(in -> 64 -> 32 -> 1) + ReLU + Sigmoid
      GAT: 3 × GATConv(in -> 32, heads=2) -> Linear(64->1) + Sigmoid
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, r2_score
        import time
    except (ImportError, OSError):
        print("  [SKIP] torch-geometric not installed.")
        print("         Run: pip install torch-geometric")
        print("         (GCN/GAT models skipped; XGBoost and MLP priors are sufficient)")
        return {}

    torch.manual_seed(SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Normalise edge attributes
    ea = np.array(edge_attr, dtype=np.float32)
    ea_mean = ea.mean(axis=0)
    ea_std  = ea.std(axis=0) + 1e-8
    ea_norm = (ea - ea_mean) / ea_std

    import torch
    from torch_geometric.data import Data

    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t  = torch.tensor(ea_norm,    dtype=torch.float)
    X_t          = torch.tensor(X_scaled,   dtype=torch.float)
    y_t          = torch.tensor(y.astype(np.float32), dtype=torch.float)

    data = Data(x=X_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)

    n_feat = X_t.shape[1]

    # ── GCN ───────────────────────────────────────────────────────────────────
    class GCNModel(nn.Module):
        def __init__(self):
            super().__init__()
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(n_feat, 64)
            self.conv2 = GCNConv(64, 32)
            self.conv3 = GCNConv(32, 16)
            self.head  = nn.Linear(16, 1)

        def forward(self, data):
            x, ei = data.x, data.edge_index
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv2(x, ei))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv3(x, ei))
            return torch.sigmoid(self.head(x)).squeeze(-1)

    # ── GAT ───────────────────────────────────────────────────────────────────
    class GATModel(nn.Module):
        def __init__(self):
            super().__init__()
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(n_feat, 32, heads=2, dropout=0.3)
            self.conv2 = GATConv(64, 16, heads=2, dropout=0.3)
            self.conv3 = GATConv(32, 16, heads=1, dropout=0.3)
            self.head  = nn.Linear(16, 1)

        def forward(self, data):
            x, ei = data.x, data.edge_index
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv2(x, ei))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv3(x, ei))
            return torch.sigmoid(self.head(x)).squeeze(-1)

    def train_gnn_model(ModelClass, n_epochs=500):
        model  = ModelClass()
        opt    = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(n_epochs):
            opt.zero_grad()
            out  = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            opt.step()
        return model

    def mc_cv_gnn(ModelClass, n_repeats=10, test_frac=0.2):
        """Monte Carlo CV: random node splits, repeated n_repeats times."""
        n = len(y)
        n_test = max(1, int(n * test_frac))
        all_true, all_pred = [], []
        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + rep)
            test_idx  = rng.choice(n, size=n_test, replace=False)
            train_idx = np.setdiff1d(np.arange(n), test_idx)

            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[train_idx] = True

            model = ModelClass()
            opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            for _ in range(300):
                model.train()
                opt.zero_grad()
                out  = model(data)
                loss = loss_fn(out[train_mask], data.y[train_mask])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                preds = model(data).numpy()

            all_true.extend(y[test_idx].tolist())
            all_pred.extend(preds[test_idx].tolist())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        mae  = mean_absolute_error(all_true, all_pred)
        r2   = r2_score(all_true, all_pred)
        rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
        rank_corr = float(np.corrcoef(
            pd.Series(all_true).rank().values,
            pd.Series(all_pred).rank().values
        )[0, 1])
        return {"mae": round(mae, 4), "rmse": round(rmse, 4),
                "r2": round(r2, 4), "rank_corr": round(rank_corr, 4)}

    results = {}

    for name, ModelClass in [("GCN", GCNModel), ("GAT", GATModel)]:
        print(f"  Training {name} (Monte Carlo CV, 10 repeats) ...")
        start_time = time.time()
        cv_metrics = mc_cv_gnn(ModelClass)
        print(f"    MC-CV MAE={cv_metrics['mae']:.4f}  RMSE={cv_metrics['rmse']:.4f}  "
              f"R²={cv_metrics['r2']:.4f}  RankCorr={cv_metrics['rank_corr']:.4f}")

        full_model = train_gnn_model(ModelClass, n_epochs=500)
        train_time_gnn = time.time() - start_time
        model_path = os.path.join(output_dir, "models", f"{name.lower()}_model.pt")
        torch.save({
            "state_dict": full_model.state_dict(),
            "model_class": name,
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_names": ALL_NODE_FEATURES,
        }, model_path)
        print(f"    Saved: {model_path}")

        full_model.eval()
        with torch.no_grad():
            full_preds = full_model(data).numpy()

        prior_df = normalise_to_prior(full_preds, node_ids, candidate_mask)
        prior_path = os.path.join(output_dir, "priors", f"prior_{name.lower()}.csv")
        prior_df.to_csv(prior_path, index=False)
        print(f"    Prior saved: {prior_path}")

        results[name.lower()] = {"model": name, "train_time_s": round(train_time_gnn, 2), **cv_metrics}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION AND COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_v1_priors(node_features_df, results, output_dir):
    """
    Adds v1.0 baseline priors (A, B, C, D equivalents) to the metrics table
    for direct comparison, using node_features.csv data.

    Prior A (uniform):      equal weight to all candidate nodes
    Prior B (topo-based):   weight proportional to 1 / (topo_depth + 1)
    Prior C (flow-based):   weight proportional to mean_flow_m3s
    Prior D (approx):       weight proportional to prior_contam_prob × mean_flow_m3s
                            (true Prior D requires a 50-run Monte Carlo;
                             this is an approximation from available features)
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    df = node_features_df.copy()
    target = df[TARGET].values
    cand   = df["is_candidate"].values.astype(float)
    node_ids = df["node_id"].values

    def eval_prior(weights):
        weights = np.array(weights, dtype=float) * cand
        total = weights.sum()
        prior = weights / total if total > 0 else cand / cand.sum()
        mae   = mean_absolute_error(target, prior)
        rmse  = float(np.sqrt(np.mean((target - prior) ** 2)))
        r2    = r2_score(target, prior)
        rc    = float(np.corrcoef(
            pd.Series(target).rank().values,
            pd.Series(prior).rank().values
        )[0, 1])
        return {"mae": round(mae,4), "rmse": round(rmse,4),
                "r2": round(r2,4), "rank_corr": round(rc,4)}

    v1_results = {
        "Prior_A_Uniform":  eval_prior(cand),
        "Prior_B_Topo":     eval_prior(1.0 / (df["topo_depth"].values + 1)),
        "Prior_C_Flow":     eval_prior(df["mean_flow_m3s"].values + 1e-9),
        "Prior_D_approx":   eval_prior(
            df["prior_contam_prob"].values * (df["mean_flow_m3s"].values + 1e-9)
        ),
    }

    all_results = {}
    for k, v in v1_results.items():
        all_results[k] = {"model": k, **v}
    all_results.update(results)

    metrics_df = pd.DataFrame(all_results).T.reset_index(drop=True)
    metrics_path = os.path.join(output_dir, "evaluation", "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df


def print_summary(metrics_df):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'RankCorr':>10}")
    print("-" * 70)
    for _, row in metrics_df.iterrows():
        print(f"{str(row['model']):<22} "
              f"{float(row['mae']):>8.4f} "
              f"{float(row['rmse']):>8.4f} "
              f"{float(row['r2']):>8.4f} "
              f"{float(row['rank_corr']):>10.4f}")
    print("=" * 70)
    print("\nNote: Lower MAE/RMSE and higher R²/RankCorr = better prior.")
    print("A prior that outperforms Prior D (approx) on RankCorr is expected")
    print("to require fewer BDN update simulations than the v1.0 baseline.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(node_features_path, raw_scenarios_path, inp_file, output_dir, skip_gnn, exclude_features=None):

    # ── Setup output directories ───────────────────────────────────────────────
    for sub in ["models", "priors", "evaluation"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    print("=" * 60)
    print("Hybrid AI Sensor Placement -- ML Training Pipeline")
    print("Mhango & Sambito (2026)")
    print("=" * 60)
    
    if exclude_features:
        print(f"\n[ABLATION STUDY] Excluding features: {exclude_features}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    node_df = load_node_features(node_features_path)
    raw_df  = load_raw_scenarios(raw_scenarios_path)
    print(f"      Nodes: {len(node_df)}  |  Raw scenario rows: {len(raw_df):,}")
    print(f"      Detection freq range: "
          f"{node_df[TARGET].min():.4f} – {node_df[TARGET].max():.4f}")
    print(f"      Candidate sensor nodes: {node_df['is_candidate'].sum()}")

    # ── Prepare feature matrix ─────────────────────────────────────────────────
    print("\n[2/6] Preparing feature matrix ...")
    
    active_features = [f for f in ALL_NODE_FEATURES if f not in (exclude_features or [])]
    if len(active_features) == 0:
        raise ValueError("All features were excluded. Cannot train models.")
        
    missing = [c for c in active_features if c not in node_df.columns]
    if missing:
        print(f"      WARNING: missing features {missing} — filling with 0")
        for c in missing:
            node_df[c] = 0.0

    X              = node_df[active_features].values.astype(np.float64)
    y              = node_df[TARGET].values.astype(np.float64)
    node_ids       = node_df["node_id"].values
    candidate_mask = node_df["is_candidate"].values.astype(float)

    print(f"      Feature matrix: {X.shape[0]} nodes × {X.shape[1]} features")
    print(f"      Features used: {active_features}")

    # ── Graph structure (for GNN) ──────────────────────────────────────────────
    if not skip_gnn:
        print("\n[3/6] Building graph structure ...")
        edge_index, edge_attr = build_edge_index_and_features(inp_file, node_ids.tolist())
        print(f"      Edges: {len(edge_index[0])}  |  "
              f"Edge features: length, roughness, diameter, shape_code")
    else:
        edge_index, edge_attr = None, None
        print("\n[3/6] Graph structure skipped (--skip_gnn)")

    # ── Train models ───────────────────────────────────────────────────────────
    all_results = {}

    print("\n[4/6] Option A -- Gradient Boosting (XGBoost + LightGBM) ...")
    if config.get('ml.models.xgboost.enabled', True) or config.get('ml.models.lightgbm.enabled', True):
        all_results.update(train_gradient_boosting(X, y, node_ids, candidate_mask, output_dir))
    else:
        print("      Option A skipped (disabled in config)")

    print("\n[5/6] Option B -- MLP with Dropout ...")
    if config.get('ml.models.mlp.enabled', True):
        all_results.update(train_mlp(X, y, node_ids, candidate_mask, output_dir))
    else:
        print("      Option B skipped (disabled in config)")

    if not skip_gnn and edge_index is not None:
        print("\n[6/6] Option C -- GCN + GAT ...")
        if config.get('ml.models.gnn.enabled', True):
            all_results.update(train_gnn(X, y, node_ids, candidate_mask, edge_index, edge_attr, output_dir))
        else:
            print("      Option C skipped (disabled in config)")
    else:
        print("\n[6/6] Option C -- GNN skipped")

    # ── Evaluate and compare ───────────────────────────────────────────────────
    print("\nComparing against v1.0 baseline priors ...")
    metrics_df = compare_with_v1_priors(node_df, all_results, output_dir)
    print_summary(metrics_df)

    print(f"\nAll outputs written to: {os.path.abspath(output_dir)}/")
    print("  models/       -- trained model files")
    print("  priors/       -- normalised prior probability CSVs")
    print("  evaluation/   -- metrics.csv and feature_importance.csv")
    print("\nNext step: pass priors/prior_*.csv to the BDN solver")
    print("           in place of the Monte Carlo Prior D.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for hybrid AI sensor placement (Mhango & Sambito 2026)"
    )
    parser.add_argument("--node_features", default="./output/node_features.csv",
                        help="Path to node_features.csv from dataset_generator.py")
    parser.add_argument("--raw_scenarios",  default="./output/raw_scenarios.csv",
                        help="Path to raw_scenarios.csv from dataset_generator.py")
    parser.add_argument("--model_path",     default="./dataset/Examples/Example8.inp",
                        help="Path to SWMM .inp file (for graph edge extraction)")
    parser.add_argument("--output_dir",     default="./ml_output",
                        help="Directory for all model and prior output files")
    parser.add_argument("--skip_gnn",       action="store_true",
                        help="Skip GNN training (use if torch-geometric is not installed)")
    parser.add_argument("--exclude_features", nargs='+', default=[],
                        help="List of features to exclude for ablation studies (e.g. mean_vel_ms prior_contam_prob)")
    args = parser.parse_args()

    main(
        node_features_path = args.node_features,
        raw_scenarios_path = args.raw_scenarios,
        inp_file           = args.model_path,
        output_dir         = args.output_dir,
        skip_gnn           = args.skip_gnn,
        exclude_features   = args.exclude_features,
    )