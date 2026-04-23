# Hybrid AI Sensor Placement Guide

This project helps you find the best places to put water quality sensors in a sewer network. It uses an "Intelligent Assistant" (AI) to learn how pollution moves through pipes and suggests where to place sensors for maximum safety.

---

## ⚡ 1. The "Master Control" (Important!)

Before you do anything, you need to decide which network you are analyzing. All settings are kept in one file: **`config/default.yaml`**.

1.  **Open `config/default.yaml`** in your text editor.
2.  **Check the `model_path`**: This is the file the AI will look at.
    - Set it to `./dataset/Examples/Example9_event.inp` for a real analysis.
    - Set it to `./dataset/Examples/Example8.inp` for a quick test.
3.  **Check `n_scenarios`**: This is how many tests the AI will run.
    - 5000 is best for real accuracy.
    - 100 is best for a fast trial.

---

## 🛠️ 2. Setup (Do this once)

Run these commands to prepare your workspace:

```bash
# 1. Prepare the workspace
python -m venv venv

# 2. Activate the environment
# Windows: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser (Run once)
#          .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install the tools
pip install -r requirements.txt
```

---

### 🚀 3. Running the Analysis

You can run the entire process (generating data, teaching the AI, and finding sensors) using a single "One-Click" command. This is the recommended way to run the analysis because it automatically handles the technical steps like **Data Splitting** to ensure your results are accurate.

#### The One-Click Command:
```bash
python run_all.py
```

---

## 🔍 4. What is happening under the hood?

If you prefer to run things step-by-step, or want to understand what the "One-Click" script is doing, here are the stages:

1.  **`dataset_generator.py`**: Runs thousands of tests on your sewer network.
2.  **`split_data.py`**: Separates your tests into a "Learning Set" and a "Testing Set" (important for accuracy!).
3.  **`feature_engineering.py`**: Prepares the specific network details for the AI.
4.  **`train_models.py`**: Teaches the different AI "Brains" (XGBoost, MLP, GNN).
5.  **`bdn_solver.py`**: Finds the best sensor locations and compares the AI models.

---




## 3. Which "AI Brain" Should I Use?

When you run the analysis, the system uses different "brains" to think. You don't need to change anything, but here is what they are:

*   **XGBoost (The Speedster):** Very fast and good for general networks.
*   **MLP (The Thinker):** Good at finding complex, hidden patterns.
*   **GNN (The Architect):** The most advanced; it understands the physical shape of the pipes.

---

## 4. Customizing Your Analysis (Optional)

If you want to change how the system works (for example, to only use one specific AI model), you can edit the **`config/default.yaml`** file.

### How to Turn Models On or Off
Open `config/default.yaml` and look for the `models` section. Change `enabled: true` to `false` for any model you want to skip:

```yaml
# Inside config/default.yaml
ml:
  models:
    xgboost:
      enabled: true   # Change to false to skip XGBoost
    gnn:
      enabled: true   # Change to false to skip GNN
```

### Other Settings You Can Change
In the same file, you can also change:
*   **`n_scenarios`**: How many pollution tests to run (default is 100).
*   **`high_risk_nodes`**: List the names of nodes you think are more likely to be contaminated.
*   **`exclude_sources`**: List the names of outfalls or tanks where pollution can't start.

---

## Table of Contents (For Technical Users)
1. [Purpose](#1-purpose)
2. [System Requirements](#2-system-requirements)
3. [File Inventory](#3-file-inventory)
4. [Dataset Generator Pipeline](#5-dataset-generator-pipeline)
5. [Configuration Reference](#7-configuration-reference)

---

## 1. Purpose

This project provides the data generation infrastructure for a hybrid AI model that recommends optimal water-quality sensor placements in combined sewer networks. The core idea is to simulate a large number of contamination injection scenarios in SWMM, record which network nodes detect contamination above a threshold, and use those detection patterns — combined with static network topology features — to train an ML model.

The scientific basis follows Sambito et al. (2020), who showed that sensor placement in urban drainage can be optimised using detection probability matrices derived from hydraulic simulations.

---

## 2. System Requirements

| Requirement     | Version / Notes                                               |
| --------------- | ------------------------------------------------------------- |
| Python          | 3.8 or later                                                  |
| pyswmm          | latest                                                        |
| swmm-toolkit    | latest (provides low-level C API bindings)                    |
| networkx        | 2.x or later                                                  |
| pandas          | 1.x or later                                                  |
| numpy           | 1.x or later                                                  |
| xgboost         | latest                                                        |
| lightgbm        | latest                                                        |
| scikit-learn    | latest                                                        |
| torch           | latest                                                        |
| torch-geometric | optional (for GCN/GAT models)                                 |
| EPA SWMM        | 5.2 (for GUI use only — not required for the Python pipeline) |

Install the core Python dependencies with:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pyswmm swmm-toolkit networkx pandas numpy xgboost lightgbm scikit-learn torch
```

Install optional graph neural network dependencies with:

```bash
pip install torch-geometric
```

Install optional ML operations dependencies with:

```bash
pip install mlflow onnxruntime onnxmltools skl2onnx joblib pyyaml
```

---

## 3. File Inventory

### `Example8.inp`

The primary SWMM input file. Defines the combined sewer network geometry, subcatchment properties, conduit cross-sections, pump curve, dry weather flows, and rainfall timeseries. This version includes a `[POLLUTANTS]` section declaring `CONTAM` as a conservative tracer (zero decay), which is required for water-quality simulations.

### `Example8_test.inp`

An alternate version of the input file used during development and debugging. The `[POLLUTANTS]` section appears after `[INFLOWS]` rather than before it in this file — this ordering is also accepted by SWMM 5.2.

### `dataset_generator.py`

The main data generation script. See [Section 5](#5-dataset-generator-pipeline) for a full description of its pipeline.

### `feature_engineering.py`

Enriches the base node features with regulator/diversion metrics and Monte Carlo prior proxies, producing `node_features_full.csv`.

### `train_models.py`

Trains ML models (XGBoost, LightGBM, MLP, and optionally GCN/GAT) on node-level features and detection targets. Produces ML-derived prior files in `ml_output/priors/`.

### `bdn_solver.py`

Runs the Bayesian Decision Network comparison over baseline and ML priors using `raw_scenarios.csv` and `node_features_full.csv`. Writes results to `bdn_output/results/comparison_table.csv`.

### `dump/train_eval_pipeline.py`

Higher-level pipeline script that orchestrates model comparison and BDN evaluation for the repo.

### `model_registry.py`

Utilities for MLflow model registry integration, enabling model versioning, staging, and deployment management.

### `cache.py`

Caching utilities for expensive computations like network parsing and feature engineering.

### `config/`

Configuration files for different environments:

- `default.yaml` - Base configuration
- `development.yaml` - Development overrides
- `production.yaml` - Production settings

---

## Advanced Features

### Performance Optimization

#### Enable Parallel Processing

For faster dataset generation on multi-core systems:

```bash
# Edit config/default.yaml and set:
dataset:
    parallel:
        enabled: true
```

#### Enable Caching

Cache expensive computations to speed up repeated runs:

```bash
# Edit config/default.yaml and set:
cache:
    enabled: true
```

### Environment Configuration

#### Development vs Production

Use different settings for different environments:

**Development (fast iteration):**

```bash
export SWMM_ENV=development
python dataset_generator.py --n_scenarios 30
```

**Production (full scale):**

```bash
export SWMM_ENV=production
python dataset_generator.py --n_scenarios 5000
```

### ML Experiment Tracking

#### View Training Results

```bash
# Install MLflow
pip install mlflow

# Start tracking UI
mlflow ui

# Open http://localhost:5000
```

#### Enable Tracking

```yaml
# config/default.yaml
ml:
    tracking:
        enabled: true
```

### Model Deployment

#### ONNX Export

Models are automatically exported for deployment:

```bash
ls ml_output/models/
# xgb_model.onnx  lgbm_model.onnx
```

#### Model Registry

```python
from model_registry import model_registry
model_registry.register_model(model, "my_model", "xgboost")
```

---

A sample output from a 30-scenario run demonstrating the column schema. Not the full training dataset.

### `output_sample/node_features.csv`

A sample of the aggregated node-level feature table from the same 30-scenario run.

### Additional pipeline stages

This repo also supports downstream ML and sensor-placement evaluation beyond dataset generation:

- `feature_engineering.py` produces `node_features_full.csv` with additional regulator and Monte Carlo prior features.
- `train_models.py` trains ML models and writes ML priors to `ml_output/priors/`.
- `bdn_solver.py` compares classical and ML priors and writes `bdn_output/results/comparison_table.csv`.
- `dump/train_eval_pipeline.py` provides a higher-level workflow for training and evaluation.

---

## 4. SWMM Network — Example8.inp

### 4.1 Network Topology

The network represents a **29-acre urban combined sewer catchment** drained by two parallel pipe systems:

- **Green pipes** — combined sewers carrying both stormwater and wastewater, routing to flow regulators
- **Brown pipes** — interceptor sewer collecting dry weather flow and conveying it to the WWTP

Flow regulation is achieved through four transverse weirs (W1–W4) and one bottom orifice (Or1), which together divert dry weather flow and small storm flows to the interceptor while allowing larger storms to overflow to the stream (O1).

### 4.2 Node Inventory

| Node ID             | Type                    | Description                                                     |
| ------------------- | ----------------------- | --------------------------------------------------------------- |
| J1, J2, J2a, J3–J13 | Combined sewer junction | Receive subcatchment runoff and/or dry weather flow             |
| Aux3                | Flow splitting node     | Splits combined sewer flow between stream and interceptor paths |
| JI2–JI13, JI18      | Interceptor junction    | Carry intercepted dry weather flow toward the WWTP              |
| Well                | Storage node            | Pump wet well; receives interceptor flow before pumping         |
| O1                  | Outfall                 | Stream discharge (overflow)                                     |
| O2                  | Outfall                 | WWTP discharge (pumped force main)                              |

### 4.3 Conduit Types

| Type                 | IDs            | Description                                         |
| -------------------- | -------------- | --------------------------------------------------- |
| Stream conduits      | C3–C11, C_Aux3 | Open trapezoidal channels; carry overflow to stream |
| Combined sewer pipes | P1–P6          | Circular pipes; carry combined flow                 |
| Interceptor pipes    | I1–I9          | Circular pipes; carry intercepted flow to wet well  |
| Force mains          | I10–I13        | Circular pipes downstream of Pump1                  |

### 4.4 Simulation Settings

| Parameter           | Value                        |
| ------------------- | ---------------------------- |
| Flow routing        | DYNWAVE (fully dynamic wave) |
| Routing time step   | 15 seconds                   |
| Simulation duration | 12 hours                     |
| Rainfall event      | 0.23 inches, 1-hour duration |
| Flow units          | CFS                          |
| Infiltration method | Horton                       |
| Allow ponding       | No                           |

### 4.5 Dry Weather Flows

Baseline wastewater flows are applied at five nodes as `DRY` type inflows:

| Node | Baseline flow (cfs) |
| ---- | ------------------- |
| J1   | 0.008               |
| J2a  | 0.010               |
| Aux3 | 0.004               |
| J13  | 0.0123              |
| J12  | 0.0125              |

### 4.6 Pollutant Definition

`CONTAM` is declared as a conservative tracer with zero decay, zero background concentrations, and no snow-melt quality component:

```
[POLLUTANTS]
;;Name  Units  Crain  Cgw   Crdii  Kdecay  SFflag
CONTAM  MG/L   0.0    0.0   0.0    0.0     NO
```

> **Important:** Do not add optional trailing columns (`CoPoll`, `CoFrac`, `Cdwf`, `Cinit`) after `SFflag=NO`. SWMM 5.2 misreads bare zeros in those positions as object references and raises ERROR 209 ("undefined object 0.0").

### 4.7 High-Risk Nodes

Three nodes have elevated prior contamination probability based on their network position and the findings of Sambito et al. (2020):

| Node | Reason                                                                  |
| ---- | ----------------------------------------------------------------------- |
| J4   | Upstream combined sewer junction with direct surface access             |
| J10  | Downstream combined sewer junction receiving multiple upstream branches |
| JI18 | Interceptor junction at the convergence of flow regulator outputs       |

These nodes are assigned `prior_contam_prob` = 2 × baseline in `node_features.csv` and are sampled at twice the rate during scenario generation.

---

## 5. Dataset Generator Pipeline

`dataset_generator.py` runs in five sequential steps.

### Step 1 — Parse Network (`parse_network`)

Reads `Example8.inp` line by line, extracting:

- All node IDs from `[JUNCTIONS]`, `[OUTFALLS]`, and `[STORAGE]` sections
- All link connectivity pairs from `[CONDUITS]`, `[PUMPS]`, `[WEIRS]`, and `[ORIFICES]`
- Node type classification (J, JI, aux, outfall, storage)

Two graph representations are built:

- `G_dir` — directed graph following flow direction (used for topology depth and betweenness)
- `G_und` — undirected graph (used for pipe-segment distance matrix)

### Step 2 — Topology Distances (`compute_topo_depth`, `compute_dist_matrix`)

**`topo_depth`:** For each node, the shortest directed path length to the nearest outfall (O1 or O2). Computed with `nx.shortest_path_length` on `G_dir`. Nodes with no downstream path to an outfall receive depth = 999.

**`dist_matrix`:** Full all-pairs shortest path lengths on the undirected graph `G_und`. Used to populate `dist_src` in `raw_scenarios.csv` — the number of pipe segments separating the injection node from each observation node.

### Step 3 — Static Feature Table (`build_topology_features`)

Computes the following features for every node, saved to `node_features.csv`:

| Column              | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| `topo_depth`        | Shortest directed path length to nearest outfall                                |
| `n_upstream_nodes`  | Number of network ancestors (nodes that can reach this node via directed edges) |
| `betweenness`       | Normalised betweenness centrality on directed graph                             |
| `downstream_paths`  | Number of distinct directed simple paths to any outfall (cutoff = 20 hops)      |
| `node_type`         | String label: J, JI, aux, outfall, storage, other                               |
| `node_type_code`    | Integer encoding: J=0, JI=1, aux=2, outfall=3, storage=4, other=5               |
| `is_high_risk`      | 1 if node is J4, J10, or JI18, else 0                                           |
| `prior_contam_prob` | Normalised prior probability of being a contamination source                    |

### Step 4 — Scenario Generation (`build_scenario_inp`)

For each scenario, the following parameters are sampled:

| Parameter      | Distribution      | Range                                            |
| -------------- | ----------------- | ------------------------------------------------ |
| `src_node`     | Weighted discrete | All 28 candidate nodes; J4/J10/JI18 at 2× weight |
| `mass_kg`      | Uniform           | 0.01 – 0.50 kg                                   |
| `duration_hrs` | Uniform           | 0.25 – 3.0 hours                                 |
| `start_hrs`    | Uniform           | 0.0 – 6.0 hours from simulation start            |

The injected concentration (mg/L) is derived from mass and carrier flow:

```
conc (mg/L) = mass_kg × 10⁶ / (CARRIER_FLOW × duration_hrs × 3600 × 28.317)
```

where `CARRIER_FLOW = 0.01 cfs` is the small background flow used to physically carry the pollutant mass through the network.

A temporary `.inp` file is built from `Example8.inp` for each scenario by:

1. Injecting a `[POLLUTANTS]` section if not already present (guard for base file compatibility)
2. Appending two `TIMESERIES` entries — one for carrier flow, one for pollutant concentration — as step functions spanning the injection window
3. Appending two `INFLOWS` entries at `src_node` — a `DIRECT` flow inflow and a `CONCEN` quality inflow

The timeseries are shaped as a flat-top pulse:

```
00:00   → 0.0
pre     → 0.0        (1 minute before injection start)
start   → value      (injection begins)
end     → value      (injection ends)
post    → 0.0        (5 minutes after injection end)
12:00   → 0.0
```

### Step 5 — SWMM Execution (`run_scenario`)

Each scenario `.inp` is executed using the low-level `swmm.toolkit` C API:

```python
swmm_open(tmp_inp, rpt, out)
swmm_start(True)          # True = save results to binary output file
while True:
    t = swmm_step()       # advance one routing step (15 seconds)
    if t == 0: break
    for i, nid in enumerate(ids_order):
        c = slv.node_get_pollutant(i, 0)[0]   # CONTAM concentration (mg/L)
        q = slv.node_get_result(i, 0)          # total inflow (cfs)
swmm_end()
swmm_close()
```

Node ordering uses `slv.project_get_id(slv.swmm_NODE, i)` to retrieve SWMM's internal ordering, ensuring the index `i` passed to `node_get_pollutant` matches the correct node.

Per-node results recorded:

| Column          | Description                                                                        |
| --------------- | ---------------------------------------------------------------------------------- |
| `peak_conc`     | Maximum concentration observed at the node across all timesteps (mg/L)             |
| `t_peak_min`    | Time from simulation start to peak concentration (minutes); `None` if no detection |
| `mean_flow_m3s` | Mean total inflow at the node averaged across all timesteps (m³/s)                 |
| `detected`      | 1 if `peak_conc ≥ 5 mg/L`, else 0                                                  |

If `swmm_open` raises an exception (e.g. input file error), all results for that scenario are set to zeros and the scenario is counted as failed.

---

## 6. Output Files

### 6.1 `raw_scenarios.csv`

One row per (scenario × node). For 100 scenarios and 31 nodes this produces 3,100 rows.

| Column          | Type  | Description                                               |
| --------------- | ----- | --------------------------------------------------------- |
| `scen_id`       | int   | Scenario number (1-indexed)                               |
| `src_node`      | str   | Injection node ID                                         |
| `mass_kg`       | float | Injected pollutant mass (kg)                              |
| `duration_hrs`  | float | Injection duration (hours)                                |
| `start_hrs`     | float | Injection start time from simulation start (hours)        |
| `conc_injected` | float | Computed injection concentration (mg/L)                   |
| `node_id`       | str   | Observation node ID                                       |
| `dist_src`      | int   | Pipe-segment distance from `src_node` to `node_id`        |
| `topo_depth`    | int   | Shortest directed path to nearest outfall                 |
| `peak_conc`     | float | Peak CONTAM concentration at `node_id` (mg/L)             |
| `t_peak_min`    | float | Time to peak concentration (minutes); blank if undetected |
| `mean_flow_m3s` | float | Mean total inflow at `node_id` (m³/s)                     |
| `detected`      | int   | 1 if `peak_conc ≥ 5 mg/L`, else 0                         |

### 6.2 `node_features.csv`

One row per node (31 rows). Static topology features joined with dynamic features aggregated across all scenarios.

| Column                 | Type  | Source    | Description                                            |
| ---------------------- | ----- | --------- | ------------------------------------------------------ |
| `node_id`              | str   | network   | Node identifier                                        |
| `topo_depth`           | int   | topology  | Shortest directed path to nearest outfall              |
| `n_upstream_nodes`     | int   | topology  | Number of upstream ancestor nodes                      |
| `betweenness`          | float | topology  | Normalised betweenness centrality                      |
| `downstream_paths`     | int   | topology  | Count of distinct paths to any outfall                 |
| `node_type`            | str   | network   | Node type label                                        |
| `node_type_code`       | int   | network   | Integer-encoded node type                              |
| `is_high_risk`         | int   | prior     | 1 if J4, J10, or JI18                                  |
| `prior_contam_prob`    | float | prior     | Normalised sampling probability                        |
| `detection_freq`       | float | scenarios | Fraction of scenarios where `detected = 1`             |
| `peak_conc_mean`       | float | scenarios | Mean peak concentration across all scenarios (mg/L)    |
| `peak_conc_std`        | float | scenarios | Std dev of peak concentration (mg/L)                   |
| `time_to_peak_mean`    | float | scenarios | Mean time-to-peak across detected scenarios (min)      |
| `mean_flow_m3s`        | float | scenarios | Mean total inflow averaged across all scenarios (m³/s) |
| `n_scenarios_detected` | int   | scenarios | Count of scenarios where `detected = 1`                |

---

## 7. Configuration Reference

All tunable constants are defined at the top of `dataset_generator.py`:

| Constant          | Default           | Description                                  |
| ----------------- | ----------------- | -------------------------------------------- |
| `THRESHOLD`       | `5.0`             | Detection threshold (mg/L)                   |
| `CARRIER_FLOW`    | `0.01`            | Carrier flow added alongside pollutant (cfs) |
| `CFS_TO_M3S`      | `0.028317`        | Unit conversion factor                       |
| `MASS_MIN`        | `0.01`            | Minimum injection mass (kg)                  |
| `MASS_MAX`        | `0.50`            | Maximum injection mass (kg)                  |
| `DURATION_MIN`    | `0.25`            | Minimum injection duration (hours)           |
| `DURATION_MAX`    | `3.0`             | Maximum injection duration (hours)           |
| `START_MIN`       | `0.0`             | Earliest injection start time (hours)        |
| `START_MAX`       | `6.0`             | Latest injection start time (hours)          |
| `HIGH_RISK_NODES` | `{J4, J10, JI18}` | Nodes sampled at 2× rate                     |
| `EXCLUDE_SOURCE`  | `{O1, O2, Well}`  | Nodes excluded as injection candidates       |

Command-line arguments:

| Argument        | Default                           | Description                     |
| --------------- | --------------------------------- | ------------------------------- |
| `--model_path`  | `./dataset/Examples/Example8.inp` | Path to SWMM input file         |
| `--n_scenarios` | `100`                             | Number of scenarios to simulate |
| `--output_dir`  | `./output`                        | Directory for output CSV files  |
| `--seed`        | `42`                              | Random seed for reproducibility |

### `feature_engineering.py` Arguments

| Argument          | Default                           | Description                    |
| ----------------- | --------------------------------- | ------------------------------ |
| `--model_path`    | `./dataset/Examples/Example8.inp` | Path to SWMM input file        |
| `--node_features` | `./output/node_features.csv`      | Path to base node features     |
| `--raw_scenarios` | `./output/raw_scenarios.csv`      | Path to raw scenarios          |
| `--output_dir`    | `./output`                        | Directory for output CSV files |

### `train_models.py` Arguments

| Argument          | Default                           | Description                |
| ----------------- | --------------------------------- | -------------------------- |
| `--model_path`    | `./dataset/Examples/Example8.inp` | Path to SWMM input file    |
| `--node_features` | `./output/node_features_full.csv` | Path to full node features |
| `--raw_scenarios` | `./output/raw_scenarios.csv`      | Path to raw scenarios      |
| `--output_dir`    | `./ml_output`                     | Directory for ML outputs   |
| `--skip_gnn`      | `False`                           | Skip GNN training if True  |

---

## 8. Known Issues and Fixes

### ERROR 209 — "undefined object CONTAM"

**Cause:** `Example8.inp` was originally generated as a hydraulics-only model with no `[POLLUTANTS]` section. SWMM raises ERROR 209 when `CONTAM` appears in `[INFLOWS]` without first being declared as a pollutant.

**Fix:** A `[POLLUTANTS]` section is now present in `Example8.inp`. `dataset_generator.py` also contains a guard in `build_scenario_inp` that injects the section into any base file that lacks it:

```python
if '[POLLUTANTS]' not in content:
    pollutants_block = (
        '[POLLUTANTS]\n'
        ';;Name  Units  Crain  Cgw   Crdii  Kdecay  SFflag\n'
        'CONTAM  MG/L   0.0    0.0   0.0    0.0     NO\n\n'
    )
    content = content.replace('[INFLOWS]', pollutants_block + '[INFLOWS]')
```

### ERROR 209 — "undefined object 0.0"

**Cause:** Including optional trailing columns (`CoPoll`, `CoFrac`, `Cdwf`, `Cinit`) after `SFflag=NO` in the `[POLLUTANTS]` line. SWMM 5.2 parses these as object references when `SFflag=NO`, treating the bare zero values as undefined object names.

**Fix:** The `[POLLUTANTS]` declaration stops at `SFflag`. No trailing columns.

### ERROR 200 / Empty `.rpt` File

**Cause:** The `QUALITY ALL` keyword was erroneously added to the `[REPORT]` section in an earlier fix attempt. `QUALITY` is not a valid SWMM `[REPORT]` keyword and causes a pre-parse failure that produces an empty `.rpt` file.

**Fix:** The `[REPORT]` section uses only standard keywords (`INPUT`, `CONTROLS`, `SUBCATCHMENTS`, `NODES`, `LINKS`). Quality output is included automatically in `NODES ALL` output when pollutants are present — no extra keyword is needed.

---

## 9. Extending the Pipeline

### Increasing scenario count

For a full ML training dataset, run with `--n_scenarios 5000` or more. Each scenario takes approximately 1–2 seconds, so 5,000 scenarios require roughly 2–3 hours.

```bash
python dataset_generator.py --n_scenarios 5000 --output_dir ./output
```

### Adding decay

To model a reactive contaminant (e.g. bacteria with first-order die-off), change the `Kdecay` value in the `[POLLUTANTS]` declaration:

```
CONTAM  MG/L   0.0    0.0   0.0    0.05    NO
```

Units for `Kdecay` are 1/hour in SWMM.

### Using a different network

Replace `Example8.inp` with any SWMM 5.x input file. The system is now robust and will automatically:
- Inject a `[POLLUTANTS]` section if missing.
- Append injection `[TIMESERIES]` and `[INFLOWS]` to the correct sections.
- Handle different network topologies and node names.

You may still want to update:
- `HIGH_RISK_NODES` in `dataset_generator.py` to reflect your network's high-priority nodes.
- `EXCLUDE_SOURCE` in `dataset_generator.py` to match your outfall and storage node names.

### Parallelisation

Each scenario is independent. For large runs, scenarios can be parallelised using Python's `multiprocessing.Pool`. Each worker needs its own temporary `.inp` filename to avoid file conflicts:

```python
tmp_inp = f'_scenario_tmp_{worker_id}.inp'
```

---

## 10. Reference

Sambito, M., Di Cristo, C., Freni, G., and Leopardi, A. (2020). Optimal water quality sensor positioning in urban drainage systems for illicit intrusion identification. _Journal of Hydroinformatics_, 22(1), 46–60.
https://doi.org/10.2166/hydro.2019.036
