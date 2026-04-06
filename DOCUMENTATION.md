# SWMM Example 8 — Combined Sewer System
### Hybrid AI Sensor Placement Research
**Authors:** Mhango, S.B. and Sambito, M. (2026)
**Source:** EPA SWMM Applications Manual, Example 8 (pages 133–154)

---

## Overview

This project generates a machine-learning training dataset for optimal water-quality sensor placement in urban drainage systems. A combined sewer network (EPA SWMM Example 8) is used as the simulation testbed. The Python pipeline injects synthetic contamination events at network nodes, runs hydraulic and water-quality simulations via [EPA SWMM 5.2](https://www.epa.gov/water-research/storm-water-management-model-swmm), and records peak concentrations and detection flags at every node across hundreds of scenarios.

The resulting dataset feeds a hybrid AI model that recommends optimal sensor locations — balancing detection coverage, network topology, and prior contamination probability.

---

## Project Structure

```
swmm_project/
├── Example8.inp            # SWMM input file — combined sewer network (with [POLLUTANTS])
├── Example8_test.inp       # Alternate version used for testing
├── dataset_generator.py    # Main pipeline — runs scenarios and builds the dataset
├── README.txt              # Original plain-text notes
└── output_sample/
    ├── raw_scenarios.csv   # One row per (scenario × node)
    └── node_features.csv   # One row per node — aggregated features
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install pyswmm swmm-toolkit networkx pandas numpy
```

### 2. Run the generator

```bash
# 100 scenarios (quick test)
python dataset_generator.py --n_scenarios 100 --output_dir ./output

# 5000 scenarios (full ML training dataset)
python dataset_generator.py --n_scenarios 5000 --output_dir ./output
```

### 3. Outputs

| File | Description |
|---|---|
| `output/raw_scenarios.csv` | One row per (scenario, node). 13 columns. |
| `output/node_features.csv` | One row per node. Topology + aggregated detection features. |

---

## Network Summary

The network models a **29-acre urban combined sewer catchment** with the following components:

| Component | Count | Notes |
|---|---|---|
| Subcatchments | 7 | S1–S7 |
| Junction nodes | 28 | Combined sewer (J-) and interceptor (JI-) |
| Outfalls | 2 | O1 = stream, O2 = WWTP |
| Storage node | 1 | Well (pump wet well) |
| Stream conduits | 10 | C3–C11, C_Aux3 |
| Combined sewer pipes | 6 | P1–P6 |
| Interceptor pipes | 9 | I1–I9 |
| Force mains | 4 | I10–I13 (downstream of pump) |
| Flow regulators | 5 | R1–R5 (weirs W1–W4, orifice Or1) |
| Pump | 1 | Pump1, Type 3 curve |

**Simulation:** 12-hour run, 0.23-inch rainfall event, 15-second routing step (DYNWAVE).

### Node naming convention

| Prefix | Type |
|---|---|
| `Jx` | Combined sewer junction |
| `JIx` | Interceptor junction |
| `Aux3` | Flow splitting node |
| `Well` | Pump wet well (storage) |
| `O1`, `O2` | Stream and WWTP outfalls |

---

## High-Risk Nodes

Based on Sambito et al. (2020), three nodes have **double the baseline contamination probability** and are sampled at twice the rate during scenario generation:

- **J4**, **J10**, **JI18**

---

## Opening in EPA SWMM GUI

1. Download and install [EPA SWMM 5.2](https://www.epa.gov/water-research/storm-water-management-model-swmm)
2. Open SWMM → **File > Open** → select `Example8.inp`
3. Click the green **Run** button (lightning bolt icon)
4. To view contamination: right-click any node → **Graph > Quality**

---

## Reference

Sambito, M., Di Cristo, C., Freni, G., and Leopardi, A. (2020). Optimal water quality sensor positioning in urban drainage systems for illicit intrusion identification. *Journal of Hydroinformatics*, 22(1), 46–60. https://doi.org/10.2166/hydro.2019.036