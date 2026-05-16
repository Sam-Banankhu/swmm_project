# Research Summary: Hybrid AI for Optimal Sensor Placement

This summary outlines the technical architecture and research value of the **Hybrid AI-BDN framework** for water quality sensor placement in urban drainage systems.

## 1. Research Problem
Water quality monitoring in combined sewer networks is critical but computationally challenging. The goal is to identify the **optimal $N$ locations** for sensors that maximize the probability of detecting and isolating contamination events.

*   **The Baseline (v1.0)**: Traditional approaches use Bayesian Decision Networks (BDN) but rely on a **uniform prior** or expensive **Monte Carlo (MC)** simulations to estimate initial node importance. This leads to slow convergence and high computational overhead.

## 2. The Hybrid AI Innovation
Our project implements the **Hybrid AI approach** (Mhango & Sambito, 2026), which replaces the generic BDN prior with a **Machine Learning-derived prior**.

Instead of starting from "zero knowledge," we train ML models to "learn" the relationship between a node's physical location (topology) and its detection probability (hydraulics).

## 3. Technical Pipeline

### Phase A: Large-Scale Simulation (The "Digital Twin")
We use **PySWMM** to run 5,000+ contamination scenarios on benchmark networks (e.g., Example 8). We vary the injection location, pollutant mass, and event duration. This creates a rich database of how contamination spreads through the network.

### Phase B: Advanced Feature Engineering
We extract four categories of "signals" for every node in the network:
1.  **Static Topology**: Betweenness centrality, downstream path counts, and distance to outfalls.
2.  **Hydraulic Influence**: Flow diversion fractions at weirs and orifices.
3.  **Bayesian Baselines**: Mean wastewater and contaminant flux (Priors C and D).
4.  **Dynamic Statistics**: Mean peak concentration and time-to-peak.

### Phase C: Machine Learning Prior Generation
We implement and compare three ML architectures:
*   **XGBoost (Tabular)**: Captures non-linear dependencies in tabular data.
*   **MLP (Deep Learning)**: Neural network for complex feature mapping.
*   **Graph Neural Networks (GCN/GAT)**: Learns directly from the network's graph structure (nodes and pipes).

**Output**: A data-driven prior vector that ranks every node's inherent importance for monitoring.

### Phase D: BDN Solver & Evaluation
We plug the ML-predicted probabilities into the **Bayesian Decision Network**. 
*   **Metrics**: We evaluate the sensor configuration using **F1 (Isolation Likelihood)** and **F2 (Detection Reliability)**.
*   **Convergence**: We compare how much faster the Hybrid AI model converges compared to traditional baselines.

## 4. Key Takeaways for Research
1.  **Efficiency**: Reaches optimal placement significantly faster than models with uniform priors.
2.  **Precision**: Better handles complex hydraulics (weirs/regulators) by learning from historical simulation data.
3.  **Scalability**: The framework is modular, allowing for easy testing on larger networks like the Massa Lubrense case study.
