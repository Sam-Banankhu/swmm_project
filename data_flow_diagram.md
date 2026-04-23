````mermaid
graph TD
    A[Example8.inp] --> B[dataset_generator.py]
    B --> C[raw_scenarios.csv]
    B --> D[node_features.csv]

    C --> E[feature_engineering.py]
    D --> E
    A --> E
    E --> F[node_features_full.csv]

    F --> G[train_models.py]
    C --> G
    A --> G
    G --> H[ml_output/priors/prior_*.csv]

    C --> I[bdn_solver.py]
    F --> I
    H --> I
    I --> J[bdn_output/results/comparison_table.csv]

    K[dump/train_eval_pipeline.py] --> L[End-to-end workflow]
    L --> M[ml_results/ with figures/]
```</content>
<parameter name="filePath">c:\Users\kamlo\Desktop\Personal\projects\swmm_project\data_flow_diagram.md
````
