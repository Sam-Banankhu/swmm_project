import os
import subprocess
import yaml
import sys

def run_command(cmd):
    print(f"\n>>> Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

def main():
    print("====================================================")
    print("   Hybrid AI-BDN Sensor Placement Pipeline")
    print("====================================================")

    # 1. Load config
    config_path = "config/default.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config['dataset']['model_path']
    n_scenarios = config['dataset']['n_scenarios']
    output_dir = config['dataset']['output_dir']

    # Step 1: Generate Dataset
    run_command(["python", "dataset_generator.py"])

    # Step 2: Split Data (Prevent Leakage)
    raw_scenarios = os.path.join(output_dir, "raw_scenarios.csv")
    raw_train = os.path.join(output_dir, "raw_train.csv")
    raw_test = os.path.join(output_dir, "raw_test.csv")
    
    run_command([
        "python", "split_data.py",
        "--input", raw_scenarios,
        "--out_train", raw_train,
        "--out_test", raw_test
    ])

    # Step 3: Feature Engineering (Training Split)
    node_features = os.path.join(output_dir, "node_features.csv")
    node_features_train = os.path.join(output_dir, "node_features_train.csv")
    
    run_command([
        "python", "feature_engineering.py",
        "--node_features", node_features,
        "--raw_scenarios", raw_train,
        "--model_path", model_path,
        "--output", node_features_train
    ])

    # Step 4: Train Models
    ml_output = config['ml']['output_dir']
    run_command([
        "python", "train_models.py",
        "--node_features", node_features_train,
        "--raw_scenarios", raw_train,
        "--model_path", model_path,
        "--output_dir", ml_output
    ])

    # Step 5: Final BDN Solver (Evaluate on Test Set)
    bdn_output = config['bdn']['output_dir']
    priors_dir = os.path.join(ml_output, "priors")
    
    run_command([
        "python", "bdn_solver.py",
        "--raw_scenarios", raw_test,
        "--node_features", node_features_train,
        "--priors_dir", priors_dir,
        "--output_dir", bdn_output
    ])

    print("\n====================================================")
    print("   Pipeline Complete!")
    print(f"   Results are in: {bdn_output}")
    print("====================================================")

if __name__ == "__main__":
    main()
