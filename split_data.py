import os
import argparse
import pandas as pd
import numpy as np

def main(input_csv, output_train, output_test, split_ratio, seed):
    print("=" * 60)
    print("Dataset Splitter -- Train/Test")
    print("=" * 60)
    
    np.random.seed(seed)
    
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    unique_scens = df['scen_id'].unique()
    n_total = len(unique_scens)
    n_train = int(n_total * split_ratio)
    
    print(f"Total unique scenarios: {n_total}")
    
    # Shuffle and split
    np.random.shuffle(unique_scens)
    train_ids = unique_scens[:n_train]
    test_ids = unique_scens[n_train:]
    
    print(f"Splitting into {len(train_ids)} train and {len(test_ids)} test scenarios.")
    
    train_df = df[df['scen_id'].isin(train_ids)]
    test_df = df[df['scen_id'].isin(test_ids)]
    
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"Saved train split to {output_train}")
    print(f"Saved test split to {output_test}")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./output/raw_scenarios.csv')
    parser.add_argument('--out_train', default='./output/raw_train.csv')
    parser.add_argument('--out_test', default='./output/raw_test.csv')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(args.input, args.out_train, args.out_test, args.ratio, args.seed)
