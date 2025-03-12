#!/usr/bin/env python3
import os
import sys
import json
import argparse
import shutil
import random
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE.git"
REPO_DIR = os.path.join(CURRENT_DIR, "BABE_repo")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define label mapping
LABEL_MAPPING = {
    "Biased": 1,
    "Non-biased": 0
}

# Define dataset names
DATASETS = ["SG1", "SG2", "MBIC"]

def clone_repository():
    """Clone the BABE repository."""
    print("\n" + "="*80)
    print("Cloning BABE Repository")
    print("="*80)
    
    if os.path.exists(REPO_DIR):
        print(f"Repository already exists at {REPO_DIR}. Skipping clone.")
        return True
    
    try:
        print(f"Cloning repository from {REPO_URL}...")
        subprocess.run(
            ["git", "clone", REPO_URL, REPO_DIR],
            check=True
        )
        print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False

def load_dataset(dataset_name):
    """Load a specific dataset from the repository."""
    print(f"\nLoading {dataset_name} dataset...")
    
    data_dir = os.path.join(REPO_DIR, "data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found at {data_dir}.")
        return None
    
    # CSV file for this dataset
    csv_file = f"final_labels_{dataset_name}.csv"
    file_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(file_path):
        print(f"Warning: File {csv_file} not found. Skipping.")
        return None
    
    data = []
    
    try:
        # Read CSV file with semicolon as delimiter
        df = pd.read_csv(file_path, sep=';')
        
        # Check if required columns exist
        if 'text' not in df.columns or 'label_bias' not in df.columns:
            print(f"Warning: Required columns missing in {csv_file}. Skipping.")
            return None
        
        # Filter for only Biased and Non-biased labels
        df = df[df['label_bias'].isin(['Biased', 'Non-biased'])]
        
        # Extract text and label
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            label = LABEL_MAPPING.get(row['label_bias'], -1)
            
            if label == -1:
                print(f"Warning: Unknown label '{row['label_bias']}'. Skipping.")
                continue
            
            data.append({
                'text': text,
                'label': label
            })
        
        print(f"Loaded {len(data)} examples from {dataset_name}.")
        return data
    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Split data into train, validation, and test sets."""
    if not data:
        return [], [], []
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split indices
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    # Split the data
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    return train_data, valid_data, test_data

def save_data(data, file_path):
    """Save data to a JSONL file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} examples to {file_path}")

def count_labels(data):
    """Count the distribution of labels in the data."""
    label_counts = {0: 0, 1: 0}  # Non-biased, Biased
    
    for item in data:
        label = item['label']
        label_counts[label] += 1
    
    return label_counts

def clean_up(keep_repo=False):
    """Clean up temporary files."""
    if keep_repo:
        print("\nKeeping repository as requested.")
        return
    
    print("\nCleaning up repository...")
    
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
        print(f"Deleted repository directory: {REPO_DIR}")
    
    print("Repository deleted.")

def process_all_datasets(seed=42):
    """Process all datasets separately."""
    print("\n" + "="*80)
    print("Processing All Datasets")
    print("="*80)
    
    # Also create a combined dataset
    all_train_data = []
    all_valid_data = []
    all_test_data = []
    
    for dataset_name in DATASETS:
        print(f"\n{'-'*40}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'-'*40}")
        
        # Load dataset
        data = load_dataset(dataset_name)
        if not data:
            print(f"Skipping {dataset_name} dataset.")
            continue
        
        # Create dataset directory
        dataset_dir = os.path.join(PROCESSED_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Split data
        print(f"Splitting {dataset_name} data...")
        train_data, valid_data, test_data = split_data(data, seed=seed)
        
        print(f"{dataset_name} split data into:")
        print(f"  Train: {len(train_data)} examples")
        print(f"  Validation: {len(valid_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        # Save data
        save_data(train_data, os.path.join(dataset_dir, "train.jsonl"))
        save_data(valid_data, os.path.join(dataset_dir, "valid.jsonl"))
        save_data(test_data, os.path.join(dataset_dir, "test.jsonl"))
        
        # Count label distribution
        train_counts = count_labels(train_data)
        valid_counts = count_labels(valid_data)
        test_counts = count_labels(test_data)
        
        print(f"\n{dataset_name} Label Distribution:")
        print(f"Train: Non-biased={train_counts[0]}, Biased={train_counts[1]}")
        print(f"Valid: Non-biased={valid_counts[0]}, Biased={valid_counts[1]}")
        print(f"Test: Non-biased={test_counts[0]}, Biased={test_counts[1]}")
        
        # Add to combined dataset
        all_train_data.extend(train_data)
        all_valid_data.extend(valid_data)
        all_test_data.extend(test_data)
    
    # Process combined dataset
    print(f"\n{'-'*40}")
    print("Processing Combined Dataset")
    print(f"{'-'*40}")
    
    # Create combined directory
    combined_dir = os.path.join(PROCESSED_DIR, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Save combined data
    save_data(all_train_data, os.path.join(combined_dir, "train.jsonl"))
    save_data(all_valid_data, os.path.join(combined_dir, "valid.jsonl"))
    save_data(all_test_data, os.path.join(combined_dir, "test.jsonl"))
    
    # Count combined label distribution
    all_train_counts = count_labels(all_train_data)
    all_valid_counts = count_labels(all_valid_data)
    all_test_counts = count_labels(all_test_data)
    
    print("\nCombined Label Distribution:")
    print(f"Train: Non-biased={all_train_counts[0]}, Biased={all_train_counts[1]}")
    print(f"Valid: Non-biased={all_valid_counts[0]}, Biased={all_valid_counts[1]}")
    print(f"Test: Non-biased={all_test_counts[0]}, Biased={all_test_counts[1]}")
    
    return True

def main():
    """Main function to process the BABE dataset."""
    parser = argparse.ArgumentParser(description='Process BABE dataset')
    parser.add_argument('--keep-repo', action='store_true', help='Keep repository after processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    args = parser.parse_args()
    
    print("Starting BABE dataset processing...")
    
    # Step 1: Clone the repository
    if not clone_repository():
        print("Failed to clone repository. Exiting.")
        sys.exit(1)
    
    # Step 2: Process all datasets
    if not process_all_datasets(seed=args.seed):
        print("Failed to process datasets. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Processed data files are available in: {PROCESSED_DIR}")
    print("="*80)
    
    # Step 3: Clean up
    clean_up(args.keep_repo)

if __name__ == "__main__":
    main() 