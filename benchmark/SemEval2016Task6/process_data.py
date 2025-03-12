#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
import requests
import zipfile
import shutil
import random
import pandas as pd
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Dataset URL
DATASET_URL = "http://alt.qcri.org/semeval2016/task6/data/uploads/stancedataset.zip"
ZIP_FILE = os.path.join(DATA_DIR, "stancedataset.zip")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_dataset():
    """Download the dataset from the provided URL."""
    print("\n" + "="*80)
    print("Downloading SemEval2016 Task 6 Dataset")
    print("="*80)
    
    if os.path.exists(ZIP_FILE):
        print(f"Zip file already exists at {ZIP_FILE}. Skipping download.")
        return True
    
    print(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(ZIP_FILE, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded dataset to {ZIP_FILE}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return False

def extract_dataset():
    """Extract the dataset from the zip file."""
    print("\n" + "="*80)
    print("Extracting SemEval2016 Task 6 Dataset")
    print("="*80)
    
    if not os.path.exists(ZIP_FILE):
        print(f"Zip file not found at {ZIP_FILE}. Please download it first.")
        return False
    
    try:
        print(f"Examining zip file: {ZIP_FILE}")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"Files in the zip archive: {file_list}")
        
        print(f"Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Check if the expected files exist
        stance_dataset_dir = os.path.join(DATA_DIR, "StanceDataset")
        if not os.path.exists(stance_dataset_dir):
            print(f"StanceDataset directory not found after extraction. Creating it...")
            os.makedirs(stance_dataset_dir, exist_ok=True)
        
        train_file = os.path.join(stance_dataset_dir, "train.csv")
        test_file = os.path.join(stance_dataset_dir, "test.csv")
        
        # Check if files were extracted to the root of DATA_DIR instead
        if not os.path.exists(train_file):
            root_train_file = os.path.join(DATA_DIR, "train.csv")
            if os.path.exists(root_train_file):
                print(f"Found train.csv in {DATA_DIR}, moving to {stance_dataset_dir}")
                shutil.move(root_train_file, train_file)
        
        if not os.path.exists(test_file):
            root_test_file = os.path.join(DATA_DIR, "test.csv")
            if os.path.exists(root_test_file):
                print(f"Found test.csv in {DATA_DIR}, moving to {stance_dataset_dir}")
                shutil.move(root_test_file, test_file)
        
        # Final check
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Required files not found after extraction:")
            print(f"  train.csv: {os.path.exists(train_file)}")
            print(f"  test.csv: {os.path.exists(test_file)}")
            
            # List all files in DATA_DIR to help debug
            print(f"\nListing all files in {DATA_DIR}:")
            for root, dirs, files in os.walk(DATA_DIR):
                for file in files:
                    print(os.path.join(root, file))
            
            return False
        
        print("Dataset extracted successfully.")
        return True
    
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def manual_parse_csv(file_path):
    """Manually parse a CSV file to avoid encoding and parsing issues."""
    print(f"Manually parsing CSV file: {file_path}")
    
    data = []
    headers = None
    
    try:
        # Read the file with a more tolerant approach
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"File is empty: {file_path}")
            return None
        
        # Parse header
        headers = [h.strip() for h in lines[0].strip().split(',')]
        print(f"Headers: {headers}")
        
        # Check if required columns exist
        required_columns = ['Tweet', 'Target', 'Stance']
        for col in required_columns:
            if col not in headers:
                print(f"Required column '{col}' not found in headers: {headers}")
                return None
        
        # Parse data rows
        for i, line in enumerate(lines[1:], 1):
            try:
                # Handle quoted fields with commas
                fields = []
                in_quotes = False
                current_field = ""
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        fields.append(current_field)
                        current_field = ""
                    else:
                        current_field += char
                
                # Don't forget the last field
                fields.append(current_field)
                
                # Ensure we have the right number of fields
                if len(fields) != len(headers):
                    # Try to fix by adjusting the number of fields
                    if len(fields) > len(headers):
                        # Combine extra fields into the last expected field
                        combined = fields[len(headers)-1:]
                        fields = fields[:len(headers)-1] + [','.join(combined)]
                    else:
                        # Add empty fields if needed
                        fields.extend([''] * (len(headers) - len(fields)))
                
                # Create a dictionary for this row
                row_dict = {headers[j]: fields[j] for j in range(len(headers))}
                data.append(row_dict)
                
            except Exception as e:
                print(f"Error parsing line {i+1}: {e}")
                # Skip problematic lines
                continue
        
        print(f"Successfully parsed {len(data)} rows from {file_path}")
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error manually parsing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None

def split_test_data():
    """Split the test data into test and validation sets."""
    print("\n" + "="*80)
    print("Splitting Test Data into Test and Validation Sets")
    print("="*80)
    
    test_file = os.path.join(DATA_DIR, "StanceDataset", "test.csv")
    
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}.")
        return False
    
    try:
        # Manually parse the test file
        test_df = manual_parse_csv(test_file)
        
        if test_df is None:
            print("Failed to parse test file.")
            return False
        
        print(f"Successfully parsed test file. Shape: {test_df.shape}")
        print(f"Columns: {test_df.columns.tolist()}")
        
        # Group by target
        grouped = test_df.groupby('Target')
        
        # Create empty dataframes for validation and new test
        valid_df = pd.DataFrame(columns=test_df.columns)
        new_test_df = pd.DataFrame(columns=test_df.columns)
        
        # For each target, split the data in half
        for target, group in grouped:
            print(f"Processing target: {target} with {len(group)} examples")
            
            # Shuffle the data
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split in half
            half_idx = len(group) // 2
            valid_group = group.iloc[:half_idx]
            test_group = group.iloc[half_idx:]
            
            # Append to the respective dataframes
            valid_df = pd.concat([valid_df, valid_group])
            new_test_df = pd.concat([new_test_df, test_group])
        
        # Save the validation and new test data
        valid_file = os.path.join(DATA_DIR, "valid.csv")
        new_test_file = os.path.join(DATA_DIR, "test_split.csv")
        
        valid_df.to_csv(valid_file, index=False)
        new_test_df.to_csv(new_test_file, index=False)
        
        print(f"Split test data into validation ({len(valid_df)} examples) and new test ({len(new_test_df)} examples).")
        return True
    
    except Exception as e:
        print(f"Error splitting test data: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_data():
    """Process the data and organize it into subdatasets based on targets."""
    print("\n" + "="*80)
    print("Processing Data into Target-Specific Subdatasets")
    print("="*80)
    
    train_file = os.path.join(DATA_DIR, "StanceDataset", "train.csv")
    valid_file = os.path.join(DATA_DIR, "valid.csv")
    test_file = os.path.join(DATA_DIR, "test_split.csv")
    
    if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
        print("One or more data files not found. Please make sure the dataset is downloaded and extracted.")
        return False
    
    try:
        # Manually parse the train file
        train_df = manual_parse_csv(train_file)
        
        if train_df is None:
            print("Failed to parse train file.")
            return False
        
        # Read validation and test files (these should be clean as we created them)
        print(f"Reading validation file: {valid_file}")
        valid_df = pd.read_csv(valid_file)
        
        print(f"Reading test file: {test_file}")
        test_df = pd.read_csv(test_file)
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Validation data shape: {valid_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Get unique targets
        all_targets = set(train_df['Target'].unique()) | set(valid_df['Target'].unique()) | set(test_df['Target'].unique())
        
        print(f"Found {len(all_targets)} unique targets: {', '.join(all_targets)}")
        
        # Process each target
        for target in all_targets:
            target_dir = os.path.join(PROCESSED_DIR, target.replace(" ", "_"))
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"\nProcessing target: {target}")
            
            # Filter data for this target
            target_train = train_df[train_df['Target'] == target]
            target_valid = valid_df[valid_df['Target'] == target]
            target_test = test_df[test_df['Target'] == target]
            
            print(f"Train: {len(target_train)} examples")
            print(f"Valid: {len(target_valid)} examples")
            print(f"Test: {len(target_test)} examples")
            
            # Convert stance labels to numeric
            stance_mapping = {'AGAINST': 0, 'NONE': 1, 'FAVOR': 2}
            
            # Process and save each split
            for split_name, df in [('train', target_train), ('valid', target_valid), ('test', target_test)]:
                if len(df) == 0:
                    print(f"No {split_name} data for target '{target}'")
                    continue
                
                output_file = os.path.join(target_dir, f"{split_name}.jsonl")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for _, row in df.iterrows():
                        output_obj = {
                            'text': row['Tweet'],
                            'label': stance_mapping.get(row['Stance'], -1)
                        }
                        f.write(json.dumps(output_obj) + '\n')
                
                print(f"Saved {len(df)} examples to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_examples():
    """Count the number of examples in each subdataset."""
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    # Get all target directories
    target_dirs = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    
    for target_dir in target_dirs:
        target_path = os.path.join(PROCESSED_DIR, target_dir)
        print(f"\nTarget: {target_dir.replace('_', ' ')}")
        
        for split in ['train', 'valid', 'test']:
            jsonl_file = os.path.join(target_path, f"{split}.jsonl")
            if os.path.exists(jsonl_file):
                # Count examples
                count = sum(1 for _ in open(jsonl_file, 'r', encoding='utf-8'))
                
                # Count stance distribution
                stance_counts = {0: 0, 1: 0, 2: 0}  # AGAINST, NONE, FAVOR
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        stance_counts[data['label']] += 1
                
                print(f"  {split}: {count} examples (AGAINST: {stance_counts[0]}, NONE: {stance_counts[1]}, FAVOR: {stance_counts[2]})")
            else:
                print(f"  {split}: No data")

def clean_up(keep_raw=False):
    """Clean up temporary files."""
    if keep_raw:
        print("\nKeeping raw data files as requested.")
        return
    
    print("\nCleaning up raw data files...")
    
    # Delete the zip file
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Deleted {ZIP_FILE}")
    
    # Delete the extracted CSV files
    for file in ['valid.csv', 'test_split.csv']:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
    
    # Delete the StanceDataset directory
    stance_dataset_dir = os.path.join(DATA_DIR, "StanceDataset")
    if os.path.exists(stance_dataset_dir):
        shutil.rmtree(stance_dataset_dir)
        print(f"Deleted directory {stance_dataset_dir}")
    
    print("Raw data files deleted.")

def main():
    """Main function to process the SemEval2016 Task 6 dataset."""
    parser = argparse.ArgumentParser(description='Process SemEval2016 Task 6 dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw data files after processing')
    args = parser.parse_args()
    
    print("Starting SemEval2016 Task 6 dataset processing...")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download as requested.")
    
    # Step 2: Extract the dataset
    if not args.skip_download:
        if not extract_dataset():
            print("Failed to extract dataset. Exiting.")
            sys.exit(1)
    
    # Step 3: Split the test data into test and validation sets
    if not split_test_data():
        print("Failed to split test data. Exiting.")
        sys.exit(1)
    
    # Step 4: Process the data into target-specific subdatasets
    if not process_data():
        print("Failed to process data. Exiting.")
        sys.exit(1)
    
    # Step 5: Count examples in each subdataset
    count_examples()
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Processed data files are available in: {PROCESSED_DIR}")
    print("="*80)
    
    # Step 6: Clean up if requested
    clean_up(args.keep_raw)
    
if __name__ == "__main__":
    main() 