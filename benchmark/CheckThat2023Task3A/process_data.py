#!/usr/bin/env python3
import os
import sys
import json
import argparse
import shutil
import pandas as pd
import glob
import requests
import zipfile
import io
import tarfile
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "data", "task_3A")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Dataset URL
DATASET_URL = "https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/archive/main/clef2023-checkthat-lab-main.zip?path=task3"
ZIP_FILE = os.path.join(DATA_DIR, "checkthat2023-task3.zip")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_dataset():
    """Download the dataset from GitLab."""
    print("\n" + "="*80)
    print("Downloading CheckThat! 2023 Task 3A Dataset")
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
    print("Extracting CheckThat! 2023 Task 3A Dataset")
    print("="*80)
    
    if not os.path.exists(ZIP_FILE):
        print(f"Zip file not found at {ZIP_FILE}. Please download it first.")
        return False
    
    try:
        print(f"Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Find the extracted directory
        extracted_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("clef2023-checkthat-lab-main")]
        if not extracted_dirs:
            print("Error: Could not find extracted directory.")
            return False
        
        extracted_dir = os.path.join(DATA_DIR, extracted_dirs[0])
        
        # Extract the task3A data
        task3_zip = os.path.join(extracted_dir, "task3/data/task_3A.tar.gz")
        if not os.path.exists(task3_zip):
            print(f"Error: Task 3A data not found at {task3_zip}")
            return False
        
        print(f"Extracting Task 3A data from {task3_zip}...")
        with tarfile.open(task3_zip, 'r:gz') as tar_ref:
            tar_ref.extractall(DATA_DIR)
        
        print("Dataset extracted successfully.")
        return True
    
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def process_data(input_tsv, output_jsonl, base_path=CURRENT_DIR):
    """Process the data from TSV and JSON files to JSONL format."""
    print(f"Processing {input_tsv}...")
    
    # Read the TSV file
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Open output file
    with open(output_jsonl, 'w', encoding='utf-8') as out_file:
        # Process each row
        processed_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            json_file_path = row['json_file_path']
            label = int(row['label'])
            
            # Fix JSON file path
            # Path format in TSV is "data/task_3A/dev_json/xxx.json"
            # We need to convert it to an absolute path
            
            # Method 1: Use CURRENT_DIR as the base path
            absolute_path = os.path.join(CURRENT_DIR, json_file_path)
            
            # Method 2: If method 1 fails, try to find the file by name in the correct directory
            if not os.path.exists(absolute_path):
                json_file_name = os.path.basename(json_file_path)
                if "dev_json" in json_file_path:
                    absolute_path = os.path.join(RAW_DATA_DIR, "dev_json", json_file_name)
                elif "train_json" in json_file_path:
                    absolute_path = os.path.join(RAW_DATA_DIR, "train_json", json_file_name)
            
            # Read the JSON file
            try:
                with open(absolute_path, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                
                # Extract title and content
                title = article_data.get('title', '')
                content = article_data.get('content', '')
                
                # Combine title and content
                text = title + " " + content if title else content
                
                # Create output object
                output_obj = {
                    "text": text,
                    "label": label
                }
                
                # Write to JSONL file
                out_file.write(json.dumps(output_obj) + '\n')
                processed_count += 1
            except Exception as e:
                print(f"Error processing {absolute_path}: {e}")
    
    print(f"Processed data saved to {output_jsonl} ({processed_count} examples)")

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
    
    # Delete extracted directories
    extracted_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("clef2023-checkthat-lab-main")]
    for d in extracted_dirs:
        shutil.rmtree(os.path.join(DATA_DIR, d))
        print(f"Deleted directory {d}")
    
    # Delete the extracted data directory
    data_subdir = os.path.join(DATA_DIR, "data")
    if os.path.exists(data_subdir):
        shutil.rmtree(data_subdir)
        print(f"Deleted directory {data_subdir}")
    
    print("Raw data files deleted.")

def main():
    """Main function to process the CheckThat! 2023 Task 3A dataset."""
    parser = argparse.ArgumentParser(description='Process CheckThat! 2023 Task 3A dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw data files after processing')
    args = parser.parse_args()
    
    print("Starting CheckThat! 2023 Task 3A dataset processing...")
    
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
    
    # Step 3: Process training data
    train_tsv = os.path.join(RAW_DATA_DIR, "task_3A_news_article_bias_train.tsv")
    train_jsonl = os.path.join(PROCESSED_DIR, "train.jsonl")
    process_data(train_tsv, train_jsonl)
    
    # Step 4: Process development data
    dev_tsv = os.path.join(RAW_DATA_DIR, "task_3A_news_article_bias_dev.tsv")
    dev_jsonl = os.path.join(PROCESSED_DIR, "valid.jsonl")
    process_data(dev_tsv, dev_jsonl)
    
    # Step 5: Create a copy of the development data as test data (since we don't have test data)
    test_jsonl = os.path.join(PROCESSED_DIR, "test.jsonl")
    shutil.copy(dev_jsonl, test_jsonl)
    print(f"Created test set (copy of dev set) at {test_jsonl}")
    
    # Count examples in each split
    train_count = sum(1 for _ in open(train_jsonl, 'r'))
    valid_count = sum(1 for _ in open(dev_jsonl, 'r'))
    test_count = sum(1 for _ in open(test_jsonl, 'r'))
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Processed data files are available in: {PROCESSED_DIR}")
    print(f"Train set: {train_count} examples")
    print(f"Validation set: {valid_count} examples")
    print(f"Test set: {test_count} examples")
    print("="*80)
    
    # Step 6: Clean up if requested
    clean_up(args.keep_raw)
    
if __name__ == "__main__":
    main() 