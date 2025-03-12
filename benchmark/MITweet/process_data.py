#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/LST1836/MITweet.git"
REPO_DIR = os.path.join(CURRENT_DIR, "MITweet_repo")
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Define label columns based on the official loading script
R_LABEL_COLS = ['R1-1-1', 'R2-1-2', 'R3-2-1', 'R4-2-2', 'R5-3-1', 'R6-3-2',
                'R7-3-3', 'R8-4-1', 'R9-4-2', 'R10-5-1', 'R11-5-2', 'R12-5-3']
I_LABEL_COLS = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12']
DOMAIN_COLS = ['R1', 'R2', 'R3', 'R4', 'R5']

# All label columns
ALL_LABEL_COLS = DOMAIN_COLS + R_LABEL_COLS + I_LABEL_COLS

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clone_repository():
    """Clone the MITweet repository."""
    print("\n" + "="*80)
    print("Downloading MITweet Dataset")
    print("="*80)
    
    # Remove repository if it already exists
    if os.path.exists(REPO_DIR):
        print(f"Removing existing repository at {REPO_DIR}...")
        shutil.rmtree(REPO_DIR)
    
    print(f"\nCloning MITweet repository from {REPO_URL}...")
    try:
        subprocess.run(
            ["git", "clone", REPO_URL, REPO_DIR],
            check=True
        )
        print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False

def copy_data_files():
    """Copy the data files from the repository to the data directory."""
    source_dir = os.path.join(REPO_DIR, "data", "random_split")
    
    if not os.path.exists(source_dir):
        print(f"Error: Data directory not found at {source_dir}")
        return False
    
    print(f"\nCopying data files from {source_dir} to {DATA_DIR}...")
    
    # Copy all files from the random_split directory
    for filename in ["train.csv", "val.csv", "test.csv", "Indicators.txt"]:
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Copied {filename}")
        else:
            print(f"Warning: File {filename} not found in {source_dir}")
    
    print("Data files copied successfully.")
    return True

def process_file(input_file, output_file):
    """Process a CSV file and convert it to JSONL format with all original labels."""
    print(f"Processing {input_file}...")
    
    processed_count = 0
    
    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(input_file)
        
        # Open output file
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            # Process each row
            for _, row in df.iterrows():
                # Extract text and topic
                text = row['tweet']
                topic = row['topic']
                
                # Extract all labels
                labels = {}
                for col in ALL_LABEL_COLS:
                    if col in row:
                        # Convert to int if possible
                        try:
                            labels[col] = int(row[col])
                        except (ValueError, TypeError):
                            # Skip if not convertible to int
                            print(f"Warning: Could not convert {col} value '{row[col]}' to int. Skipping this label.")
                
                # Create output object
                output_obj = {
                    "text": text,
                    "topic": topic,
                    "label": labels
                }
                
                # Write to JSONL file
                jsonl_file.write(json.dumps(output_obj) + '\n')
                processed_count += 1
        
        print(f"Processed {processed_count} examples from {input_file}")
        return processed_count
    
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return 0

def clean_up():
    """Clean up temporary files and directories."""
    print("\nCleaning up...")
    
    # Delete the data files
    for filename in ["train.csv", "val.csv", "test.csv", "Indicators.txt"]:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
    
    # Delete the repository
    if os.path.exists(REPO_DIR):
        try:
            shutil.rmtree(REPO_DIR)
            print(f"Deleted repository directory: {REPO_DIR}")
        except Exception as e:
            print(f"Error deleting repository: {e}")
    
    print("Cleanup complete.")

def main():
    """Main function to download and process the MITweet dataset."""
    parser = argparse.ArgumentParser(description='Download and process MITweet dataset')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files after processing')
    args = parser.parse_args()
    
    print("Starting MITweet dataset download and processing...")
    
    # Step 1: Clone the repository
    if not clone_repository():
        print("Failed to clone repository. Exiting.")
        sys.exit(1)
    
    # Step 2: Copy data files
    if not copy_data_files():
        print("Failed to copy data files. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Processing MITweet Dataset")
    print("="*80)
    
    # Step 3: Process each file
    train_count = process_file(
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(PROCESSED_DIR, "train.jsonl")
    )
    
    val_count = process_file(
        os.path.join(DATA_DIR, "val.csv"),
        os.path.join(PROCESSED_DIR, "valid.jsonl")
    )
    
    test_count = process_file(
        os.path.join(DATA_DIR, "test.csv"),
        os.path.join(PROCESSED_DIR, "test.jsonl")
    )
    
    # Print summary
    print("\nProcessing summary:")
    print(f"Train set: {train_count} examples")
    print(f"Validation set: {val_count} examples")
    print(f"Test set: {test_count} examples")
    print(f"Total: {train_count + val_count + test_count} examples")
    
    # Step 4: Clean up if requested
    if not args.keep_temp:
        clean_up()
    
    print("\n" + "="*80)
    print("Download and processing complete!")
    print(f"Processed data files are available in: {PROCESSED_DIR}")
    print("="*80)

if __name__ == "__main__":
    main() 