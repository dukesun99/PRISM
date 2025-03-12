#!/usr/bin/env python3
import os
import json
import random
import argparse
import shutil
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
PROCESSED_DIR = os.path.join(CURRENT_DIR, 'processed_data')

# Input files
INPUT_FILES = {
    "left": "BIGNEWSBLN_left.json",
    "center": "BIGNEWSBLN_center.json",
    "right": "BIGNEWSBLN_right.json"
}

# Label mapping
LABEL_MAPPING = {
    "left": "0",
    "center": "1",
    "right": "2"
}

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def find_input_files():
    """Find the input files in the data directory and its subdirectories."""
    found_files = {}
    
    for bias, filename in INPUT_FILES.items():
        # Search for the file in the data directory and its subdirectories
        for root, _, files in os.walk(DATA_DIR):
            if filename in files:
                found_files[bias] = os.path.join(root, filename)
                break
    
    return found_files

def process_text(text_data):
    """Process text data to convert from list to string and handle Unicode escapes."""
    # If text is a list, join it with spaces
    if isinstance(text_data, list):
        text = " ".join(text_data)
    else:
        text = text_data
    
    # The text is already properly decoded when loaded from JSON
    # No need to manually decode Unicode escape sequences
    return text

def load_data(file_paths):
    """Load data from the input files."""
    data = []
    
    for bias, file_path in file_paths.items():
        print(f"Loading {bias} data from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                
                # Add bias label to each article
                for article in articles:
                    if 'text' in article:
                        # Process the text to convert from list to string
                        processed_text = process_text(article['text'])
                        
                        data.append({
                            'text': processed_text,
                            'label': LABEL_MAPPING[bias]
                        })
                    else:
                        print(f"Warning: Article without 'text' field found in {bias} data")
                
                print(f"Loaded {len(articles)} articles from {bias} data")
        except Exception as e:
            print(f"Error loading {bias} data: {e}")
    
    return data

def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """Split data into train, validation, and test sets."""
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split indices
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    # Split the data
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    print(f"Data split: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test")
    
    return train_data, valid_data, test_data

def save_data(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} examples to {file_path}")

def count_labels(data):
    """Count the number of examples for each label."""
    counts = {"0": 0, "1": 0, "2": 0}
    
    for item in data:
        counts[item['label']] += 1
    
    return counts

def clean_source_data(file_paths):
    """Delete source data files to save space."""
    print("\nCleaning up source data to save space...")
    
    # First, delete the specific input files
    for bias, file_path in file_paths.items():
        try:
            os.remove(file_path)
            print(f"Deleted {bias} data file: {file_path}")
        except Exception as e:
            print(f"Error deleting {bias} data file: {e}")
    
    # Check if the data directory is empty or only contains empty directories
    empty_dirs = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=False):
        # If this directory has no files and no non-empty subdirectories
        if not files and all(os.path.join(root, d) in empty_dirs for d in dirs):
            empty_dirs.append(root)
            if root != DATA_DIR:  # Don't delete the main data directory itself
                try:
                    os.rmdir(root)
                    print(f"Deleted empty directory: {root}")
                except Exception as e:
                    print(f"Error deleting directory {root}: {e}")
    
    print("Source data cleanup complete.")

def process_data(seed=42, clean_source=True):
    """Process the BIGNEWSBLN dataset."""
    print("\n" + "="*80)
    print("BIGNEWSBLN Dataset Processing")
    print("="*80)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find input files
    file_paths = find_input_files()
    
    if len(file_paths) != 3:
        print(f"Error: Could not find all required input files. Found {len(file_paths)} out of 3.")
        print("Make sure to run the download script first.")
        return
    
    # Load data
    data = load_data(file_paths)
    
    if not data:
        print("Error: No data loaded. Check the input files.")
        return
    
    print(f"\nLoaded a total of {len(data)} articles")
    
    # Split data
    train_data, valid_data, test_data = split_data(data)
    
    # Save data
    save_data(train_data, os.path.join(PROCESSED_DIR, 'train.jsonl'))
    save_data(valid_data, os.path.join(PROCESSED_DIR, 'valid.jsonl'))
    save_data(test_data, os.path.join(PROCESSED_DIR, 'test.jsonl'))
    
    # Count labels in train set
    train_counts = count_labels(train_data)
    
    print("\nLabel distribution in train set:")
    print(f"Left-leaning (0): {train_counts['0']} examples")
    print(f"Center (1): {train_counts['1']} examples")
    print(f"Right-leaning (2): {train_counts['2']} examples")
    
    # Clean up source data if requested
    if clean_source:
        clean_source_data(file_paths)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process BIGNEWSBLN dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    parser.add_argument('--keep-source', action='store_true', help='Keep source data files after processing')
    args = parser.parse_args()
    
    process_data(seed=args.seed, clean_source=not args.keep_source)

if __name__ == "__main__":
    main() 