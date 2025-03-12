#!/usr/bin/env python3
import os
import sys
import json
import argparse
import shutil
import requests
import pandas as pd
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Dataset URLs
DATASET_URLS = {
    "train": "https://zenodo.org/records/3271522/files/proppy_1.0.train.tsv?download=1",
    "dev": "https://zenodo.org/records/3271522/files/proppy_1.0.dev.tsv?download=1",
    "test": "https://zenodo.org/records/3271522/files/proppy_1.0.test.tsv?download=1"
}

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_dataset():
    """Download the Proppy dataset from Zenodo."""
    print("\n" + "="*80)
    print("Downloading Proppy Dataset")
    print("="*80)
    
    for split, url in DATASET_URLS.items():
        output_file = os.path.join(DATA_DIR, f"proppy_1.0.{split}.tsv")
        
        if os.path.exists(output_file):
            print(f"File already exists: {output_file}. Skipping download.")
            continue
        
        print(f"Downloading {split} split from {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {split}") as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Downloaded {split} split to {output_file}")
        
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {split} split: {e}")
            return False
    
    return True

def process_propaganda_labels(propaganda_label):
    """Process propaganda labels into binary categories."""
    if pd.isna(propaganda_label):
        return -1
    
    propaganda_label = str(propaganda_label).lower().strip()
    
    # Skip known invalid labels without printing warnings
    if propaganda_label in ["nil", "unknown", "none", ""]:
        return -1
    
    # Check for common patterns in propaganda labels
    if propaganda_label in ["yes", "1", "true", "propaganda"]:
        return 1  # Propaganda
    elif propaganda_label in ["no", "0", "false", "not propaganda"]:
        return 0  # Not propaganda
    
    # Try to handle numeric values
    try:
        # If it's a numeric string
        value = float(propaganda_label)
        if value == 1 or value > 0.5:  # Assuming values close to 1 mean propaganda
            return 1
        elif value == 0 or value < 0.5:  # Assuming values close to 0 mean not propaganda
            return 0
    except (ValueError, TypeError):
        pass
    
    # Handle special case for -1 which might be used to indicate "unknown"
    if propaganda_label == "-1":
        return -1
    
    # Only print unrecognized labels that aren't common
    # and limit the number of warnings to avoid flooding the console
    if not SUPPRESS_WARNINGS:
        global unrecognized_propaganda_labels
        if 'unrecognized_propaganda_labels' not in globals():
            unrecognized_propaganda_labels = set()
        
        if propaganda_label not in unrecognized_propaganda_labels and len(unrecognized_propaganda_labels) < 10:
            print(f"Unrecognized propaganda label: '{propaganda_label}'")
            unrecognized_propaganda_labels.add(propaganda_label)
        elif len(unrecognized_propaganda_labels) == 10:
            print("Too many unrecognized propaganda labels. Suppressing further warnings.")
            unrecognized_propaganda_labels.add("__suppressed__")
    
    return -1  # Unknown

def process_file(input_file, output_file):
    """Process a TSV file and extract text with propaganda labels."""
    print(f"Processing {input_file}...")
    
    try:
        # Try to read the first few lines to check the format
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            first_lines = [next(f) for _ in range(5) if _ < 5]
        
        print(f"First few lines of {input_file}:")
        for i, line in enumerate(first_lines):
            print(f"Line {i+1}: {line[:100]}...")  # Print first 100 chars of each line
        
        # Define column names based on the Proppy dataset documentation
        column_names = [
            'article_text', 'event_location', 'average_tone', 'article_date', 
            'article_ID', 'article_URL', 'author', 'title', 'MBFC_factuality_label', 
            'source_url', 'source_name', 'source_description', 'MBFC_bias_label', 
            'source_domain', 'propaganda_label'
        ]
        
        print(f"Using predefined column names: {column_names}")
        
        # Read TSV file with pandas, using predefined column names
        try:
            df = pd.read_csv(input_file, sep='\t', names=column_names, header=None, on_bad_lines='warn')
        except Exception as e:
            print(f"Error with default settings: {e}")
            print("Trying with different settings...")
            try:
                # Try with different encoding
                df = pd.read_csv(input_file, sep='\t', names=column_names, header=None, 
                                on_bad_lines='warn', encoding='latin1')
            except Exception as e:
                print(f"Error with latin1 encoding: {e}")
                # Try with Python engine
                df = pd.read_csv(input_file, sep='\t', names=column_names, header=None,
                                on_bad_lines='warn', engine='python')
        
        print(f"Successfully read {len(df)} rows from {input_file}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if the number of columns matches our expectation
        if len(df.columns) != len(column_names):
            print(f"Warning: Number of columns ({len(df.columns)}) doesn't match expected ({len(column_names)})")
            print("Attempting to adjust column names...")
            
            # If there are fewer columns than expected, use a subset of column names
            if len(df.columns) < len(column_names):
                # Try to ensure the important columns are included
                adjusted_names = column_names[:len(df.columns)]
                
                # Check if we have enough columns for the required ones
                required_indices = [0, 14]  # Indices for article_text, propaganda_label
                if len(df.columns) <= max(required_indices):
                    print(f"Error: Not enough columns in the file. Need at least {max(required_indices) + 1}, got {len(df.columns)}")
                    return False
                
                # Re-read with adjusted column names
                df = pd.read_csv(input_file, sep='\t', names=adjusted_names, header=None, 
                                on_bad_lines='warn', engine='python')
                print(f"Re-read with adjusted column names: {adjusted_names}")
            
            # If there are more columns than expected, add generic names for extra columns
            else:
                adjusted_names = column_names + [f'extra_{i}' for i in range(len(df.columns) - len(column_names))]
                df = pd.read_csv(input_file, sep='\t', names=adjusted_names, header=None, 
                                on_bad_lines='warn', engine='python')
                print(f"Re-read with extended column names: {adjusted_names[:5]}... (total: {len(adjusted_names)})")
        
        # Check if required columns exist
        required_columns = ['article_text', 'propaganda_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Required columns missing in {input_file}: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            
            # If we're missing columns but have the right number, the names might be wrong
            # Let's try to map columns by position
            if len(df.columns) >= 15:  # Original expected column count
                print("Attempting to map columns by position...")
                # Map columns by their expected positions
                column_mapping = {
                    df.columns[0]: 'article_text',
                    df.columns[14]: 'propaganda_label'
                }
                df = df.rename(columns=column_mapping)
                print(f"Mapped columns: {column_mapping}")
            else:
                return False
        
        # Initialize counters for statistics
        total_rows = len(df)
        skipped_empty_text = 0
        skipped_invalid_propaganda = 0
        propaganda_count = 0
        
        # Pre-process labels to avoid repeated computation
        print("Pre-processing labels...")
        df['processed_propaganda_label'] = df['propaganda_label'].apply(process_propaganda_labels)
        
        # Count invalid labels
        invalid_propaganda = (df['processed_propaganda_label'] == -1).sum()
        empty_text = df['article_text'].isna().sum() + (df['article_text'] == '').sum()
        
        print(f"Statistics before processing:")
        print(f"  Total rows: {total_rows}")
        print(f"  Empty text: {empty_text} ({empty_text/total_rows*100:.1f}%)")
        print(f"  Invalid propaganda labels: {invalid_propaganda} ({invalid_propaganda/total_rows*100:.1f}%)")
        
        # Process propaganda labels
        print(f"Processing propaganda labels and writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Filter rows with valid text and propaganda labels
            valid_df = df[(df['processed_propaganda_label'] != -1) & (~df['article_text'].isna()) & (df['article_text'] != '')]
            
            for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Processing propaganda labels"):
                try:
                    # Create output object
                    output_obj = {
                        'text': row['article_text'],
                        'label': int(row['processed_propaganda_label'])  # Ensure it's an integer
                    }
                    
                    # Write to JSONL file
                    f.write(json.dumps(output_obj) + '\n')
                    propaganda_count += 1
                except Exception as e:
                    print(f"Error processing row for propaganda label: {e}")
                    continue
        
        # Print final statistics
        print(f"\nProcessing complete for {input_file}:")
        print(f"  Total rows processed: {total_rows}")
        print(f"  Valid entries: {propaganda_count} ({propaganda_count/total_rows*100:.1f}%)")
        
        return True
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_labels(file_path):
    """Count the distribution of labels in a processed file."""
    if not os.path.exists(file_path):
        return None
    
    # Initialize empty dictionary to store counts
    label_counts = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            label = data['label']
            
            # Add label to counts if not already present
            if label not in label_counts:
                label_counts[label] = 0
            
            # Increment count for this label
            label_counts[label] += 1
    
    return label_counts

def clean_up(keep_raw=False):
    """Clean up temporary files."""
    if keep_raw:
        print("\nKeeping raw data files as requested.")
        return
    
    print("\nCleaning up raw data files...")
    
    for split in DATASET_URLS.keys():
        file_path = os.path.join(DATA_DIR, f"proppy_1.0.{split}.tsv")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
    
    print("Raw data files deleted.")

def main():
    """Main function to process the Proppy dataset."""
    parser = argparse.ArgumentParser(description='Process Proppy dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw data files after processing')
    parser.add_argument('--quiet', action='store_true', help='Suppress warnings about unrecognized labels')
    args = parser.parse_args()
    
    # Set global flag for suppressing warnings
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS = args.quiet
    
    print("Starting Proppy dataset processing...")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download as requested.")
    
    # Step 2: Process each split
    for split in ["train", "dev", "test"]:
        input_file = os.path.join(DATA_DIR, f"proppy_1.0.{split}.tsv")
        
        # For validation split, use "valid" in the output filename
        output_split = "valid" if split == "dev" else split
        
        output_file = os.path.join(PROCESSED_DIR, f"{output_split}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} does not exist. Skipping.")
            continue
        
        if not process_file(input_file, output_file):
            print(f"Error processing {split} split. Skipping.")
            continue
    
    # Step 3: Count labels in processed files
    print("\n" + "="*80)
    print("Label Distribution in Processed Files")
    print("="*80)
    
    for split in ["train", "valid", "test"]:
        file_path = os.path.join(PROCESSED_DIR, f"{split}.jsonl")
        label_counts = count_labels(file_path)
        
        if label_counts:
            total = sum(label_counts.values())
            print(f"  {split.capitalize()} split ({total} examples):")
            
            # Get label descriptions
            label_descriptions = {0: "Not Propaganda", 1: "Propaganda"}
            
            # Sort labels and print with descriptions
            for label, count in sorted(label_counts.items()):
                percentage = (count / total) * 100
                description = label_descriptions.get(label, f"Unknown ({label})")
                print(f"    Label {label} ({description}): {count} examples ({percentage:.1f}%)")
        else:
            print(f"  {split.capitalize()} split: No data found")
    
    # Step 4: Clean up
    clean_up(args.keep_raw)
    
    print("\nProppy dataset processing complete!")
    print(f"Processed data saved to {PROCESSED_DIR}")

# Add global variable for suppressing warnings
SUPPRESS_WARNINGS = False

if __name__ == "__main__":
    main() 