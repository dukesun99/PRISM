#!/usr/bin/env python3
import os
import sys
import json
import pickle
import argparse
import shutil
import random
import requests
import gdown
from tqdm import tqdm

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Define subdatasets and their Google Drive URLs
SUBDATASETS = {
    "abortion": "https://drive.google.com/drive/folders/1g9P6vHTJ1nNec1zrBZggcNYu_Lmm95Oa?usp=sharing",
    "immigration": "https://drive.google.com/drive/folders/1OAND83Jtng46WuVKMZx0dpO2o3UtbuUD?usp=sharing",
    "gun_control": "https://drive.google.com/drive/folders/1nJ-kzZeIJkUzwRTjFAOAPTbMpd64N7L_?usp=sharing"
}

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_dataset():
    """Download the Roy2020 dataset from Google Drive."""
    print("\n" + "="*80)
    print("Downloading Roy2020 Dataset")
    print("="*80)
    
    for subdataset, url in SUBDATASETS.items():
        subdataset_dir = os.path.join(DATA_DIR, subdataset)
        os.makedirs(subdataset_dir, exist_ok=True)
        
        output_file = os.path.join(subdataset_dir, "article_data.pkl")
        
        if os.path.exists(output_file):
            print(f"File already exists: {output_file}. Skipping download.")
            continue
        
        print(f"Downloading {subdataset} dataset from {url}...")
        
        try:
            # Extract folder ID from URL
            folder_id = url.split("folders/")[1].split("?")[0]
            
            # Use gdown to download the file from the folder
            gdown.download_folder(url, output=subdataset_dir, quiet=False, use_cookies=False)
            
            # Check if the file was downloaded
            if os.path.exists(output_file):
                print(f"Downloaded {subdataset} dataset to {output_file}")
            else:
                print(f"Error: Could not find article_data.pkl in the downloaded folder for {subdataset}")
                return False
        
        except Exception as e:
            print(f"Error downloading {subdataset} dataset: {e}")
            return False
    
    return True

def process_subdataset(subdataset, seed=42):
    """Process a subdataset and convert it to JSONL format with train/valid/test splits."""
    print(f"\nProcessing {subdataset} dataset...")
    
    input_file = os.path.join(DATA_DIR, subdataset, "article_data.pkl")
    
    # Create output directory for this subdataset
    output_dir = os.path.join(PROCESSED_DIR, subdataset)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return False
    
    try:
        # Load the pickle file
        with open(input_file, "rb") as in_file:
            [article2URL, article2dop, article2headline, article2text, art2label, article2segment_ids, seg_id2text] = pickle.load(in_file)
        
        print(f"Loaded {len(article2text)} articles from {input_file}")
        
        # Process articles and prepare for splitting
        processed_articles = []
        skipped_count = 0
        
        for article_id, text in tqdm(article2text.items(), desc=f"Processing {subdataset}"):
            try:
                # Skip articles with missing text or label
                if not text or article_id not in art2label:
                    skipped_count += 1
                    continue
                
                label = art2label[article_id].lower()
                
                # Convert label to numeric format
                if label == "left":
                    numeric_label = 0
                elif label == "right":
                    numeric_label = 1
                else:
                    print(f"Warning: Unknown label '{label}' for article {article_id}. Skipping.")
                    skipped_count += 1
                    continue
                
                # Create output object
                output_obj = {
                    "text": text,
                    "label": numeric_label,
                    "topic": subdataset
                }
                
                # Add metadata if available
                if article_id in article2headline:
                    output_obj["title"] = article2headline[article_id]
                if article_id in article2URL:
                    output_obj["url"] = article2URL[article_id]
                if article_id in article2dop:
                    output_obj["date"] = article2dop[article_id]
                
                processed_articles.append(output_obj)
            
            except Exception as e:
                print(f"Error processing article {article_id}: {e}")
                skipped_count += 1
        
        # Split the data into train, valid, and test sets
        random.seed(seed)
        random.shuffle(processed_articles)
        
        total_articles = len(processed_articles)
        train_size = int(0.8 * total_articles)
        valid_size = int(0.1 * total_articles)
        
        train_data = processed_articles[:train_size]
        valid_data = processed_articles[train_size:train_size + valid_size]
        test_data = processed_articles[train_size + valid_size:]
        
        # Save the splits
        splits = {
            "train": train_data,
            "valid": valid_data,
            "test": test_data
        }
        
        # Save each split
        for split_name, split_data in splits.items():
            output_file = os.path.join(output_dir, f"{split_name}.jsonl")
            
            with open(output_file, "w", encoding="utf-8") as f:
                for article in split_data:
                    f.write(json.dumps(article) + "\n")
            
            print(f"Saved {len(split_data)} articles to {output_file}")
        
        print(f"Processed {len(processed_articles)} articles, skipped {skipped_count} articles.")
        print(f"Split sizes: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {subdataset} dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_labels():
    """Count the distribution of labels in the processed files."""
    print("\n" + "="*80)
    print("Label Distribution in Processed Files")
    print("="*80)
    
    # Define label descriptions
    label_descriptions = {0: "Left", 1: "Right"}
    
    # Count labels for each subdataset and split
    for subdataset in SUBDATASETS.keys():
        print(f"\n{subdataset.capitalize()} Dataset:")
        
        # Count for each split
        for split in ["train", "valid", "test"]:
            file_path = os.path.join(PROCESSED_DIR, subdataset, f"{split}.jsonl")
            
            if not os.path.exists(file_path):
                print(f"  {split.capitalize()} split: File not found")
                continue
            
            label_counts = {}
            total = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    label = data["label"]
                    
                    if label not in label_counts:
                        label_counts[label] = 0
                    
                    label_counts[label] += 1
                    total += 1
            
            print(f"  {split.capitalize()} split ({total} examples):")
            
            for label, count in sorted(label_counts.items()):
                percentage = (count / total) * 100 if total > 0 else 0
                description = label_descriptions.get(label, f"Unknown ({label})")
                print(f"    Label {label} ({description}): {count} examples ({percentage:.1f}%)")

def clean_up(keep_raw=False):
    """Clean up temporary files."""
    if keep_raw:
        print("\nKeeping raw data files as requested.")
        return
    
    print("\nCleaning up raw data files...")
    
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Deleted {DATA_DIR}")
    
    print("Raw data files deleted.")

def main():
    """Main function to process the Roy2020 dataset."""
    parser = argparse.ArgumentParser(description="Process Roy2020 dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading the dataset")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw data files after processing")
    parser.add_argument("--subdataset", choices=list(SUBDATASETS.keys()) + ["all"], default="all",
                        help="Process only a specific subdataset (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting (default: 42)")
    args = parser.parse_args()
    
    print("Starting Roy2020 dataset processing...")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download as requested.")
    
    # Step 2: Process subdatasets
    if args.subdataset == "all":
        subdatasets_to_process = SUBDATASETS.keys()
    else:
        subdatasets_to_process = [args.subdataset]
    
    for subdataset in subdatasets_to_process:
        if not process_subdataset(subdataset, seed=args.seed):
            print(f"Error processing {subdataset} subdataset. Skipping.")
    
    # Step 3: Count labels
    count_labels()
    
    # Step 4: Clean up
    clean_up(args.keep_raw)
    
    print("\nRoy2020 dataset processing complete!")
    print(f"Processed data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main() 