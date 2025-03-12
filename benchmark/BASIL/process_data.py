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
REPO_URL = "https://github.com/launchnlp/BASIL.git"
REPO_DIR = os.path.join(CURRENT_DIR, "BASIL_repo")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define stance mapping
STANCE_MAPPING = {
    "conservative": 2,
    "right": 2,
    "center": 1,
    "liberal": 0,
    "left": 0
}

def clone_repository():
    """Clone the BASIL repository."""
    print("\n" + "="*80)
    print("Cloning BASIL Repository")
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

def load_articles():
    """Load all articles from the repository."""
    print("\n" + "="*80)
    print("Loading Articles")
    print("="*80)
    
    articles_dir = os.path.join(REPO_DIR, "articles")
    if not os.path.exists(articles_dir):
        print(f"Articles directory not found at {articles_dir}.")
        return None
    
    articles = {}
    
    # Iterate through year directories
    for year_dir in os.listdir(articles_dir):
        year_path = os.path.join(articles_dir, year_dir)
        if not os.path.isdir(year_path):
            continue
        
        print(f"Processing articles from {year_dir}...")
        
        # Iterate through JSON files in the year directory
        for json_file in tqdm(os.listdir(year_path)):
            if not json_file.endswith('.json'):
                continue
            
            json_path = os.path.join(year_path, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                
                # Extract UUID and text
                uuid = article_data.get('uuid')
                if not uuid:
                    print(f"Warning: No UUID found in {json_path}. Skipping.")
                    continue
                
                # Extract body paragraphs and concatenate text
                body_paragraphs = article_data.get('body-paragraphs', [])
                text = ""
                for paragraph in body_paragraphs:
                    if isinstance(paragraph, list):
                        # Flatten the list of strings
                        text += " ".join([str(s) for s in paragraph if s])
                    else:
                        # Handle case where paragraph might not be a list
                        text += str(paragraph)
                    text += " "
                
                # Store article data
                articles[uuid] = {
                    'text': text.strip(),
                    'year': year_dir,
                    'file': json_file
                }
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
    
    print(f"Loaded {len(articles)} articles.")
    return articles

def load_annotations():
    """Load all annotations from the repository."""
    print("\n" + "="*80)
    print("Loading Annotations")
    print("="*80)
    
    annotations_dir = os.path.join(REPO_DIR, "annotations")
    if not os.path.exists(annotations_dir):
        print(f"Annotations directory not found at {annotations_dir}.")
        return None
    
    annotations = {}
    
    # Iterate through year directories
    for year_dir in os.listdir(annotations_dir):
        year_path = os.path.join(annotations_dir, year_dir)
        if not os.path.isdir(year_path):
            continue
        
        print(f"Processing annotations from {year_dir}...")
        
        # Iterate through JSON files in the year directory
        for json_file in tqdm(os.listdir(year_path)):
            if not json_file.endswith('.json'):
                continue
            
            json_path = os.path.join(year_path, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                # Extract UUID and stance
                uuid = annotation_data.get('uuid')
                if not uuid:
                    print(f"Warning: No UUID found in {json_path}. Skipping.")
                    continue
                
                # Extract article-level annotations
                article_level = annotation_data.get('article-level-annotations', {})
                stance = article_level.get('relative_stance')
                
                if not stance:
                    print(f"Warning: No stance found in {json_path}. Skipping.")
                    continue
                
                # Store annotation data
                annotations[uuid] = {
                    'stance': stance,
                    'year': year_dir,
                    'file': json_file
                }
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
    
    print(f"Loaded {len(annotations)} annotations.")
    return annotations

def merge_data(articles, annotations):
    """Merge articles and annotations based on UUID."""
    print("\n" + "="*80)
    print("Merging Articles and Annotations")
    print("="*80)
    
    merged_data = []
    
    for uuid, article in articles.items():
        if uuid in annotations:
            stance = annotations[uuid]['stance']
            
            # Map stance to numeric label
            label = STANCE_MAPPING.get(stance, -1)
            
            if label == -1:
                print(f"Warning: Unknown stance '{stance}' for UUID {uuid}. Skipping.")
                continue
            
            merged_data.append({
                'uuid': uuid,
                'text': article['text'],
                'label': label,
                'year': article['year']
            })
    
    print(f"Merged {len(merged_data)} articles with annotations.")
    return merged_data

def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """Split data into train, validation, and test sets."""
    print("\n" + "="*80)
    print("Splitting Data into Train, Validation, and Test Sets")
    print("="*80)
    
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
    
    print(f"Split data into:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(valid_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    return train_data, valid_data, test_data

def save_data(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Create output object with only text and label
            output_obj = {
                'text': item['text'],
                'label': item['label']
            }
            f.write(json.dumps(output_obj) + '\n')
    
    print(f"Saved {len(data)} examples to {file_path}")

def count_labels(data):
    """Count the distribution of labels in the data."""
    label_counts = {0: 0, 1: 0, 2: 0}  # liberal, center, conservative
    
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

def main():
    """Main function to process the BASIL dataset."""
    parser = argparse.ArgumentParser(description='Process BASIL dataset')
    parser.add_argument('--keep-repo', action='store_true', help='Keep repository after processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    args = parser.parse_args()
    
    print("Starting BASIL dataset processing...")
    
    # Step 1: Clone the repository
    if not clone_repository():
        print("Failed to clone repository. Exiting.")
        sys.exit(1)
    
    # Step 2: Load articles
    articles = load_articles()
    if articles is None:
        print("Failed to load articles. Exiting.")
        sys.exit(1)
    
    # Step 3: Load annotations
    annotations = load_annotations()
    if annotations is None:
        print("Failed to load annotations. Exiting.")
        sys.exit(1)
    
    # Step 4: Merge data
    merged_data = merge_data(articles, annotations)
    if not merged_data:
        print("Failed to merge data. Exiting.")
        sys.exit(1)
    
    # Step 5: Split data
    train_data, valid_data, test_data = split_data(merged_data, seed=args.seed)
    
    # Step 6: Save data
    save_data(train_data, os.path.join(PROCESSED_DIR, "train.jsonl"))
    save_data(valid_data, os.path.join(PROCESSED_DIR, "valid.jsonl"))
    save_data(test_data, os.path.join(PROCESSED_DIR, "test.jsonl"))
    
    # Step 7: Print label distribution
    print("\n" + "="*80)
    print("Label Distribution")
    print("="*80)
    
    train_counts = count_labels(train_data)
    valid_counts = count_labels(valid_data)
    test_counts = count_labels(test_data)
    
    print(f"Train: Liberal={train_counts[0]}, Center={train_counts[1]}, Conservative={train_counts[2]}")
    print(f"Valid: Liberal={valid_counts[0]}, Center={valid_counts[1]}, Conservative={valid_counts[2]}")
    print(f"Test: Liberal={test_counts[0]}, Center={test_counts[1]}, Conservative={test_counts[2]}")
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Processed data files are available in: {PROCESSED_DIR}")
    print("="*80)
    
    # Step 8: Clean up
    clean_up(args.keep_repo)

if __name__ == "__main__":
    main() 