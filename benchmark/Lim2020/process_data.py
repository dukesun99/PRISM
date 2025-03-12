#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import shutil
import random
import requests
from tqdm import tqdm
import re

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")

# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/skymoonlight/biased-sents-annotation/refs/heads/master/Sora_LREC2020_biasedsentences.csv"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define events and their corresponding IDs
EVENTS = {
    "Johnson": "1",
    "Facebook": "2",
    "NFL": "3",
    "NorthKora": "4"
}

# Define bias labels mapping
BIAS_LABELS = {
    "1": 0,  # neutral
    "2": 1,  # slightly biased but acceptable
    "3": 2,  # biased
    "4": 3   # very biased
}

def download_dataset():
    """Download the Lim2020 dataset from GitHub."""
    print("\n" + "="*80)
    print("Downloading Lim2020 Dataset")
    print("="*80)
    
    output_file = os.path.join(DATA_DIR, "biased_sentences.csv")
    
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}. Skipping download.")
        return True
    
    print(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded dataset to {output_file}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return False

def clean_text(text):
    """Remove [n]: tags from the beginning of the text."""
    return re.sub(r'^\[\d+\]:\s*', '', text)

def process_dataset(seed=42):
    """Process the dataset and split it into train/valid/test sets."""
    print("\nProcessing Lim2020 dataset...")
    
    input_file = os.path.join(DATA_DIR, "biased_sentences.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return False
    
    try:
        # Read the CSV file
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Find the indices of relevant columns
            id_event_idx = header.index("id_event") if "id_event" in header else None
            event_idx = header.index("event") if "event" in header else None
            date_event_idx = header.index("date_event") if "date_event" in header else None
            id_article_idx = header.index("id_article") if "id_article" in header else None
            source_idx = header.index("source") if "source" in header else None
            source_bias_idx = header.index("source_bias") if "source_bias" in header else None
            loc_idx = header.index("loc") if "loc" in header else None
            preknow_idx = header.index("preknow") if "preknow" in header else None
            
            # Find sentence bias columns (t, 0, 1, 2, etc.)
            bias_columns = []
            for i, col in enumerate(header):
                if col == "t" or col.isdigit():
                    bias_columns.append(i)
            
            # Find sentence text columns (s0, s1, s2, etc.)
            sentence_columns = []
            for i, col in enumerate(header):
                if col.startswith("s") and col[1:].isdigit():
                    sentence_columns.append(i)
            
            if not bias_columns or not sentence_columns:
                print("Error: Could not find bias or sentence columns in the CSV file.")
                return False
            
            # Dictionary to store sentences with their annotations
            # Key: (event, article_id, sentence_position)
            # Value: list of (bias_label, annotator_info)
            sentence_annotations = {}
            
            # Process each row and extract sentences with their bias labels
            for row_idx, row in enumerate(tqdm(reader, desc="Processing rows")):
                event = row[event_idx] if event_idx is not None else ""
                date = row[date_event_idx] if date_event_idx is not None else ""
                article_id = row[id_article_idx] if id_article_idx is not None else ""
                source = row[source_idx] if source_idx is not None else ""
                source_bias = row[source_bias_idx] if source_bias_idx is not None else ""
                preknow = row[preknow_idx] if preknow_idx is not None else ""
                
                # Process each sentence in the row
                for sent_idx in sentence_columns:
                    sentence_text = row[sent_idx]
                    if not sentence_text or sentence_text.strip() == "":
                        continue
                    
                    # Get the sentence position (s0, s1, s2, etc.)
                    sent_pos = header[sent_idx][1:] if header[sent_idx].startswith("s") else ""
                    
                    # Find the corresponding bias label column
                    bias_idx = None
                    for col_idx in bias_columns:
                        if header[col_idx] == sent_pos or (header[col_idx] == "t" and sent_pos == "0"):
                            bias_idx = col_idx
                            break
                    
                    if bias_idx is None:
                        continue
                    
                    # Get the bias label
                    bias_label = row[bias_idx]
                    if not bias_label or bias_label not in BIAS_LABELS:
                        continue
                    
                    # Create a unique key for this sentence
                    sentence_key = (event, article_id, sent_pos, sentence_text)
                    
                    # Add this annotation to the sentence
                    if sentence_key not in sentence_annotations:
                        sentence_annotations[sentence_key] = []
                    
                    # Add the annotation with annotator info
                    sentence_annotations[sentence_key].append({
                        "label": BIAS_LABELS[bias_label],
                        "annotator_row": row_idx,
                        "preknow": preknow
                    })
            
            print(f"Found {len(sentence_annotations)} unique sentences with {sum(len(anns) for anns in sentence_annotations.values())} total annotations.")
            
            # Analyze annotations per sentence
            annotation_counts = {}
            for key, annotations in sentence_annotations.items():
                count = len(annotations)
                if count not in annotation_counts:
                    annotation_counts[count] = 0
                annotation_counts[count] += 1
            
            print("Annotations per sentence:")
            for count, num_sentences in sorted(annotation_counts.items()):
                print(f"  {count} annotations: {num_sentences} sentences")
            
            # Process sentences with multiple annotations
            sentences = []
            for (event, article_id, sent_pos, sentence_text), annotations in sentence_annotations.items():
                # Aggregate labels from multiple annotators
                if len(annotations) > 1:
                    # Count occurrences of each label
                    label_counts = {}
                    for ann in annotations:
                        label = ann["label"]
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                    
                    # Use the most common label (majority vote)
                    final_label = max(label_counts.items(), key=lambda x: x[1])[0]
                    
                    # In case of a tie, use the higher bias level
                    if len(set(label_counts.values())) < len(label_counts):
                        tied_labels = [l for l, c in label_counts.items() if c == max(label_counts.values())]
                        if len(tied_labels) > 1:
                            final_label = max(tied_labels)
                else:
                    # Only one annotation, use it directly
                    final_label = annotations[0]["label"]
                
                # Clean the sentence text
                cleaned_text = clean_text(sentence_text)
                
                # Create a sentence object
                sentence_obj = {
                    "text": cleaned_text,
                    "label": final_label,
                    "event": event,
                    "date": date,
                    "article_id": article_id,
                    "source": source,
                    "source_bias": source_bias,
                    "position": sent_pos,
                    "num_annotations": len(annotations)
                }
                
                sentences.append(sentence_obj)
            
            print(f"Processed {len(sentences)} unique sentences with aggregated labels.")
            
            # Split the data by event
            event_sentences = {}
            for event_name in EVENTS.keys():
                event_sentences[event_name] = [s for s in sentences if s["event"] == event_name]
                print(f"Event {event_name}: {len(event_sentences[event_name])} sentences")
            
            # Create directories for each event
            for event_name in EVENTS.keys():
                os.makedirs(os.path.join(PROCESSED_DIR, event_name.lower()), exist_ok=True)
            
            # Process each event separately
            for event_name, event_data in event_sentences.items():
                if not event_data:
                    continue
                
                # Shuffle the data
                random.seed(seed)
                random.shuffle(event_data)
                
                # Split into train (80%), valid (10%), test (10%)
                total = len(event_data)
                train_size = int(0.8 * total)
                valid_size = int(0.1 * total)
                
                train_data = event_data[:train_size]
                valid_data = event_data[train_size:train_size + valid_size]
                test_data = event_data[train_size + valid_size:]
                
                # Save the splits
                event_dir = os.path.join(PROCESSED_DIR, event_name.lower())
                
                for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
                    output_file = os.path.join(event_dir, f"{split_name}.jsonl")
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for item in split_data:
                            f.write(json.dumps(item) + "\n")
                    
                    print(f"Saved {len(split_data)} sentences to {output_file}")
            
            # Also create a combined dataset with all events
            os.makedirs(os.path.join(PROCESSED_DIR, "combined"), exist_ok=True)
            
            # Shuffle all sentences
            random.seed(seed)
            random.shuffle(sentences)
            
            # Split into train (80%), valid (10%), test (10%)
            total = len(sentences)
            train_size = int(0.8 * total)
            valid_size = int(0.1 * total)
            
            train_data = sentences[:train_size]
            valid_data = sentences[train_size:train_size + valid_size]
            test_data = sentences[train_size + valid_size:]
            
            # Save the combined splits
            combined_dir = os.path.join(PROCESSED_DIR, "combined")
            
            for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
                output_file = os.path.join(combined_dir, f"{split_name}.jsonl")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item) + "\n")
                
                print(f"Saved {len(split_data)} sentences to {output_file}")
            
            return True
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_labels():
    """Count the distribution of labels in the processed files."""
    print("\n" + "="*80)
    print("Label Distribution in Processed Files")
    print("="*80)
    
    # Define label descriptions
    label_descriptions = {
        0: "Neutral",
        1: "Slightly Biased",
        2: "Biased",
        3: "Very Biased"
    }
    
    # Count labels for each event and split
    for event_name in list(EVENTS.keys()) + ["combined"]:
        event_dir = os.path.join(PROCESSED_DIR, event_name.lower())
        
        if not os.path.exists(event_dir):
            continue
        
        print(f"\n{event_name} Dataset:")
        
        for split in ["train", "valid", "test"]:
            file_path = os.path.join(event_dir, f"{split}.jsonl")
            
            if not os.path.exists(file_path):
                print(f"  {split.capitalize()} split: File not found")
                continue
            
            label_counts = {}
            total = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
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
                description = label_descriptions.get(int(label), f"Unknown ({label})")
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
    """Main function to process the Lim2020 dataset."""
    parser = argparse.ArgumentParser(description="Process Lim2020 dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading the dataset")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw data files after processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting (default: 42)")
    args = parser.parse_args()
    
    print("Starting Lim2020 dataset processing...")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download as requested.")
    
    # Step 2: Process the dataset
    if not process_dataset(seed=args.seed):
        print("Failed to process dataset. Exiting.")
        sys.exit(1)
    
    # Step 3: Count labels
    count_labels()
    
    # Step 4: Clean up
    clean_up(args.keep_raw)
    
    print("\nLim2020 dataset processing complete!")
    print(f"Processed data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main() 