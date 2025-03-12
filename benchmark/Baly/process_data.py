import os
import json
import csv
import subprocess
import sys

# Define paths using relative paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(CURRENT_DIR, 'Article-Bias-Prediction')
DATA_DIR = os.path.join(REPO_DIR, 'data')
JSON_DIR = os.path.join(DATA_DIR, 'jsons')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits', 'media')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'processed_data')

def check_and_clone_repo():
    """Check if the repository exists, if not clone it."""
    if not os.path.exists(REPO_DIR):
        print(f"Article-Bias-Prediction repository not found at {REPO_DIR}")
        print("Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ramybaly/Article-Bias-Prediction.git", REPO_DIR],
                check=True
            )
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)
    
    # Check if data directory exists after cloning
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found at {DATA_DIR} even after cloning.")
        print("Please check the repository structure.")
        sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_split_data(split_name):
    """Load the IDs and labels from a split file."""
    split_file = os.path.join(SPLITS_DIR, f"{split_name}.tsv")
    
    id_label_pairs = []
    with open(split_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                article_id = row[0]
                label = row[1]
                id_label_pairs.append((article_id, label))
    
    return id_label_pairs

def process_split(split_name):
    """Process a split and save the (text, label) pairs."""
    print(f"Processing {split_name} split...")
    
    # Load split data
    id_label_pairs = load_split_data(split_name)
    
    # Process each article
    total = len(id_label_pairs)
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (article_id, label) in enumerate(id_label_pairs):
            if i % 100 == 0:
                print(f"Processing {i}/{total}...")
            
            # Load the JSON file
            json_path = os.path.join(JSON_DIR, f"{article_id}.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    article_data = json.load(json_file)
                
                # Get the article text (using content_original if available, otherwise content)
                text = article_data.get('content_original', article_data.get('content', ''))
                
                # Create a simple JSON object with text and label
                output_obj = {
                    'text': text,
                    'label': label
                }
                
                # Write as a JSON line
                f.write(json.dumps(output_obj) + '\n')
            except Exception as e:
                print(f"Error processing {article_id}: {e}")
    
    print(f"Saved {total} examples to {output_path}")

def main():
    """Process all splits."""
    # Check if repository exists and clone if needed
    check_and_clone_repo()
    
    # Process each split
    for split_name in ['train', 'valid', 'test']:
        process_split(split_name)
    
    print("All data processing complete!")

if __name__ == "__main__":
    main() 