#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import subprocess
import shutil
import pandas as pd
import requests
import zipfile
import io
import glob
import xml.etree.ElementTree as ET

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data")
BYARTICLE_DIR = os.path.join(PROCESSED_DIR, "byarticle")
BYPUBLISHER_DIR = os.path.join(PROCESSED_DIR, "bypublisher")

# Zenodo record URL
ZENODO_URL = "https://zenodo.org/records/1489920"
ZENODO_API_URL = "https://zenodo.org/api/records/1489920"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(BYARTICLE_DIR, exist_ok=True)
os.makedirs(BYPUBLISHER_DIR, exist_ok=True)

def download_dataset():
    """Download the dataset files from Zenodo."""
    print("\n" + "="*80)
    print("Downloading SemEval 2019 Task 4 Dataset")
    print("="*80)
    
    print(f"Fetching metadata from {ZENODO_API_URL}...")
    
    try:
        # Get the record metadata to find the download links
        response = requests.get(ZENODO_API_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        metadata = response.json()
        files = metadata.get('files', [])
        
        if not files:
            print("Error: No files found in the Zenodo record.")
            return False
        
        # Download each file
        for file_info in files:
            file_url = file_info.get('links', {}).get('self', '')
            filename = file_info.get('key', '')
            
            if not file_url or not filename:
                print(f"Warning: Incomplete file information for {filename}. Skipping.")
                continue
            
            file_path = os.path.join(DATA_DIR, filename)
            
            # Skip if file already exists
            if os.path.exists(file_path):
                print(f"File {filename} already exists. Skipping download.")
                continue
            
            print(f"Downloading {filename}...")
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {filename} to {file_path}")
            
            # If it's a zip file, extract it and delete the zip
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print(f"Extracted {filename}")
                
                # Delete the zip file
                os.remove(file_path)
                print(f"Deleted zip file {filename}")
        
        print("All files downloaded and extracted successfully.")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading files: {e}")
        return False
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def find_xml_files():
    """Find XML files in the data directory."""
    article_files = {}
    ground_truth_files = {}
    
    # Find all XML files
    xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    
    for xml_file in xml_files:
        filename = os.path.basename(xml_file)
        
        if "articles-training-byarticle" in filename:
            article_files["byarticle"] = xml_file
        elif "articles-training-bypublisher" in filename:
            article_files["bypublisher"] = xml_file
        elif "ground-truth-training-byarticle" in filename:
            ground_truth_files["byarticle"] = xml_file
        elif "ground-truth-training-bypublisher" in filename:
            ground_truth_files["bypublisher"] = xml_file
    
    return article_files, ground_truth_files

def get_article_text(article_elem):
    """Extract the full text content from an article element."""
    # Get the full text content of the article element
    if article_elem.text is None:
        article_text = ""
    else:
        article_text = article_elem.text.strip()
    
    # Append text from all child elements
    for child in article_elem:
        if child.text:
            article_text += " " + child.text.strip()
        if child.tail:
            article_text += " " + child.tail.strip()
    
    return article_text

def process_xml_to_jsonl(article_file, ground_truth_file, output_dir):
    """Process XML files directly to JSONL format."""
    print(f"Processing {article_file} and {ground_truth_file}...")
    
    # Parse ground truth file
    ground_truth = {}
    try:
        tree = ET.parse(ground_truth_file)
        root = tree.getroot()
        
        for article in root.findall(".//article"):
            article_id = article.get("id")
            hyperpartisan = 1 if article.get("hyperpartisan") == "true" else 0
            ground_truth[article_id] = hyperpartisan
    except Exception as e:
        print(f"Error parsing ground truth file: {e}")
        return False
    
    # Parse article file
    try:
        tree = ET.parse(article_file)
        root = tree.getroot()
        
        # Get all articles
        articles = []
        for article_elem in root.findall(".//article"):
            article_id = article_elem.get("id")
            
            if article_id in ground_truth:
                # Extract text content using the new function
                text = get_article_text(article_elem)
                
                # Create article object with only text and label
                article_obj = {
                    "text": text,
                    "label": ground_truth[article_id]
                }
                
                articles.append(article_obj)
        
        # Shuffle and split the data
        import random
        random.seed(42)
        random.shuffle(articles)
        
        # Split into train (80%), validation (10%), and test (10%)
        train_split = int(len(articles) * 0.8)
        valid_split = int(len(articles) * 0.9)
        
        train_articles = articles[:train_split]
        valid_articles = articles[train_split:valid_split]
        test_articles = articles[valid_split:]
        
        # Save to JSONL files
        train_file = os.path.join(output_dir, "train.jsonl")
        valid_file = os.path.join(output_dir, "valid.jsonl")
        test_file = os.path.join(output_dir, "test.jsonl")
        
        # Write train file
        with open(train_file, 'w', encoding='utf-8') as f:
            for article in train_articles:
                f.write(json.dumps(article) + '\n')
        
        # Write validation file
        with open(valid_file, 'w', encoding='utf-8') as f:
            for article in valid_articles:
                f.write(json.dumps(article) + '\n')
        
        # Write test file
        with open(test_file, 'w', encoding='utf-8') as f:
            for article in test_articles:
                f.write(json.dumps(article) + '\n')
        
        print(f"Created {train_file}, {valid_file}, and {test_file}")
        print(f"Train set: {len(train_articles)} examples")
        print(f"Validation set: {len(valid_articles)} examples")
        print(f"Test set: {len(test_articles)} examples")
        
        return True
    
    except Exception as e:
        print(f"Error processing article file: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_byarticle_data():
    """Process the byarticle dataset."""
    print("\n" + "="*80)
    print("Processing By-Article Dataset")
    print("="*80)
    
    # Find XML files
    article_files, ground_truth_files = find_xml_files()
    
    if "byarticle" not in article_files or "byarticle" not in ground_truth_files:
        print("Error: Required XML files for By-Article dataset not found.")
        return False
    
    # Process XML files directly to JSONL
    return process_xml_to_jsonl(
        article_files["byarticle"], 
        ground_truth_files["byarticle"], 
        BYARTICLE_DIR
    )

def process_bypublisher_data():
    """Process the bypublisher dataset."""
    print("\n" + "="*80)
    print("Processing By-Publisher Dataset")
    print("="*80)
    
    # Find XML files
    article_files, ground_truth_files = find_xml_files()
    
    if "bypublisher" not in article_files or "bypublisher" not in ground_truth_files:
        print("Error: Required XML files for By-Publisher dataset not found.")
        return False
    
    # Process XML files directly to JSONL
    return process_xml_to_jsonl(
        article_files["bypublisher"], 
        ground_truth_files["bypublisher"], 
        BYPUBLISHER_DIR
    )

def clean_up():
    """Clean up temporary files."""
    print("\nCleaning up raw data files...")
    
    # Delete all files in the data directory
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {filename}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
    
    print("Raw data files deleted.")

def main():
    """Main function to download and process the SemEval 2019 Task 4 dataset."""
    parser = argparse.ArgumentParser(description='Download and process SemEval 2019 Task 4 dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--keep-raw', action='store_true', help='Keep raw data files after processing')
    parser.add_argument('--yes', '-y', action='store_true', help='Answer yes to all prompts')
    args = parser.parse_args()
    
    print("Starting SemEval 2019 Task 4 dataset download and processing...")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        if not download_dataset():
            print("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download as requested.")
    
    # Step 2: Process the byarticle dataset
    if not process_byarticle_data():
        print("Failed to process By-Article dataset.")
    
    # Step 3: Process the bypublisher dataset
    if not process_bypublisher_data():
        print("Failed to process By-Publisher dataset.")
    
    # Step 4: Clean up if requested
    if not args.keep_raw:
        clean_up()
    else:
        print("\nKeeping raw data files as requested.")
    
    print("\n" + "="*80)
    print("Download and processing complete!")
    print(f"Processed data files are available in:")
    print(f"  - By-Article: {BYARTICLE_DIR}")
    print(f"  - By-Publisher: {BYPUBLISHER_DIR}")
    print("="*80)

if __name__ == "__main__":
    main() 