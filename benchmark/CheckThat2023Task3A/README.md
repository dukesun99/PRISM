# CheckThat! 2023 Task 3A Dataset Processing

This directory contains a script to download and process the CheckThat! 2023 Task 3A dataset for political bias detection in news articles.

## Dataset Description

The CheckThat! 2023 Task 3A dataset contains news articles labeled for political bias. This is an ordinal classification task where articles are classified as:
- Left-leaning (label 0)
- Center (label 1)
- Right-leaning (label 2)

The dataset is part of the CLEF 2023 CheckThat! Lab, which focuses on detecting political bias in news articles.

## Processing Script

The `process_data.py` script handles downloading, extracting, and processing the dataset from its original format (TSV files pointing to JSON files) to a simplified JSONL format.

### How to Run

```bash
# Run the script (will download, extract, and process the dataset)
python process_data.py

# Skip downloading if you already have the data
python process_data.py --skip-download

# Keep raw data files after processing
python process_data.py --keep-raw
```

### Process

1. The script will download the dataset from the CheckThat! Lab GitLab repository
2. It will extract the zip file and locate the Task 3A data
3. The script reads the TSV files that contain paths to JSON files and their corresponding labels
4. For each entry, it reads the JSON file which contains the article title and content
5. It combines the title and content into a single text field
6. The processed data is saved in JSONL format with two fields: text and label
7. The script creates three splits:
   - Train: Original training data
   - Valid: Original development data
   - Test: Copy of the development data (since test data is not publicly available)
8. By default, the script will delete the raw data files to save space (use `--keep-raw` to keep them)

## Data Format

The processed data will be in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text (title + content)
- `label`: The political bias label (0 for left-leaning, 1 for center, 2 for right-leaning)

## Requirements

- Python 3.6+
- pandas
- tqdm
- requests

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `train.jsonl`: Training data (~45,000 examples)
  - `valid.jsonl`: Validation data (~5,000 examples)
  - `test.jsonl`: Test data (copy of validation data)

## Label Distribution

The dataset has the following label distribution:

| Split | Left (0) | Center (1) | Right (2) | Total |
|-------|----------|------------|-----------|-------|
| Train | 12,073   | 15,449     | 17,544    | 45,066|
| Valid | 1,342    | 1,717      | 1,949     | 5,008 |

## Data Source

The dataset is available from the CheckThat! Lab: [https://gitlab.com/checkthat_lab/clef2023-checkthat-lab](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab) 