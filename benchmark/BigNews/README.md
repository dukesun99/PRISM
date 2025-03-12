# BIGNEWSBLN Dataset Processing

This directory contains scripts to download and process the BIGNEWSBLN dataset.

## Download Script

The `download_data.py` script is an interactive tool that guides you through the process of obtaining and processing the BIGNEWSBLN dataset.

### How to Run

```bash
# Run the download script
python download_data.py
```

### Process

1. The script will open a Google Form in your web browser
2. Fill out the form, making sure to select the **BIGNEWSBLN** dataset option
3. Copy the link and paste it back into the script when prompted
4. The script will extract the folder ID from the link
5. All files from the Google Drive folder will be downloaded to the `data/` directory
6. The script will verify that all required files are present:
   - BIGNEWSBLN_center.json
   - BIGNEWSBLN_left.json
   - BIGNEWSBLN_right.json

## Processing Script

After downloading the data, you can use the `process_data.py` script to process the BIGNEWSBLN dataset into a format similar to the Baly dataset.

### How to Run

```bash
# Run the processing script (will delete source files after processing)
python process_data.py

# Run with a specific random seed
python process_data.py --seed 42

# Keep the source files after processing
python process_data.py --keep-source
```

### Process

1. The script will find and load the three JSON files (left, center, right)
2. It will extract the text from each article and assign the appropriate label
3. If the text is stored as a list of sentences, it will be joined into a single string
4. The data will be randomly split into train (80%), validation (10%), and test (10%) sets
5. The processed data will be saved as JSONL files in the `processed_data` directory
6. By default, the script will delete the source data files to save disk space

## Data Format

The processed data will be in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text as a single string (sentences from the original list are joined with spaces)
- `label`: The political bias label (0 for left-leaning, 1 for center, 2 for right-leaning)

## Requirements

- Python 3.6+
- Internet connection
- Web browser
- gdown package (included in project requirements.txt)

## Data Structure

After running both scripts, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `train.jsonl`: Training data (80% of the dataset)
  - `valid.jsonl`: Validation data (10% of the dataset)
  - `test.jsonl`: Test data (10% of the dataset)

Note: By default, the source data files in the `data/` directory will be deleted after processing to save disk space. Use the `--keep-source` flag if you want to keep them.

## Note

The BIGNEWSBLN dataset requires filling out a form before access is granted. This script automates the download process as much as possible, but you will still need to manually fill out the form and select the correct dataset option. 