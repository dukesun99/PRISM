# MITweet Dataset Processing

This directory contains a script to download and process the MITweet dataset, which provides multifaceted ideology labels for Twitter posts.

## Processing Script

The `process_data.py` script automatically handles both downloading and processing of the MITweet dataset in a single step.

### How to Run

```bash
# Run the script (will delete temporary files after processing)
python process_data.py

# Keep temporary files after processing
python process_data.py --keep-temp
```

### Process

1. The script will automatically clone the MITweet repository from GitHub
2. It will extract the random_split data files (train.csv, val.csv, test.csv, Indicators.txt)
3. The script will process the data, extracting text and all original labels
4. The processed data will be saved as JSONL files in the `processed_data/` directory
5. By default, the script will delete temporary files and the cloned repository to save disk space

## Data Format

The processed data will be in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The original tweet text
- `topic`: The topic of the tweet
- `label`: A dictionary containing all numeric labels from the dataset, including:
  - Domain relevance labels (R1, R2, R3, R4, R5)
  - Facet relevance labels (R1-1-1, R2-1-2, R3-2-1, etc.)
  - Ideology labels (I1, I2, I3, ..., I12)

Example:
```json
{
  "text": "Tweet text here...",
  "topic": "Abortion",
  "label": {
    "R1": 0,
    "R2": 1,
    "R3": 0,
    "R4": 0,
    "R5": 1,
    "R1-1-1": 0,
    "R2-1-2": 1,
    "R3-2-1": 0,
    "I1": -1,
    "I2": -1,
    "I3": 0,
    "I10": 0,
    "I11": 1,
    "I12": -1
  }
}
```

Note: According to the MITweet paper, the dataset contains:
- Domain relevance labels (R1-R5): 1 means "Related", 0 means "Unrelated"
- Facet relevance labels (R1-1-1, etc.): 1 means "Related", 0 means "Unrelated"
- Ideology labels (I1-I12): 0 means left-leaning, 1 means center, 2 means right-leaning, -1 means "Unrelated"

## Requirements

- Python 3.6+
- Git
- pandas
- numpy

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `train.jsonl`: Training data
  - `valid.jsonl`: Validation data
  - `test.jsonl`: Test data

Note: By default, temporary files and the cloned repository will be deleted after processing to save disk space. Use the `--keep-temp` flag if you want to keep them.