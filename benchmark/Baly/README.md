# Baly Article Bias Prediction Data Processing

This directory contains scripts to process the Article Bias Prediction dataset from the Baly repository.

## Data Structure

The original data is organized as follows:
- JSON files in `Article-Bias-Prediction/data/jsons/`
- Media split files in `Article-Bias-Prediction/data/splits/media/`

Each JSON file contains an article with its content and metadata. The media split files (train.tsv, valid.tsv, test.tsv) contain the article IDs and their bias labels.

## Processing Script

The `process_data.py` script reads the data according to the media split and organizes it into (text, label) pairs. The processed data is saved as JSONL files in the `processed_data` directory.

The script will:
1. Check if the `Article-Bias-Prediction` repository exists, and clone it from GitHub if not
2. Process the data according to the media split
3. Save the processed data as JSONL files

### How to Run

```bash
# Run the processing script
python process_data.py
```

### Output

The script will create three JSONL files in the `processed_data` directory:
- `train.jsonl`: Training data with text and label fields (26,590 examples)
- `valid.jsonl`: Validation data with text and label fields (2,356 examples)
- `test.jsonl`: Test data with text and label fields (1,300 examples)

Each line in the JSONL files is a valid JSON object with the following structure:
```json
{
    "text": "The article text content...",
    "label": "0"  // or "1" or "2"
}
```

### Label Mapping

The bias labels are as follows:
- 0: Left-leaning
- 1: Center
- 2: Right-leaning

The train set has the following label distribution:
- Left-leaning (0): 8,861 examples
- Center (1): 7,488 examples
- Right-leaning (2): 10,241 examples