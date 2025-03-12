# BABE Dataset Processing

This directory contains a script to download and process the BABE (Bias Annotation Benchmark for Event Detection) dataset.

## Dataset Description

The BABE dataset contains news articles annotated for media bias. It includes articles from various news sources, annotated for both bias and opinion. The dataset was created using distant supervision techniques.

The dataset consists of three separate collections:
- **SG1**: First collection of articles
- **SG2**: Second collection of articles
- **MBIC**: Media Bias Identification Corpus

The original dataset contains several CSV files with the following columns:
- `text`: The article text
- `news_link`: Link to the original news article
- `outlet`: The news outlet that published the article
- `topic`: The topic of the article
- `type`: The type of article
- `group_id`: Group identifier
- `num_sent`: Number of sentences
- `label_bias`: Bias label (Biased or Non-biased)
- `label_opinion`: Opinion label
- `article`: Article identifier
- `biased_words`: Words identified as biased

## Processing Script

The `process_data.py` script handles downloading, extracting, and processing the BABE dataset.

### How to Run

```bash
# Run the script (will delete repository after processing)
python process_data.py

# Keep repository after processing
python process_data.py --keep-repo

# Use a specific random seed for data splitting
python process_data.py --seed 42
```

### Process

1. The script clones the BABE repository from GitHub
2. It processes each dataset (SG1, SG2, MBIC) separately:
   - Loads the data from the corresponding CSV file
   - Extracts the text and bias label (Biased or Non-biased) from each article
   - Splits the data into train (80%), validation (10%), and test (10%) sets
   - Saves the processed data in a dataset-specific subdirectory
3. It also creates a combined dataset with all examples from all three datasets
4. By default, the script will delete the cloned repository to save disk space (use `--keep-repo` to keep it)

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text
- `label`: The bias label (0 for Non-biased, 1 for Biased)

## Requirements

- Python 3.6+
- pandas
- numpy
- tqdm
- git

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `SG1/`: First dataset
    - `train.jsonl`: Training data
    - `valid.jsonl`: Validation data
    - `test.jsonl`: Test data
  - `SG2/`: Second dataset
    - `train.jsonl`: Training data
    - `valid.jsonl`: Validation data
    - `test.jsonl`: Test data
  - `MBIC/`: Third dataset
    - `train.jsonl`: Training data
    - `valid.jsonl`: Validation data
    - `test.jsonl`: Test data
  - `combined/`: Combined dataset
    - `train.jsonl`: Training data (combined from all datasets)
    - `valid.jsonl`: Validation data (combined from all datasets)
    - `test.jsonl`: Test data (combined from all datasets)

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{spinde-etal-2021-neural,
    title = "Neural Media Bias Detection Using Distant Supervision With {BABE} - Bias Annotations By Experts",
    author = "Spinde, Timo and Rudnitckaia, Lada and Mitrović, Jelena and Hamborg, Felix and Granitzer, Michael and Gipp, Bela and Donnay, Karsten",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "1166--1177",
}
```

## Data Source

The dataset is available from the original repository: [https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE) 