# SemEval 2016 Task 6 Dataset Processing

This directory contains a script to download and process the SemEval 2016 Task 6 dataset for stance detection in tweets.

## Dataset Description

The SemEval 2016 Task 6 dataset contains tweets labeled for stance detection. Each tweet is labeled with one of three stances (FAVOR, AGAINST, or NONE) toward a specific target. The dataset includes five different targets:

1. Atheism
2. Climate Change is a Real Concern
3. Feminist Movement
4. Hillary Clinton
5. Legalization of Abortion

## Processing Script

The `process_data.py` script handles downloading, extracting, and processing the dataset. It organizes the data into separate subdatasets for each target.

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

1. The script downloads the dataset from the SemEval 2016 Task 6 website
2. It extracts the zip file to access the train.csv and test.csv files in the StanceDataset directory
3. The script splits the test data into validation and test sets (50% each)
4. It processes the data into separate subdatasets for each target
5. The processed data is saved in JSONL format with fields for text and label
6. By default, the script will delete the raw data files to save space (use `--keep-raw` to keep them)

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The tweet text
- `label`: The stance label (0 for AGAINST, 1 for NONE, 2 for FAVOR)

## Requirements

- Python 3.6+
- pandas
- requests
- tqdm

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - Subdirectories for each target (with spaces replaced by underscores):
    - `Atheism/`
    - `Climate_Change_is_a_Real_Concern/`
    - `Feminist_Movement/`
    - `Hillary_Clinton/`
    - `Legalization_of_Abortion/`
  - Each target subdirectory contains:
    - `train.jsonl`: Training data for that target
    - `valid.jsonl`: Validation data for that target
    - `test.jsonl`: Test data for that target

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{mohammad-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 6: Detecting Stance in Tweets",
    author = "Mohammad, Saif and Kiritchenko, Svetlana and Sobhani, Parinaz and Zhu, Xiaodan and Cherry, Colin",
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)",
    year = "2016",
    publisher = "Association for Computational Linguistics",
    pages = "31--41",
}
```

## Data Source

The dataset is available from the SemEval 2016 Task 6 website: [http://alt.qcri.org/semeval2016/task6/](http://alt.qcri.org/semeval2016/task6/) 