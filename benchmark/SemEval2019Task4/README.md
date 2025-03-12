# SemEval 2019 Task 4 Dataset Processing

This directory contains a script to download and process the SemEval 2019 Task 4 dataset for hyperpartisan news detection.

## Dataset Description

The SemEval 2019 Task 4 dataset contains news articles labeled for hyperpartisan bias (i.e., whether they exhibit extreme bias in favor of one political side). The dataset is available in two versions:

1. **By-Article**: Articles labeled individually by human annotators
2. **By-Publisher**: Articles labeled based on the overall bias of their publisher

## Processing Script

The `process_data.py` script handles downloading and processing both versions of the dataset directly from the XML files.

### How to Run

```bash
# Run the script (will download and process the dataset)
python process_data.py

# Skip downloading (if you already have the data)
python process_data.py --skip-download

# Keep raw data files after processing
python process_data.py --keep-raw

# Automatically answer yes to all prompts (for batch processing)
python process_data.py --yes
```

### Process

1. The script will download files from the Zenodo repository (skipping files that already exist)
2. It will extract any zip files and delete the original zip files to save space
3. The script will process both the By-Article and By-Publisher datasets directly from the XML files:
   - It will parse the article XML files and corresponding ground truth XML files
   - It will match articles with their ground truth labels
   - It will split the data into train (80%), validation (10%), and test (10%) sets
4. The processed data will be converted to JSONL format and saved in separate directories
5. By default, the script will delete the raw data files to save space (use `--keep-raw` to keep them)

## Data Format

The processed data will be in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text
- `label`: The hyperpartisan label (1 for hyperpartisan, 0 for not hyperpartisan)

## Requirements

- Python 3.6+
- pandas
- requests
- xml.etree.ElementTree (standard library)

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - By-Article dataset in `processed_data/byarticle/`:
    - `train.jsonl`: Training data (80% of the dataset)
    - `valid.jsonl`: Validation data (10% of the dataset)
    - `test.jsonl`: Test data (10% of the dataset)
  - By-Publisher dataset in `processed_data/bypublisher/`:
    - `train.jsonl`: Training data (80% of the dataset)
    - `valid.jsonl`: Validation data (10% of the dataset)
    - `test.jsonl`: Test data (10% of the dataset)

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{kiesel2019semeval,
  title={SemEval-2019 Task 4: Hyperpartisan News Detection},
  author={Kiesel, Johannes and Mestre, Maria and Shukla, Rishabh and Vincent, Emmanuel and Adineh, Payam and Corney, David and Stein, Benno and Potthast, Martin},
  booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation},
  pages={829--839},
  year={2019}
}
```

## Data Source

The dataset is available from Zenodo: [https://zenodo.org/records/1489920](https://zenodo.org/records/1489920) 