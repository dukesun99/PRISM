# Roy2020 Dataset Processing

This directory contains a script to download and process the Roy2020 dataset for political bias detection in news articles.

## Dataset Description

The Roy2020 dataset contains news articles labeled for political bias (left or right) across three different topics:
1. **Abortion**
2. **Immigration**
3. **Gun Control**

Each subdataset is stored in a pickle file with the following structure:
```python
[article2URL, article2dop, article2headline, article2text, art2label, article2segment_ids, seg_id2text]
```

Where:
- `article2URL`: Dictionary mapping article IDs to URLs
- `article2dop`: Dictionary mapping article IDs to dates of publication
- `article2headline`: Dictionary mapping article IDs to headlines
- `article2text`: Dictionary mapping article IDs to article text
- `art2label`: Dictionary mapping article IDs to political bias labels (left/right)
- `article2segment_ids`: Dictionary mapping article IDs to segment IDs
- `seg_id2text`: Dictionary mapping segment IDs to segment text

## Processing Script

The `process_data.py` script handles downloading and processing the Roy2020 dataset.

### How to Run

```bash
# Run the script (will download and process all subdatasets)
python process_data.py

# Skip downloading if you already have the data
python process_data.py --skip-download

# Keep raw data files after processing
python process_data.py --keep-raw

# Process only a specific subdataset
python process_data.py --subdataset abortion
python process_data.py --subdataset immigration
python process_data.py --subdataset gun_control

# Use a specific random seed for data splitting
python process_data.py --seed 123
```

### Process

1. The script downloads the pickle files from Google Drive for each subdataset
2. It processes each subdataset to extract:
   - The article text
   - The political bias label (left/right)
   - Additional metadata (title, URL, date, topic)
3. It splits each subdataset into train (80%), validation (10%), and test (10%) sets
4. It saves the processed data in JSONL format for each split
5. By default, the script will delete the raw data files to save disk space (use `--keep-raw` to keep them)

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text
- `label`: The political bias label (0 for left, 1 for right)
- `topic`: The topic of the article (abortion, immigration, or gun_control)
- `title`: The headline of the article (if available)
- `url`: The URL of the article (if available)
- `date`: The publication date of the article (if available)

## Requirements

- Python 3.6+
- pandas
- gdown
- tqdm

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `abortion/`: Articles about abortion
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `immigration/`: Articles about immigration
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `gun_control/`: Articles about gun control
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{roy-goldwasser-2020-weakly,
    title = "Weakly Supervised Learning of Nuanced Frames for Analyzing Polarization in News Media",
    author = "Roy, Shamik  and
      Goldwasser, Dan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.620",
    doi = "10.18653/v1/2020.emnlp-main.620",
    pages = "7698--7716"
}
```

## Data Source

The dataset is available from Google Drive:
- Abortion: [https://drive.google.com/drive/folders/1g9P6vHTJ1nNec1zrBZggcNYu_Lmm95Oa?usp=sharing](https://drive.google.com/drive/folders/1g9P6vHTJ1nNec1zrBZggcNYu_Lmm95Oa?usp=sharing)
- Immigration: [https://drive.google.com/drive/folders/1OAND83Jtng46WuVKMZx0dpO2o3UtbuUD?usp=sharing](https://drive.google.com/drive/folders/1OAND83Jtng46WuVKMZx0dpO2o3UtbuUD?usp=sharing)
- Gun Control: [https://drive.google.com/drive/folders/1nJ-kzZeIJkUzwRTjFAOAPTbMpd64N7L_?usp=sharing](https://drive.google.com/drive/folders/1nJ-kzZeIJkUzwRTjFAOAPTbMpd64N7L_?usp=sharing) 