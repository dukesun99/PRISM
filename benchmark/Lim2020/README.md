# Lim2020 Dataset Processing

This directory contains a script to download and process the Lim2020 dataset for sentence-level bias detection in news articles.

## Dataset Description

The Lim2020 dataset contains news articles about four major events, with sentences annotated for political bias. The dataset includes articles from various news sources with different political leanings.

According to the original paper, the dataset contains 966 unique sentences with 4,515 annotations (approximately 5 annotations per sentence) from 46 news articles.

The four events covered in the dataset are:
1. **Johnson**: News about Kentucky lawmaker Dan Johnson's suicide after sexual assault allegations
2. **Facebook**: News about Facebook's data privacy scandal
3. **NFL**: News about NFL players kneeling during the national anthem
4. **NorthKora**: News about North Korea's nuclear program

Each sentence in the dataset is labeled with one of four bias levels:
- **Neutral** (0): No bias detected
- **Slightly Biased but Acceptable** (1): Mild bias that is considered acceptable
- **Biased** (2): Clear bias in the sentence
- **Very Biased** (3): Strong bias in the sentence

## Processing Script

The `process_data.py` script handles downloading and processing the Lim2020 dataset at the sentence level.

### How to Run

```bash
# Run the script (will download and process the dataset)
python process_data.py

# Skip downloading if you already have the data
python process_data.py --skip-download

# Keep raw data files after processing
python process_data.py --keep-raw

# Use a specific random seed for data splitting
python process_data.py --seed 123
```

### Process

1. The script downloads the CSV file from GitHub
2. It processes the dataset to extract sentences with their bias labels
3. It identifies unique sentences and aggregates multiple annotations:
   - For sentences with multiple annotations, it uses majority voting
   - In case of ties, it selects the higher bias level
4. It organizes the data by event (Johnson, Facebook, NFL, NorthKora)
5. It splits each event's data into train (80%), validation (10%), and test (10%) sets
6. It also creates a combined dataset with all events
7. By default, the script will delete the raw data files to save disk space (use `--keep-raw` to keep them)

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The sentence text
- `label`: The aggregated bias label (0 for neutral, 1 for slightly biased, 2 for biased, 3 for very biased)
- `event`: The event the sentence is about (Johnson, Facebook, NFL, NorthKora)
- `date`: The publication date of the article
- `article_id`: The ID of the article
- `source`: The news source of the article
- `source_bias`: The political leaning of the news source
- `position`: The position of the sentence in the article
- `num_annotations`: The number of annotations this sentence received in the original dataset

## Requirements

- Python 3.6+
- requests
- tqdm

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `johnson/`: Sentences about the Johnson event
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `facebook/`: Sentences about the Facebook event
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `nfl/`: Sentences about the NFL event
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `northkora/`: Sentences about the North Korea event
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)
  - `combined/`: All sentences combined
    - `train.jsonl`: Training set (80%)
    - `valid.jsonl`: Validation set (10%)
    - `test.jsonl`: Test set (10%)

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{lim-etal-2020-annotating,
    title = "Annotating and Analyzing Biased Sentences in News Articles using Crowdsourcing",
    author = "Lim, Sora  and
      Jang, Adam  and
      Bak, JinYeong  and
      Oh, Alice",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.171",
    pages = "1478--1484",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```

## Data Source

The dataset is available from GitHub: [https://github.com/skymoonlight/biased-sents-annotation](https://github.com/skymoonlight/biased-sents-annotation) 