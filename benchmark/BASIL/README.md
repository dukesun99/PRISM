# BASIL Dataset Processing

This directory contains a script to download and process the BASIL (Bias Annotation Spans on the Interpretability of Language) dataset.

## Dataset Description

The BASIL dataset contains news articles annotated for media bias. It includes articles from three major news sources with different political leanings:
- Fox News (conservative)
- New York Times (liberal)
- Washington Post (liberal)

The dataset provides article-level annotations for political stance (conservative/right, center, liberal/left) as well as span-level annotations for bias.

## Dataset Structure

The original repository has the following structure:
- `articles/`: Contains JSON files organized by year, each representing an article with body paragraphs
- `annotations/`: Contains JSON files organized by year, each representing annotations for an article

## Processing Script

The `process_data.py` script handles downloading, extracting, and processing the BASIL dataset.

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

1. The script clones the BASIL repository from GitHub
2. It loads all articles from the repository, extracting the text from body paragraphs
3. It loads all annotations from the repository, extracting the relative stance (conservative/right, center, liberal/left)
4. It merges articles and annotations based on UUID
5. The data is split into train (80%), validation (10%), and test (10%) sets
6. The processed data is saved in JSONL format with fields for text and label
7. By default, the script will delete the cloned repository to save disk space (use `--keep-repo` to keep it)

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text (concatenated from body paragraphs)
- `label`: The political stance label (0 for liberal/left, 1 for center, 2 for conservative/right)

## Requirements

- Python 3.6+
- pandas
- numpy
- tqdm
- git

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `train.jsonl`: Training data (80% of the dataset)
  - `valid.jsonl`: Validation data (10% of the dataset)
  - `test.jsonl`: Test data (10% of the dataset)

## Citation

If you use this dataset, please cite the original paper:
```
@inproceedings{fan-etal-2019-plain,
    title = "In Plain Sight: Media Bias Through the Lens of Factual Reporting",
    author = "Fan, Lisa and White, Marshall and Sharma, Eva and Su, Ruisi and Choubey, Prafulla Kumar and Huang, Ruihong and Wang, Lu",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    pages = "6343--6349",
}
```

## Data Source

The dataset is available from the original repository: [https://github.com/launchnlp/BASIL](https://github.com/launchnlp/BASIL) 