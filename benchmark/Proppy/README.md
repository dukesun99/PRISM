# Proppy Dataset Processing

This directory contains a script to download and process the Proppy dataset for propaganda detection in news articles.

## Dataset Description

The Proppy dataset contains approximately 52,000 news articles from over 100 news outlets, labeled for propaganda content. The dataset was created by Alberto Barrón-Cedeño et al. and uses distant supervision for labeling (i.e., an article is considered propagandistic if it comes from a news outlet that has been labeled as propagandistic by human annotators).

The original dataset contains three TSV files (train, dev, test) with the following columns (note: the files do not contain headers):
- Column 0: `article_text`: The text of the article
- Column 1: `event_location`: The geographical location
- Column 2: `average_tone`: Measures the impact of the event
- Column 3: `article_date`: Article's publish date
- Column 4: `article_ID`: GDELT ID
- Column 5: `article_URL`: The direct URL for the article
- Column 6: `author`: The article's author
- Column 7: `title`: The article's title
- Column 8: `MBFC_factuality_label`: Factuality label for the source
- Column 9: `source_url`: URL of the news source
- Column 10: `source_name`: Name of the news source
- Column 11: `source_description`: Description of the news source
- Column 12: `MBFC_bias_label`: Political bias label (not used in this processing)
- Column 13: `source_domain`: Domain of the news source
- Column 14: `propaganda_label`: Whether the article is propagandistic or not (1 for propaganda, 0 for not propaganda)

## Processing Script

The `process_data.py` script handles downloading and processing the Proppy dataset for propaganda detection.

### How to Run

```bash
# Run the script (will download and process the dataset)
python process_data.py

# Skip downloading if you already have the data
python process_data.py --skip-download

# Keep raw data files after processing
python process_data.py --keep-raw

# Suppress warnings about unrecognized labels
python process_data.py --quiet
```

### Process

1. The script downloads the three TSV files from Zenodo
2. It processes each file to extract:
   - The article text
   - The propaganda label (propaganda, not propaganda)
3. It saves the processed data directly in the `processed_data/` directory
4. By default, the script will delete the raw TSV files to save disk space (use `--keep-raw` to keep them)

### Data Format Handling

The script is designed to handle the TSV files that do not contain headers. It:
1. Assigns predefined column names based on the dataset documentation
2. Automatically adjusts if the number of columns doesn't match expectations
3. Maps important columns by position if needed
4. Provides detailed logging about the file structure and processing steps
5. Filters out invalid labels like "nil" and "unknown" automatically
6. Provides statistics on the number of valid and invalid entries

### Label Mapping

For propaganda labels, the mapping is simple:
- **0**: Not propaganda
- **1**: Propaganda

## Data Format

The processed data is in JSONL format, with each line containing a JSON object with the following fields:
- `text`: The article text
- `label`: The propaganda label (0 for not propaganda, 1 for propaganda)

## Requirements

- Python 3.6+
- pandas
- requests
- tqdm

## Data Structure

After running the script, the data will be organized as follows:
- Processed data in the `processed_data/` directory:
  - `train.jsonl`: Training data
  - `valid.jsonl`: Validation data
  - `test.jsonl`: Test data

## Citation

If you use this dataset, please cite the original paper:
```
@article{BARRONCEDENO20191849,
    author = "Barr\\'{o}n-Cede\\\~no, Alberto and Da San Martino, Giovanni and Jaradat, Israa and Nakov, Preslav",
    title = "{Proppy: Organizing the news based on their propagandistic content}",
    journal = "Information Processing & Management",
    volume = "56",
    number = "5",
    pages = "1849 - 1864",
    year = "2019",
    issn = "0306-4573",
    doi = "https://doi.org/10.1016/j.ipm.2019.03.005",
}
```

## Data Source

The dataset is available from Zenodo: [https://zenodo.org/records/3271522](https://zenodo.org/records/3271522) 