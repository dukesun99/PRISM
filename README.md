# PRISM🏳️‍🌈⃤── : A Framework for **P**roducing **I**nterpretable Pol**iI**tical Bia**S** E**M**beddings with Political-Aware Cross-Encoder

## Overview
This repository contains the code for the PRISM paper: "PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder". 

Key features:
- Mining controversial topics from weakly-labeled news articles
- Training a political-aware cross-encoder for interpretable political bias embedding
- Running the inference with the pre-trained PRISM models on the BigNews dataset

## Data
Please get your own data from the [BigNews dataset](https://github.com/launchnlp/POLITICS) and the [NewsSpectrum dataset](https://github.com/dukesun99/DiversiNews).

## Installation

### Prerequisites
- Anaconda: Recommended, but optional. You may use any other environment management system. We use conda in the following instructions. See Anaconda Installation Guide.
- Git CLI and Git LFS installed.
- OpenAI API key: Required for calling OpenAI APIs. See OpenAI Quickstart to get one with sufficient credits.
- GPU: A GPU with CUDA support with at least 12GB VRAM for faster encoding and training is highly recommended, but optional.


### Setup
Clone our repository and install the dependencies. Repository URL hidden for anonymity. Note this also downloads the LFS files (model checkpoints) automatically so please be patient.

```bash
git clone <repository_url>
cd CQG-MBQA
```

> 🔔 As you are viewing an Anonymous GitHub, git clone may not work. Please download the repository as a zip file and unzip it, and pull the LFS files manually.

> 💡 If you have not set up your OpenAI API key, please do so before proceeding. See Setup OpenAI API Key for instructions.

### Anaconda Environment
First, we need to create and activate a conda environment.
```bash
conda create -n prism python=3.9
conda activate prism
```

Then, install the dependencies.
```bash
pip install -r requirements.txt
```

🏆 Great! You're all set to go. 

### Data Preparation

This section describes how to mine controversial topics and prepare training data for the PRISM model using the BigNews dataset.

#### Controversial Topic and Bias Indicator Mining

The first phase involves discovering politically controversial topics and their associated bias indicators from the news articles:

1. **Initial Data Processing**
   ```bash
   python prepare_data.py
   ```
   - Loads and preprocesses articles from the BigNews dataset (left, center, right)
   - Encodes article texts using a pre-trained generic text encoder
   - Splits data into training (90%) and test (10%) sets

2. **Topic Clustering and Filtering**
   - Applies K-means clustering (3000 clusters) to group similar articles
   - Computes political diversity score for each cluster using weighted variance
   - Filters clusters based on:
     * Political diversity (variance > 0.5)
     * Minimum size (> 50 articles)
     * Balanced representation of different political views

3. **Bias Indicator Extraction**
   - For each politically diverse cluster:
     * Samples representative articles (up to 50)
     * Uses GPT-4 to extract:
       - Topic description (neutral summary)
       - Left-leaning viewpoint indicators
       - Right-leaning viewpoint indicators
     * Validates extracted indicators for quality

#### Political-Aware Cross-Encoder Training Data Generation

The second phase prepares the training data for the cross-encoder model:

1. **Positive Example Generation**
   - For each article in a cluster:
     * Pairs with its cluster's left indicator (label 1 if article is left-leaning, 0 otherwise)
     * Pairs with its cluster's right indicator (label 1 if article is right-leaning, 0 otherwise)

2. **Negative Example Generation**
   - For each article:
     * Randomly selects indicators from other clusters
     * Creates article-indicator pairs with label 0
   - Ensures balanced distribution of positive and negative examples

#### Requirements

- OpenAI API key (for GPT-4o-mini topic extraction)
- Pre-trained generic text encoder (UAE-Large-V1)
- scikit-learn with Intel Extension
- At least 256GB RAM for processing large datasets
- CUDA-capable GPU (recommended for faster encoding)

#### Configuration

Key parameters in `prepare_data.py`:
- Topic clustering: 3000 initial clusters
- Cluster filtering:
  * Minimum size: 50 articles
  * Political variance threshold: 0.5
- Processing:
  * Encoding batch size: 32
  * Topic sampling: Up to 50 articles per cluster

## Training the Political-Aware Cross-Encoder

This section describes how to train the PRISM political-aware cross-encoder model on your prepared dataset.

### Overview

The training process involves:
1. Loading the preprocessed political bias dataset
2. Training a cross-encoder model to learn political bias patterns
3. Monitoring training progress and saving checkpoints

### Requirements

- PyTorch
- Transformers library
- wandb (Weights & Biases) for experiment tracking
- CUDA-capable GPU with at least 12GB VRAM

### Training Process

1. **Prepare Your Environment**
   ```bash
   wandb login  # Login to Weights & Biases for experiment tracking
   ```

2. **Start Training**
   ```bash
   python training_cross_encoder.py
   ```

   The script will:
   - Split your data into training (90%) and validation (10%) sets
   - Initialize the model and training configurations
   - Train for the specified number of epochs
   - Save checkpoints and monitor progress via wandb

### Training Configuration

Key parameters in `training_cross_encoder.py`:
- Batch size: 16
- Learning rate: 1e-6
- Number of epochs: 5
- Optimizer: AdamW with cosine annealing scheduler
- Checkpoint saving: Every 25,000 steps
- Validation: Maximum 30,000 samples evaluated

### Model Architecture

The political-aware cross-encoder is designed to:
- Take pairs of (article text, political indicator) as input
- Output a bias score indicating the political alignment
- Use mean squared error loss for training

### Monitoring Training

Training progress can be monitored through:
- Weights & Biases dashboard
- Console output showing:
  - Training loss
  - Validation loss
  - Sample predictions
  - Learning rate changes

### Output Files

The training process saves:
- Best model checkpoint: `checkpoints/bignews/best_model.pt`
- Latest checkpoint: `checkpoints/bignews/latest_checkpoint.pt`
- Training logs: Available in your wandb dashboard

#### NewsSpectrum Dataset

#### Pre-trained Models
We provide the pre-trained PRISM models for the BigNews dataset and the NewsSpectrum dataset. Please find them in the `checkpoints` folder. Note the LFS files might not work properly in the Anonymous GitHub, but you can pull the LFS files manually.

## Deploying PRISM Model as an API Service

This section describes how to deploy the trained PRISM model and mined controversial topics as an inference API service. The service allows you to analyze political bias in new texts using your trained model.

### Overview

The service deploys your trained political-aware cross-encoder model along with a pre-trained generic text encoder to analyze political bias in text. It uses the controversial topics and left/right indicators mined during the training phase as reference points for bias analysis.

### Requirements

- PyTorch
- Flask
- AnglE embeddings library
- Your trained PRISM model checkpoint
- The mined controversial topics
- CUDA-capable GPU

### Setup

Required files from training:
- `checkpoints/bignews/parsed_topics.json`: Your mined controversial topics with left/right indicators
- `checkpoints/bignews/model.pt`: Your trained PRISM model checkpoint

### Usage

#### Starting the Server

```bash
python run_cross_encoder_inference_api.py --port 5000 --gpu 0
```

Parameters:
- `--port`: Port number to run the server (default: 5000)
- `--gpu`: GPU ID to use (default: 0)

#### API Endpoint

The service exposes a single endpoint for bias analysis:

**POST /encode**

Request body:
```json
{
    "texts": ["your text here"] // Single string or array of strings
}
```

Response:
```json
{
    "features": [[...]] // Array of political bias feature vectors
}
```

### How it Works

1. The service loads your mined controversial topics and their pre-computed embeddings
2. For each input text:
   - Computes text embeddings using the pre-trained generic text encoder
   - Finds most relevant controversial topics using cosine similarity
   - Uses your trained political-aware cross-encoder to compare text against left/right indicators
   - Returns feature vectors representing political bias scores across topics

### Performance Notes

- Uses batch processing for both embedding generation and PRISM model inference
- Supports multiple texts in a single request
- Processes topics in batches of 16 and texts in batches of 64 for optimal performance




