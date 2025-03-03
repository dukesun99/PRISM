from cross_encoder_hf_cls import PoliticalBiasCrossEncoder
import torch
import os
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity
import json
import numpy as np
from tqdm import tqdm
from flask import Flask, request, jsonify
import argparse

# Add command line arguments
parser = argparse.ArgumentParser(description='Run the political bias encoding API server')
parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
args = parser.parse_args()

# Set the GPU device
device = f'cuda:{args.gpu}'
torch.cuda.set_device(args.gpu)

app = Flask(__name__)

# Load all the required data and models at startup
with open("processed_data/parsed_topics_bignews_normalized.json", "r") as f:
    parsed_topics = json.load(f)

parsed_topics = {int(k): v for k, v in parsed_topics.items() if "left" in v and "right" in v and "topic" in v}

# Initialize models and embeddings with specified GPU
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)

cluster_ids = list(parsed_topics.keys())
# sort cluster_ids
cluster_ids = sorted(cluster_ids)
topics = [parsed_topics[cluster_id]["topic"] for cluster_id in cluster_ids]
left_indicators = [parsed_topics[cluster_id]["left"] for cluster_id in cluster_ids]
right_indicators = [parsed_topics[cluster_id]["right"] for cluster_id in cluster_ids]

def angle_encode(texts, batch_size=64):
    encoded_texts = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoded_texts.append(angle.encode(batch_texts, to_numpy=True))
    encoded_texts = np.concatenate(encoded_texts, axis=0)
    return encoded_texts

# Pre-compute embeddings
topic_embeddings = angle_encode(topics)
left_embeddings = angle_encode(left_indicators)
right_embeddings = angle_encode(right_indicators)

# Load the model
model_path = "checkpoints/bignews/model.pt"
model = PoliticalBiasCrossEncoder(model_name="microsoft/deberta-v3-large")
checkpoint = torch.load(model_path, map_location="cpu")
# Extract just the model state dict and load it
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

from tqdm import tqdm
def extract_features_with_crossencoder_both_sim_and_div(texts):
    model.eval()    
    # encode the texts
    BATCH_SIZE = 64
    TOPIC_BATCH_SIZE = 16
    threshold = 0.1
    encoded_texts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        encoded_texts.append(angle.encode(batch_texts, to_numpy=True))
    encoded_texts = np.concatenate(encoded_texts, axis=0)
    # normalize the encoded_texts
    encoded_texts = encoded_texts / np.linalg.norm(encoded_texts, axis=1, keepdims=True)
    # first get top similar linear_encodeings for each encoded_text that above the threshold
    extracted_features = []
    for i in tqdm(range(len(encoded_texts))):
        encoded_text = encoded_texts[i]
        left_similarities = left_embeddings @ encoded_text.T
        right_similarities = right_embeddings @ encoded_text.T
        topic_similarities = topic_embeddings @ encoded_text.T
        div_scores = right_similarities - left_similarities
        abs_div_scores = np.abs(div_scores)
        weight = 0.8
        overall_scores = topic_similarities * weight + abs_div_scores * (1 - weight)
        
        # keep top 10 topics with highest overall_scores
        top_idx = np.argsort(overall_scores)[-10:]
        idx_above_threshold = np.zeros_like(overall_scores)
        idx_above_threshold[top_idx] = True
        
        feature_row = np.zeros(len(cluster_ids))
        valid_indices = np.where(idx_above_threshold)[0]
        # Process topics in batches
        for batch_start in range(0, len(valid_indices), TOPIC_BATCH_SIZE):
            batch_indices = valid_indices[batch_start:batch_start + TOPIC_BATCH_SIZE]
            
            # Prepare batch data
            batch_articles = [texts[i]] * len(batch_indices)
            batch_lefts = [parsed_topics[int(cluster_ids[idx])]["left"] for idx in batch_indices]
            batch_rights = [parsed_topics[int(cluster_ids[idx])]["right"] for idx in batch_indices]
            # Process batch
            with torch.no_grad():
                left_scores = model(batch_articles, batch_lefts)
                left_scores = left_scores.squeeze(1)
                right_scores = model(batch_articles, batch_rights)
                right_scores = right_scores.squeeze(1)
                assert left_scores.shape == right_scores.shape
                overall_scores = right_scores - left_scores
            # Assign scores to feature row
            for score_idx, orig_idx in enumerate(batch_indices):
                feature_row[orig_idx] = overall_scores[score_idx]
            
        extracted_features.append(feature_row)
    return np.array(extracted_features)

@app.route('/encode', methods=['POST'])
def encode_text():
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            texts = [texts]
            
        features = extract_features_with_crossencoder_both_sim_and_div(texts)
        return jsonify({
            'features': features.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting server on port {args.port} using GPU {args.gpu}")
    app.run(host='0.0.0.0', port=args.port)