from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import os
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity
import json
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans

def encode(texts, batch_size=4):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = angle.encode(batch_texts, to_numpy=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def load_bignewsbln_data():
    # Load BIGNEWSBLN dataset
    with open("BIGNEWSBLN_left.json", "r") as f:
        left_data = json.load(f)
    with open("BIGNEWSBLN_center.json", "r") as f:
        center_data = json.load(f)
    with open("BIGNEWSBLN_right.json", "r") as f:
        right_data = json.load(f)
    
    # Prepare data and labels
    all_texts = []
    all_labels = []
    
    # Add left articles (label: -1)
    for article in left_data:
        all_texts.append(article['text'])
        all_labels.append(-1)
    
    # Add center articles (label: 0)
    for article in center_data:
        all_texts.append(article['text'])
        all_labels.append(0)
    
    # Add right articles (label: 1)
    for article in right_data:
        all_texts.append(article['text'])
        all_labels.append(1)
    
    concat_texts = []
    for text in all_texts:
        concat_texts.append(" ".join(text))
    
    return concat_texts, all_labels

def calculate_weighted_variance(cluster):
    # Get bias labels and counts
    labels = np.array(list(cluster.keys()))
    counts = np.array(list(cluster.values()))
    
    # Normalize counts to get proportions
    total = counts.sum()
    proportions = counts / total
    
    # Compute mean bias (weighted average)
    mean_bias = np.sum(proportions * labels)
    
    # Compute weighted variance
    variance = np.sum(proportions * (labels - mean_bias)**2)
    
    return variance


from openai import OpenAI
class OpenAIWrapper:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OpenAIWrapper, cls).__new__(cls)
            API_KEY = kwargs.get("API_KEY", None)
            if API_KEY is not None:
                os.environ["OPENAI_API_KEY"] = API_KEY
            cls._instance.client = OpenAI()
            try:
                test_response = cls._instance.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ]
                )
                print(f"Connected to OpenAI")
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(f"Error connecting to OpenAI: {e}\nNote we currently only support OpenAI models. Make sure you have setup the API key correctly or have passed the API key in as parameter. Make sure you have sufficient credits. Refer to https://platform.openai.com/docs/quickstart for more details on how to setup the API key.")
                raise e
        return cls._instance
        
def construct_openai_prompt_bias(texts, labels):
    txt = "Please summarize the following texts into a common topic, which the VAST MAJORITY of the texts debate on and can reflect the bias of the texts, which different bias (left, center, right) holds different views on this topic. Note there are multiple sides to the topic. Please summarize the topic in a neutral tone. Please return the topic and the bias indicators, without any other words or sentences. Please summarize the topic in less than 10 words. Give me the results in the following format: Topic: <Topic> \nLeft Indicator: <Some key points that Left or Lean Left have>\nRight Indicator: <Some key points that Right or Lean Right have>\nNeutral Indicator: <Some key points that Center have>"
    txt += "\n\n"
    # sort the texts and labels by bias
    texts, labels = zip(*sorted(zip(texts, labels), key=lambda x: x[1]))
    for text, label in zip(texts, labels):
        txt += f"Text: {text}\nBias: {label}\n\n"
    return txt

def parse_topic(topic):
    lines = topic.split("\n")
    topic_dict = {}
    for line in lines:
        if "Left Indicator" in line:
            topic_dict["left"] = line.split(":")[1].strip()
        elif "Right Indicator" in line:
            topic_dict["right"] = line.split(":")[1].strip()
        elif "Neutral Indicator" in line:
            topic_dict["neutral"] = line.split(":")[1].strip()
        elif "Topic" in line:
            topic_dict["topic"] = line.split(":")[1].strip()
    return topic_dict

all_texts, all_labels = load_bignewsbln_data()

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()


test_indices = np.random.choice(len(all_texts), 100000, replace=False)
train_indices = np.setdiff1d(np.arange(len(all_texts)), test_indices)

# # # save the test and train indices
with open("processed_data/bignewsbln_train_test_indices.pkl", "wb") as f:
    pickle.dump((train_indices, test_indices), f)

BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 10000  # Save every 10k samples

vecs = []
checkpoint_path = "/home/sunyq/PoliticalBiasEmbedding_New/vector_checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)

# Load the last checkpoint if exists
checkpoint_files = sorted([f for f in os.listdir(checkpoint_path) if f.startswith('vectors_')])
start_idx = 0
if checkpoint_files:
    last_checkpoint = checkpoint_files[-1]
    start_idx = int(last_checkpoint.split('_')[1].split('.')[0])
    loaded_vecs = np.load(os.path.join(checkpoint_path, last_checkpoint))
    vecs = [loaded_vecs]
    print(f"Resuming from checkpoint at index {start_idx}")

last_checkpoint_file = None
processed_samples = start_idx

for i in tqdm(range(start_idx, len(all_texts), BATCH_SIZE)):
    batch_texts = all_texts[i:i+BATCH_SIZE]
    batch_vecs = angle.encode(batch_texts, to_numpy=True)
    vecs.append(batch_vecs)
    
    # Update processed samples count
    current_batch_size = len(batch_texts)  # This handles the last batch which might be smaller
    processed_samples += current_batch_size
    
    # Save checkpoint every CHECKPOINT_INTERVAL samples
    if processed_samples // CHECKPOINT_INTERVAL > (processed_samples - current_batch_size) // CHECKPOINT_INTERVAL:
        # Delete previous checkpoint if exists
        if last_checkpoint_file and os.path.exists(last_checkpoint_file):
            os.remove(last_checkpoint_file)
            
        current_vecs = np.concatenate(vecs, axis=0)
        checkpoint_file = os.path.join(checkpoint_path, f'vectors_{processed_samples}.npy')
        np.save(checkpoint_file, current_vecs)
        print(f"Saved checkpoint at {processed_samples} samples")
        last_checkpoint_file = checkpoint_file

arr_vecs = np.concatenate(vecs, axis=0)

# Save final vectors and delete last checkpoint
if last_checkpoint_file and os.path.exists(last_checkpoint_file):
    os.remove(last_checkpoint_file)
final_file = os.path.join(checkpoint_path, 'vectors_final.npy')
np.save(final_file, arr_vecs)

# load the vectors
arr_vecs = np.load("/home/sunyq/PoliticalBiasEmbedding_New/vector_checkpoints/vectors_final.npy")

train_vecs = arr_vecs[train_indices]
test_vecs = arr_vecs[test_indices]

train_labels = [all_labels[i] for i in train_indices]
test_labels = [all_labels[i] for i in test_indices]

train_vecs_norm = train_vecs / np.linalg.norm(train_vecs, axis=1, keepdims=True)
test_vecs_norm = test_vecs / np.linalg.norm(test_vecs, axis=1, keepdims=True)



kmeans = KMeans(n_clusters=3000, random_state=42)
labels = kmeans.fit_predict(train_vecs_norm)
labels.shape

# save the labels and cluster centers and the kmeans model
with open("processed_data/bignews_labels.pkl", "wb") as f:
    pickle.dump(labels, f)
with open("processed_data/bignews_cluster_centers.pkl", "wb") as f:
    pickle.dump(kmeans.cluster_centers_, f)
with open("processed_data/bignews_kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

label_bias_info = {}
for i in range(len(labels)):
    cluster_id = labels[i]
    if cluster_id not in label_bias_info:
        label_bias_info[cluster_id] = {}
    original_index = train_indices[i]
    bias_num = all_labels[original_index]
    label_bias_info[cluster_id][bias_num] = label_bias_info[cluster_id].get(bias_num, 0) + 1

weighted_variance = {}
for cluster_id in label_bias_info:
    weighted_variance[cluster_id] = calculate_weighted_variance(label_bias_info[cluster_id])
    
# filter out clusters with less than 50 articles, and less than 1.0 variance
filtered_clusters = {cluster_id: variance for cluster_id, variance in weighted_variance.items() if variance > 0.5 and sum(label_bias_info[cluster_id].values()) > 50}
len(filtered_clusters)

# save the filtered clusters
with open("processed_data/filtered_clusters_bignews_normalized.pkl", "wb") as f:
    pickle.dump(filtered_clusters, f)

with open("processed_data/filtered_clusters_bignews_normalized.pkl", "rb") as f:
    filtered_clusters = pickle.load(f)

client = OpenAIWrapper().client

topics = {}

for cluster_id in tqdm(filtered_clusters):
    texts = [all_texts[i] for i in range(len(labels)) if labels[i] == cluster_id]
    gt_labels = [all_labels[i] for i in range(len(labels)) if labels[i] == cluster_id]
    gt_labels_text = ["Left" if label == -1 else "Right" if label == 1 else "Center" for label in gt_labels]
    if len(texts) > 50:
        sampled_texts = random.sample(texts, 50)
    else:
        sampled_texts = texts
    prompt = construct_openai_prompt_bias(sampled_texts, gt_labels_text)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    topics[int(cluster_id)] = response.choices[0].message.content

with open("processed_data/topics_bignews_normalized.json", "w") as f:
    json.dump(topics, f)

parsed_topics = {}
for cluster_id in tqdm(filtered_clusters):
    parsed_topics[int(cluster_id)] = parse_topic(topics[cluster_id])
with open("processed_data/parsed_topics_bignews_normalized.json", "w") as f:
    json.dump(parsed_topics, f)

with open("processed_data/parsed_topics_bignews_normalized.json", "r") as f:
    parsed_topics = json.load(f)

parsed_topics = {int(k): v for k, v in parsed_topics.items()}

good_topics = 0
for cluster_id in tqdm(parsed_topics):
    if "left" in parsed_topics[cluster_id] and "right" in parsed_topics[cluster_id] and "topic" in parsed_topics[cluster_id]:
        good_topics += 1
print(f"Percentage of good topics: {good_topics / len(filtered_clusters)}")

parsed_topics = {int(k): v for k, v in parsed_topics.items() if "left" in v and "right" in v and "topic" in v}

cluster_ids = list(parsed_topics.keys())
topics = [parsed_topics[cluster_id]["topic"] for cluster_id in cluster_ids]
left_indicators = [parsed_topics[cluster_id]["left"] for cluster_id in cluster_ids]
right_indicators = [parsed_topics[cluster_id]["right"] for cluster_id in cluster_ids]

data_for_train_new_format = []
for i in range(len(train_indices)):
    cluster_id = int(labels[i])
    if cluster_id not in parsed_topics:
        continue
    original_index = train_indices[i]
    text = all_texts[original_index]
    topic_info = parsed_topics[cluster_id]
    bias_score = all_labels[original_index]
    data_for_train_new_format.append({
        "text": text,
        "indicator": topic_info["left"],
        "bias_score": 1 if bias_score == -1 else 0
    })
    data_for_train_new_format.append({
        "text": text,
        "indicator": topic_info["right"],
        "bias_score": 1 if bias_score == 1 else 0
    })

for i in range(len(train_indices)):
    cluster_id = int(labels[i])
    if cluster_id not in parsed_topics:
        continue
    original_index = train_indices[i]
    text = all_texts[original_index]
    random_cluster_id = np.random.choice(list(parsed_topics.keys()))
    topic_info = parsed_topics[random_cluster_id]
    bias_score = all_labels[original_index]
    data_for_train_new_format.append({
        "text": text,
        "indicator": topic_info["left"],
        "bias_score": 0
    })
    data_for_train_new_format.append({
        "text": text,
        "indicator": topic_info["right"],
        "bias_score": 0
    })

with open("processed_data/data_for_train_new_format_with_non_relevant_bignews_normalized.json", "w") as f:
    json.dump(data_for_train_new_format, f)