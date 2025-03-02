import pickle
import json
import numpy as np
from tqdm import tqdm
import pydkmips
from angle_emb import AnglE
import os

def calculate_diversity_score(list_of_bias):
    sum_of_pairwise_bias = 0
    count_of_pairwise_bias = 0
    for i in range(len(list_of_bias)):
        for j in range(i+1, len(list_of_bias)):
            sum_of_pairwise_bias += abs(list_of_bias[i] - list_of_bias[j])
            count_of_pairwise_bias += 1
    return sum_of_pairwise_bias / count_of_pairwise_bias


def evaluate_random_retrieval(query_vecs, doc_vecs, doc_labels):
    """Evaluate random retrieval baseline"""
    print("\nEvaluating Random baseline...")
    sum_similarity_score = 0
    sum_diversity_score = 0
    doc_vecs = np.array(doc_vecs)
    doc_labels = np.array(doc_labels)
    query_vecs = np.array(query_vecs)
    
    for query_idx in tqdm(range(len(query_vecs))):
        # randomly select 10 documents
        random_indices = np.random.choice(len(doc_vecs), 10, replace=False)
        random_documents = doc_vecs[random_indices]
        random_document_labels = doc_labels[random_indices]
        sum_similarity_score += np.mean(np.dot(random_documents, query_vecs[query_idx]))
        sum_diversity_score += calculate_diversity_score(random_document_labels)
    
    diversity = sum_diversity_score / len(query_vecs)
    similarity = sum_similarity_score / len(query_vecs)
    print(f"Random baseline - diversity: {diversity:.4f}, similarity: {similarity:.4f}")
    return {'random': {0: (diversity, similarity)}}

def evaluate_linear_retrieval(query_vecs, doc_vecs, doc_labels):
    """Evaluate linear retrieval baseline"""
    print("\nEvaluating Linear baseline...")
    sum_similarity_score = 0
    sum_diversity_score = 0
    doc_vecs = np.array(doc_vecs)
    doc_labels = np.array(doc_labels)
    query_vecs = np.array(query_vecs)
    
    for query_idx in tqdm(range(len(query_vecs))):
        query_vec = query_vecs[query_idx]
        # get top 10 similar documents
        similarity_scores = np.dot(doc_vecs, query_vec)
        top_10_indices = np.argsort(similarity_scores)[-10:]
        top_10_document_labels = doc_labels[top_10_indices]
        sum_similarity_score += np.mean(similarity_scores[top_10_indices])
        sum_diversity_score += calculate_diversity_score(top_10_document_labels)
    
    diversity = sum_diversity_score / len(query_vecs)
    similarity = sum_similarity_score / len(query_vecs)
    print(f"Linear baseline - diversity: {diversity:.4f}, similarity: {similarity:.4f}")
    return {'linear': {0: (diversity, similarity)}}

def evaluate_method(index, query_vecs, doc_vecs, doc_labels, method_name):
    print(f"\nEvaluating {method_name}...")
    scores = {}
    for c in tqdm(np.arange(0.01, 1.01, 0.1)):
        sum_diversity_score = 0
        sum_similarity_score = 0
        for idx in range(len(query_vecs)):
            user_vec = query_vecs[idx]
            D, I = index.search(user_vec, 10, lambda_param=0.5, c=c, objective="avg")
            list_of_bias = [doc_labels[i] for i in I]
            sum_diversity_score += calculate_diversity_score(list_of_bias)
            retrieved_vecs = [doc_vecs[i] for i in I]
            sum_similarity_score += np.mean(D)
        scores[c] = (sum_diversity_score / len(query_vecs), 
                    sum_similarity_score / len(query_vecs))
        print(f"c: {c:.2f}, diversity: {scores[c][0]:.4f}, similarity: {scores[c][1]:.4f}")
    return scores

def angle_encode_with_batch(texts, angle_model, batch_size=64, cache_file=None):
    """Encode texts with angle embeddings using manual batching and caching"""
    # Try to load from cache first
    if cache_file and os.path.exists(cache_file):
        print(f"Loading angle embeddings from cache: {cache_file}")
        return np.load(cache_file)
    
    print("Computing angle embeddings...")
    encoded_texts = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = angle_model.encode(batch_texts, to_numpy=True)
        encoded_texts.append(batch_embeddings)
    
    embeddings = np.concatenate(encoded_texts, axis=0)
    
    # Save to cache if cache_file is provided
    if cache_file:
        print(f"Saving angle embeddings to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, embeddings)
    
    return embeddings

def main():
    # Load data
    print("Loading data...")
    with open("./processed_data/bignewsbln_data.json", "r") as f:
        data = json.load(f)
        all_texts = data["texts"]
        all_labels = data["labels"]
        
    # Load train/test indices
    print("Loading train/test indices...")
    with open("./processed_data/bignewsbln_train_test_indices.pkl", "rb") as f:
        train_indices, test_indices = pickle.load(f)
    
    # Split test indices into query and document sets
    print("Splitting test indices into query and document sets...")
    test_indices = np.array(test_indices)
    np.random.seed(42)  # For reproducibility
    query_indices = np.random.choice(test_indices, size=1000, replace=False)
    doc_indices = np.array([idx for idx in test_indices if idx not in query_indices])

    print(f"Number of query documents: {len(query_indices)}")
    print(f"Number of candidate documents: {len(doc_indices)}")

    # Get document labels
    doc_labels = [all_labels[i] for i in doc_indices]

    # Load all embeddings
    print("\nLoading embeddings...")
    
    # 1. Load angle embeddings
    print("Generating angle embeddings...")
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    query_texts = [all_texts[i] for i in query_indices]
    doc_texts = [all_texts[i] for i in doc_indices]
    
    # Create cache directory if it doesn't exist
    os.makedirs("embeddings_cache", exist_ok=True)
    
    print("Encoding query texts...")
    angle_embeddings_query = angle_encode_with_batch(
        query_texts, angle, batch_size=64,
        cache_file="embeddings_cache/angle_query_embeddings.npy"
    )
    print("Encoding document texts...")
    angle_embeddings_doc = angle_encode_with_batch(
        doc_texts, angle, batch_size=64,
        cache_file="embeddings_cache/angle_doc_embeddings.npy"
    )
    
    # 2. Load politics embeddings
    print("Loading politics embeddings...")
    with open("bignews_politics_embeddings.pkl", "rb") as f:
        politics_data = pickle.load(f)
    politics_embeddings = politics_data["test_embeddings"]
    # Map to query and doc embeddings
    test_indices_to_position = {idx: pos for pos, idx in enumerate(test_indices)}
    query_positions = [test_indices_to_position[idx] for idx in query_indices]
    doc_positions = [test_indices_to_position[idx] for idx in doc_indices]
    politics_embeddings_query = politics_embeddings[query_positions]
    politics_embeddings_doc = politics_embeddings[doc_positions]
    
    # 3. Load instructor embeddings
    print("Loading instructor embeddings...")
    with open("bignews_instructor_embeddings.pkl", "rb") as f:
        instructor_data = pickle.load(f)
    instructor_embeddings = instructor_data["test_embeddings"]
    instructor_embeddings_query = instructor_embeddings[query_positions]
    instructor_embeddings_doc = instructor_embeddings[doc_positions]
    
    # 4. Load inbedder embeddings
    print("Loading inbedder embeddings...")
    with open("bignews_inbedder_embeddings.pkl", "rb") as f:
        inbedder_data = pickle.load(f)
    inbedder_embeddings = inbedder_data["test_embeddings"]
    inbedder_embeddings_query = inbedder_embeddings[query_positions]
    inbedder_embeddings_doc = inbedder_embeddings[doc_positions]

    # Normalize all embeddings
    print("\nNormalizing embeddings...")
    def normalize_embeddings(embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    angle_embeddings_query = normalize_embeddings(angle_embeddings_query)
    angle_embeddings_doc = normalize_embeddings(angle_embeddings_doc)
    politics_embeddings_query = normalize_embeddings(politics_embeddings_query)
    politics_embeddings_doc = normalize_embeddings(politics_embeddings_doc)
    instructor_embeddings_query = normalize_embeddings(instructor_embeddings_query)
    instructor_embeddings_doc = normalize_embeddings(instructor_embeddings_doc)
    inbedder_embeddings_query = normalize_embeddings(inbedder_embeddings_query)
    inbedder_embeddings_doc = normalize_embeddings(inbedder_embeddings_doc)

    # Create indices for all methods
    print("\nCreating indices...")
    results = {}
    
    # Evaluate baselines first
    results.update(evaluate_random_retrieval(angle_embeddings_query, angle_embeddings_doc, doc_labels))
    results.update(evaluate_linear_retrieval(angle_embeddings_query, angle_embeddings_doc, doc_labels))
    
    # 1. Angle embeddings
    index_angle = pydkmips.Greedy(angle_embeddings_doc.shape[1], angle_embeddings_doc.shape[1])
    index_angle.add(angle_embeddings_doc, angle_embeddings_doc)
    results['angle'] = evaluate_method(index_angle, angle_embeddings_query, angle_embeddings_doc, 
                                     doc_labels, "Angle embeddings")
    
    # 2. Politics embeddings
    index_politics = pydkmips.Greedy(angle_embeddings_doc.shape[1], politics_embeddings_doc.shape[1])
    index_politics.add(angle_embeddings_doc, politics_embeddings_doc)
    results['politics'] = evaluate_method(index_politics, angle_embeddings_query, politics_embeddings_doc, 
                                        doc_labels, "Politics embeddings")
    
    # 3. Instructor embeddings
    index_instructor = pydkmips.Greedy(angle_embeddings_doc.shape[1], instructor_embeddings_doc.shape[1])
    index_instructor.add(angle_embeddings_doc, instructor_embeddings_doc)
    results['instructor'] = evaluate_method(index_instructor, angle_embeddings_query, instructor_embeddings_doc, 
                                          doc_labels, "Instructor embeddings")
    
    # 4. Inbedder embeddings
    index_inbedder = pydkmips.Greedy(angle_embeddings_doc.shape[1], inbedder_embeddings_doc.shape[1])
    index_inbedder.add(angle_embeddings_doc, inbedder_embeddings_doc)
    results['inbedder'] = evaluate_method(index_inbedder, angle_embeddings_query, inbedder_embeddings_doc, 
                                        doc_labels, "Inbedder embeddings")

    # Save results
    print("\nSaving results...")
    results['query_indices'] = query_indices.tolist()
    results['doc_indices'] = doc_indices.tolist()
    with open('retrieval_results_all.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults summary:")
    for method in ['random', 'linear', 'angle', 'politics', 'instructor', 'inbedder']:
        if method in ['random', 'linear']:
            score = results[method][0]
            print(f"\n{method.capitalize()} baseline:")
            print(f"Diversity score: {score[0]:.4f}")
            print(f"Similarity score: {score[1]:.4f}")
        else:
            best_c = max(results[method].items(), key=lambda x: x[1][0])[0]
            print(f"\n{method.capitalize()} embeddings:")
            print(f"Best diversity score at c={best_c:.2f}: {results[method][best_c][0]:.4f}")
            print(f"Corresponding similarity score: {results[method][best_c][1]:.4f}")

if __name__ == "__main__":
    main()
