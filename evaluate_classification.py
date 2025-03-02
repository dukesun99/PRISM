import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import torch
from tqdm import tqdm
import json
import os
import pickle
import glob

def evaluate_embeddings_with_svc(train_embeddings, test_embeddings, train_labels, test_labels, embedding_type, cache_dir="svc_results_cache_bignews", random_state=42):
    """
    Evaluate embeddings using SVC classifier with default parameters and caching
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{embedding_type}_results.json")
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"\nLoading cached results for {embedding_type} embeddings...")
        with open(cache_file, 'r') as f:
            results = json.load(f)
            
        # Print cached results
        print("\nCached Metrics:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro): {results['recall_macro']:.4f}")
        print(f"F1-macro: {results['f1_macro']:.4f}")
        print(f"F1-micro: {results['f1_micro']:.4f}")
        
        return results
    
    print(f"\nEvaluating {embedding_type} embeddings...")
    # Train SVC with default parameters
    svc = SVC(random_state=random_state)
    svc.fit(train_embeddings, train_labels)
    
    # Predict
    y_pred = svc.predict(test_embeddings)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro')
    f1_micro = f1_score(test_labels, y_pred, average='micro')
    
    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro)
    }
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-micro: {f1_micro:.4f}")
    
    # Also print classification report for additional information
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, y_pred))
    
    # Save results to cache
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def save_results(results, output_file):
    """Save evaluation results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Load original dataset for labels
    print("\nLoading original dataset...")
    with open("processed_data/bignewsbln_data.json", "r") as f:
        data = json.load(f)
        all_labels = np.array(data["labels"])
    
    # Find all embedding files
    embedding_files = glob.glob("processed_data/bignews_*_embeddings.pkl")
    print(f"\nFound {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"- {os.path.basename(f)}")
    
    # Load any embedding file to get test_indices
    print("\nLoading test indices...")
    with open(embedding_files[0], "rb") as f:
        data = pickle.load(f)
        original_test_indices = data["test_indices"]
    
    # assert all test indices are the same across all embedding files
    for f in embedding_files:
        with open(f, "rb") as f:
            data = pickle.load(f)
            assert np.array_equal(data["test_indices"], original_test_indices), f"Test indices in {f} do not match"
    
    # Further split test indices into train and test
    print("\nSplitting test indices into train and test...")
    # train_indices, test_indices, train_labels, test_labels = train_test_split(
    #     original_test_indices,
    #     all_labels[original_test_indices],
    #     test_size=0.1,
    #     random_state=42,
    #     stratify=all_labels[original_test_indices]
    # )
    # random sample 10% of the data indices in the range of the original test indices
    all_idxes = np.arange(len(original_test_indices))
    downstream_test_indices = np.random.choice(all_idxes, size=int(len(all_idxes) * 0.1), replace=False)
    downstream_train_indices = np.setdiff1d(all_idxes, downstream_test_indices)

    # Save the split indices for reproducibility
    split_file = "processed_data/bignews_svc_split_indices.pkl"
    print(f"\nSaving split indices to {split_file}")
    with open(split_file, "wb") as f:
        pickle.dump({
            "train_indices": downstream_train_indices,
            "test_indices": downstream_test_indices
        }, f)
    
    # Evaluate each embedding
    all_results = {}
    
    for embedding_file in embedding_files:
        embedding_type = os.path.basename(embedding_file).replace("bignews_", "").replace("_embeddings.pkl", "")
        print(f"\n{'='*50}")
        print(f"Processing {embedding_type} embeddings...")
        
        # Load embeddings
        with open(embedding_file, "rb") as f:
            data = pickle.load(f)
            all_embeddings = data["test_embeddings"]
        
        train_embeddings = np.array([all_embeddings[i] for i in downstream_train_indices])
        test_embeddings = np.array([all_embeddings[i] for i in downstream_test_indices])
        
        # Evaluate
        all_results[embedding_type] = evaluate_embeddings_with_svc(
            train_embeddings,
            test_embeddings,
            np.array([all_labels[original_test_indices[i]] for i in downstream_train_indices]),
            np.array([all_labels[original_test_indices[i]] for i in downstream_test_indices]),
            embedding_type
        )
    
    # Save all results
    save_results(all_results, 'processed_data/svc_evaluation_results_bignews.json')
    
    # Print summary results
    print("\nSummary of Results:")
    print("="*50)
    for embedding_type, results in all_results.items():
        print(f"\n{embedding_type.upper()} Embeddings:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro): {results['recall_macro']:.4f}")
        print(f"F1-macro: {results['f1_macro']:.4f}")
        print(f"F1-micro: {results['f1_micro']:.4f}")

if __name__ == '__main__':
    main() 