"""
Local PDGD training script (non-federated) for Tribler dataset
Trains a single PDGD ranker on all training data and evaluates on test/validation data.
"""
import numpy as np
import os
from tqdm import tqdm
from dart.tribler_dataset import TriblerDataset
from fpdgd.ranker.PDGDLinearRanker import PDGDLinearRanker
from fpdgd.client.federated_optimize import average_mrr_at_k
from dart.evl_tool import average_ndcg_at_k


def get_num_features(data_path: str) -> int:
    """Extract number of features from first line of LETOR-format data file."""
    with open(data_path, 'r') as f:
        first_line = f.readline()
    max_feature_idx = max(
        int(part.split(':')[0])
        for part in first_line.split()
        if ':' in part and part.split(':')[0].isdigit()
    )
    return max_feature_idx + 1


def train_local_pdgd(ranker: PDGDLinearRanker,
                     train_ds: TriblerDataset,
                     batch_size: int = 100,
                     epochs: int = 1):
    """
    Train PDGD ranker locally on training queries

    Args:
        ranker: PDGDLinearRanker instance
        train_ds: Training dataset
        batch_size: Number of queries to process before reporting progress
        epochs: Number of passes through training data
    """
    train_qids = [str(qid) for qid in train_ds.get_all_querys()]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle queries each epoch
        shuffled_qids = np.random.permutation(train_qids)

        epoch_mrrs = []

        for i in tqdm(range(0, len(shuffled_qids), batch_size), desc="Training"):
            batch_qids = shuffled_qids[i:i + batch_size]
            batch_mrrs = []

            for qid in batch_qids:
                # Get ranking from current model
                ranking_result, scores = ranker.get_query_result_list(train_ds, qid)

                # Get relevance labels (simulating clicks)
                ranking_relevance = np.zeros(ranking_result.shape[0])
                for j, docid in enumerate(ranking_result):
                    relevance = train_ds.get_relevance_label_by_query_and_docid(qid, docid)
                    ranking_relevance[j] = relevance

                # Compute online metrics
                mrr = compute_mrr(ranking_relevance)
                batch_mrrs.append(mrr)

                # Compute gradient and update
                feature_matrix = train_ds.get_all_features_by_query(qid)
                gradient = ranker.update_to_clicks(
                    ranking_relevance,
                    ranking_result,
                    scores,
                    feature_matrix,
                    return_gradients=True
                )
                ranker.update_to_gradients(gradient)

            epoch_mrrs.extend(batch_mrrs)

        avg_mrr = np.mean(epoch_mrrs)
        print(f"Epoch {epoch + 1} - Training MRR: {avg_mrr:.4f}")


def compute_mrr(relevance_labels: np.ndarray, k: int = 10000) -> float:
    """Compute MRR@k from relevance labels"""
    for i in range(min(k, len(relevance_labels))):
        if relevance_labels[i] > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(ranker: PDGDLinearRanker,
             eval_ds: TriblerDataset,
             k: int = 20,
             dataset_name: str = "test") -> dict:
    """
    Evaluate ranker on evaluation queries

    Args:
        ranker: Trained PDGDLinearRanker
        eval_ds: Evaluation dataset
        k: Cutoff for evaluation metrics
        dataset_name: Name for logging (e.g., "test", "validation")

    Returns:
        Dictionary with MRR@k and NDCG@k
    """
    eval_qids = [str(qid) for qid in eval_ds.get_all_querys()]

    # Generate rankings for all evaluation queries
    print(f"\nGenerating rankings for {len(eval_qids)} {dataset_name} queries...")
    all_result = ranker.get_all_query_result_list(eval_ds, eval_qids)

    # Compute metrics
    mrr = average_mrr_at_k(eval_ds, all_result, k)
    ndcg = average_ndcg_at_k(eval_ds, all_result, k)

    return {
        f'mrr@{k}': mrr,
        f'ndcg@{k}': ndcg
    }


def main():
    # Configuration
    lr = 0.1
    tau = 1.0
    batch_size = 1
    epochs = 5
    eval_k = 20

    # Data paths
    train_path = 'tribler_data/_normalized/train.txt'
    vali_path = 'tribler_data/_normalized/vali.txt'
    test_path = 'tribler_data/_normalized/test.txt'

    num_features = get_num_features(train_path)

    print("="*60)
    print("Local PDGD Training (Non-Federated) - Tribler Dataset")
    print("="*60)
    print(f"Learning rate: {lr}")
    print(f"Tau: {tau}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Evaluation k: {eval_k}")
    print(f"Features: {num_features}")
    print("="*60)

    # Load datasets (80/10/10 split already provided)
    print("\nLoading training dataset...")
    train_ds = TriblerDataset(
        path=train_path,
        feature_size=num_features,
    )
    print(f"Training queries: {len(train_ds.get_all_querys())}")

    print("\nLoading validation dataset...")
    vali_ds = TriblerDataset(
        path=vali_path,
        feature_size=num_features,
    )
    print(f"Validation queries: {len(vali_ds.get_all_querys())}")

    print("\nLoading test dataset...")
    test_ds = TriblerDataset(
        path=test_path,
        feature_size=num_features,
    )
    print(f"Test queries: {len(test_ds.get_all_querys())}")

    # Initialize ranker
    print("\nInitializing PDGD ranker...")
    ranker = PDGDLinearRanker(
        num_features=num_features,
        learning_rate=lr,
        tau=tau,
        learning_rate_decay=1.0,
        random_initial=True
    )

    # Train
    print("\nStarting training...")
    train_local_pdgd(
        ranker=ranker,
        train_ds=train_ds,
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluate on validation set
    print("\n" + "="*60)
    print("Validation Set Evaluation")
    print("="*60)
    vali_metrics = evaluate(
        ranker=ranker,
        eval_ds=vali_ds,
        k=eval_k,
        dataset_name="validation"
    )
    for metric_name, metric_value in vali_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    test_metrics = evaluate(
        ranker=ranker,
        eval_ds=test_ds,
        k=eval_k,
        dataset_name="test"
    )
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*60)

    # Save trained model
    print("\nSaving trained model...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pdgd_ranker.npy")
    np.save(model_path, ranker.get_current_weights())
    print(f"Model saved to: {model_path}")
    print("="*60)



if __name__ == '__main__':
    main()
