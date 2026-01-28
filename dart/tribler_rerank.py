"""
Tribler Search Reranking Module

Provides feature extraction for reranking search results using a trained PDGD model.
Extracts 40 features matching the training dataset from utils/ltr_helper.py.
"""
import re
import time
from datetime import datetime
import numpy as np
from dart.common import (
    tokenize, Corpus, QueryDocumentRelationVector, FeatureScaler,
    calculate_shannon_entropy, calculate_trigram_diversity
)


def encode_search_results(
    results: list[dict],
    query: str,
    scaler: FeatureScaler,
    current_time = None
) -> np.ndarray:
    """
    Encode search results into feature matrix for reranking.

    Uses the same feature extraction logic as LTRDatasetMaker.process_row
    to produce normalized features compatible with the trained model.

    Args:
        results: List of search result dicts with keys:
            - infohash, name, num_seeders, num_leechers, created, size, tags (optional)
        query: Original query string
        scaler: FeatureScaler for normalizing features
        current_time: Current timestamp (defaults to time.time())

    Returns:
        NumPy array of shape (n_results, N) containing normalized feature vectors
    """
    if not results:
        return np.array([])

    if current_time is None:
        current_time = time.time()

    query_terms = tokenize(query)

    # Build local corpus from results
    local_corpus = {r['infohash']: r['name'].lower() for r in results}
    corpus = Corpus(local_corpus)

    feature_matrix = []
    for pos, result in enumerate(results):
        title = result['name']
        created = result.get('created', current_time)

        v = QueryDocumentRelationVector()
        v.title = corpus.compute_features(result['infohash'], query_terms)
        v.seeders = result.get('num_seeders') or 0
        v.leechers = result.get('num_leechers') or 0
        v.age = current_time - created
        v.pos = pos
        v.tag_count = len(result.get('tags') or [])
        v.size = result.get('size') or 0
        v.entropy = calculate_shannon_entropy(title)
        v.diversity = calculate_trigram_diversity(title)

        # Extract year from title for content_age and temporal_gap
        match = re.search(r'\b(19|20)\d{2}\b', title)
        if match:
            title_year = int(match.group(0))
            upload_year = datetime.fromtimestamp(created).year
            current_year = datetime.fromtimestamp(current_time).year
            content_age = current_year - title_year
            if content_age > 0:
                v.content_age = np.log1p(content_age)
            temporal_gap = title_year - upload_year
            if temporal_gap > 0:
                v.temporal_gap = np.log1p(temporal_gap)

        feature_matrix.append(v.features)

    return scaler.transform(np.array(feature_matrix))
