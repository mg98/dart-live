import sqlite3
import os
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import functools
import time
import re
import warnings
import difflib
import math
import json
from collections import Counter
from rank_bm25 import BM25Okapi
from contextlib import contextmanager
from sklearn.metrics import ndcg_score, average_precision_score
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

np.random.seed(42)

# =============================================================================
# FEATURE DEFINITIONS - Single source of truth
# =============================================================================
# Format: (name, normalization_strategy)
# Strategies:
#   - "per_query": skipped from global norm (user applies per-query normalization at inference)
#   - "bounded": already in [0,1], no normalization needed
#   - "minmax": MinMaxScaler
#   - "exp_decay": log1p + MinMaxScaler
#   - "robust": RobustScaler + MinMaxScaler
#
# The ORDER here determines the feature index. Rearrange freely.
# =============================================================================
FEATURE_DEFS = [
    # Term-based metrics (query-dependent, need per-query normalization)
    ("bm25", "per_query"),
    ("bm25_seeders", "per_query"),
    ("tf_min", "per_query"),
    ("tf_max", "per_query"),
    ("tf_mean", "per_query"),
    ("tf_sum", "per_query"),
    ("tf_variance", "per_query"),
    ("idf_min", "per_query"),
    ("idf_max", "per_query"),
    ("idf_mean", "per_query"),
    ("idf_sum", "per_query"),
    ("idf_variance", "per_query"),
    ("tf_idf_min", "per_query"),
    ("tf_idf_max", "per_query"),
    ("tf_idf_mean", "per_query"),
    ("tf_idf_sum", "per_query"),
    ("tf_idf_variance", "per_query"),
    ("cos_sim", "per_query"),
    ("covered_query_term_number", "per_query"),
    ("covered_query_term_ratio", "per_query"),
    # Document-level metrics
    ("char_len", "robust"),
    ("term_len", "robust"),
    ("total_query_terms", "robust"),
    ("exact_match", "bounded"),
    ("match_ratio", "bounded"),
    ("starts_with_query", "bounded"),
    ("starts_with_query_ratio", "bounded"),
    ("query_earliness_score", "bounded"),
    # Torrent metrics
    ("seeders", "exp_decay"),
    ("leechers", "exp_decay"),
    ("seeder_leecher_ratio", "exp_decay"),
    ("is_hot_release", "bounded"),
    ("is_stuck_torrent", "bounded"),
    ("entropy", "minmax"),
    ("diversity", "minmax"),
    ("content_age", "minmax"),
    ("temporal_gap", "minmax"),
    ("size", "exp_decay"),
    ("age", "exp_decay"),
    ("pos", "per_query"),
    ("uppercase_ratio", "bounded"),
    ("non_alnum_ratio", "bounded"),
]

# Derived constants (auto-generated from FEATURE_DEFS)
FEATURE_NAMES = [name for name, _ in FEATURE_DEFS]
FEATURE_NORMS = {name: norm for name, norm in FEATURE_DEFS}
NUM_FEATURES = len(FEATURE_DEFS)

def _indices_for(norm: str) -> list[int]:
    return [i for i, (_, n) in enumerate(FEATURE_DEFS) if n == norm]

FEATURES_PER_QUERY = _indices_for("per_query")
FEATURES_BOUNDED = _indices_for("bounded")
FEATURES_MINMAX = _indices_for("minmax")
FEATURES_EXP_DECAY = _indices_for("exp_decay")
FEATURES_ROBUST = _indices_for("robust")
FEATURES_SKIP = FEATURES_PER_QUERY + FEATURES_BOUNDED
FEATURES_GLOBAL = FEATURES_MINMAX + FEATURES_EXP_DECAY + FEATURES_ROBUST

def ranking_func(_func=None, *, shuffle=True):
    def _decorate(func):
        @functools.wraps(func)
        def wrapper(arg1, arg2=None, *args, **kwargs):
            if func.__name__ == 'tribler_rank':
                return func(arg1, arg2, *args, **kwargs)
                
            clicklogs = arg1
            activities = deepcopy(arg2) if arg2 is not None else deepcopy(arg1)

            if shuffle:
                for ua in clicklogs:
                    np.random.shuffle(ua.results)
                for ua in activities:
                    np.random.shuffle(ua.results)
                np.random.shuffle(clicklogs)
                np.random.shuffle(activities)

            return func(clicklogs, activities, *args, **kwargs)
        return wrapper
    
    if _func is not None and callable(_func):
        return _decorate(_func)
    
    return _decorate

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().split()

class TorrentInfo:
    title: str
    timestamp: float
    size: int

    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.timestamp = kwargs.get('timestamp', 0)
        self.size = kwargs.get('size', 0)
    
    def __repr__(self):
        return f"TorrentInfo(title='{self.title}', timestamp={self.timestamp}, size={self.size})"

    def __getstate__(self):
        return {
            'title': self.title,
            'timestamp': self.timestamp,
            'size': self.size
        }

    def __setstate__(self, state):
        self.title = state['title']
        self.timestamp = state['timestamp']
        self.size = state['size']

class UserActivityTorrent:
    infohash: str
    seeders: int
    leechers: int
    pos: int
    torrent_info: TorrentInfo

    def __init__(self, data):
        self.infohash = data['infohash']
        self.seeders = data['seeders'] 
        self.leechers = data['leechers']
        self.torrent_info = None

    def __str__(self):
        return f"Infohash: {self.infohash}, Pos: {self.pos}, Seeders: {self.seeders}, Leechers: {self.leechers}, Torrent Info: {self.torrent_info}"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.torrent_info, TorrentInfo):
            state['torrent_info'] = {
                'title': self.torrent_info.title,
                'timestamp': self.torrent_info.timestamp,
                'size': self.torrent_info.size,
            }
        return state

    def __setstate__(self, state):
        if isinstance(state, UserActivityTorrent):
            self.infohash = state.infohash
            self.seeders = state.seeders
            self.leechers = state.leechers
            self.pos = state.pos
            self.torrent_info = state.torrent_info
        else:
            self.__dict__.update(state)
            if isinstance(self.torrent_info, dict):
                self.torrent_info = TorrentInfo(**self.torrent_info)

class UserActivity:
    issuer: str
    query: str
    timestamp: int
    results: list[UserActivityTorrent]
    chosen_result: UserActivityTorrent

    def __init__(self, data: dict):
        self.issuer = data['issuer']
        self.query = data['query']
        self.timestamp = int(data['timestamp'] / 1000)
        self.results = []
        for pos, result in enumerate(data['results']):
            torrent = UserActivityTorrent(result)
            torrent.pos = pos
            self.results.append(torrent)
        self.chosen_result = self.results[data['chosen_index']]

    @property
    def chosen_index(self) -> int:
        """Returns the index of the chosen result in the results list"""
        for i, result in enumerate(self.results):
            if result.infohash == self.chosen_result.infohash:
                return i
        return -1

    def __repr__(self):
        return (f"UserActivity(issuer={self.issuer}, query={self.query}, "
                f"timestamp={self.timestamp}, chosen_result={self.chosen_result}, results=[{len(self.results)}  items...])")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['results'] = [result.__getstate__() for result in self.results]
        return state

    def __setstate__(self, state):
        # Extract the list of results from the state
        results_state = state.pop('results', [])
        chosen_result_state = state.pop('chosen_result', None)  # Extract chosen_result separately

        # Update the rest of the fields
        self.__dict__.update(state)

        # Convert each of the dicts in results_state back into a UserActivityTorrent
        self.results = []
        for r_state in results_state:
            torrent = UserActivityTorrent.__new__(UserActivityTorrent)
            torrent.__setstate__(r_state)
            self.results.append(torrent)

        # Handle chosen_result reconstruction
        if chosen_result_state is None:
            self.chosen_result = None
        else:
            # Create a new UserActivityTorrent for chosen_result
            chosen_torrent = UserActivityTorrent.__new__(UserActivityTorrent)
            chosen_torrent.__setstate__(chosen_result_state)
            
            # Find the matching torrent in results
            self.chosen_result = next(
                (t for t in self.results if t.infohash == chosen_torrent.infohash), 
                None
            )
            
            if self.chosen_result is None:
                print(f"Warning: Could not find matching torrent for chosen_result with infohash: {chosen_torrent.infohash}")


def fetch_torrent_infos(user_activities: list[UserActivity]):
    """Fetch torrent info for a list of UserActivityTorrent objects using batched SQL queries"""
    all_torrents = [t for ua in user_activities for t in ua.results]
    infohashes = list(set(t.infohash for t in all_torrents))
    
    BATCH_SIZE = 50000
    torrent_info_map = {}
    
    conn = sqlite3.connect(os.path.expanduser('./metadata.db'))
    cursor = conn.cursor()

    for i in range(0, len(infohashes), BATCH_SIZE):
        batch = infohashes[i:i + BATCH_SIZE]
        placeholders = ','.join(['?' for _ in batch])
        
        cursor.execute(f"""
            SELECT infohash_hex, title, timestamp/1000 as timestamp, size 
            FROM ChannelNode
            WHERE infohash_hex IN ({placeholders})
            """, batch)
        
        results = cursor.fetchall()
        
        for result in results:
            info = TorrentInfo()
            info.title = result[1]
            info.timestamp = result[2]
            info.size = result[3]
            torrent_info_map[result[0]] = info
    
    conn.close()
    
    # Update torrents in original user_activities
    found = 0
    not_found = 0
    for ua in user_activities:
        for torrent in ua.results:
            if torrent.infohash in torrent_info_map:
                torrent.torrent_info = torrent_info_map[torrent.infohash]
                found += 1

                if ua.chosen_result.infohash == torrent.infohash:
                    ua.chosen_result = torrent
            else:
                not_found += 1
    
    print(f'Found {found} torrents, skipped {not_found}')

    

class TFIDF:
    def __init__(self, corpus: Dict[str, str]):
        self.documents = list(corpus.values())
        self.doc_ids = list(corpus.keys())
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Precompute term counts for each document
        self.term_counts = [doc.split() for doc in self.documents]
        self.total_terms = [len(doc) for doc in self.term_counts]

    def get_tf_idf(self, doc_id: str, term: str) -> dict[str, float]:
        try:
            word_idx = list(self.feature_names).index(term)
        except ValueError:
            return { "tf": 0, "tf_idf": 0, "idf": 0 }
    
        doc_idx = self.doc_ids.index(doc_id)
        tf_idf = self.tfidf_matrix[doc_idx, word_idx]
        idf = self.vectorizer.idf_[word_idx]
        tf = tf_idf / idf if idf != 0 else 0
        
        return { "tf": tf, "tf_idf": tf_idf, "idf": idf }

    def get_cos_sim(self, doc_id: str, query: list[str]) -> float:
        """Compute cosine similarity between the query and a document."""
        query = ' '.join(query)
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        doc_idx = self.doc_ids.index(doc_id)
        document_vector = self.tfidf_matrix[doc_idx].toarray()[0]
        dot_product = np.dot(query_vector, document_vector)
        query_magnitude = np.linalg.norm(query_vector)
        document_magnitude = np.linalg.norm(document_vector)
        if query_magnitude == 0 or document_magnitude == 0:
            return 0.0
        return dot_product / (query_magnitude * document_magnitude)

@dataclass
class TermBasedMetrics:
    raw_text: str = ""
    bm25: float = 0.0
    tf_min: float = 0.0
    tf_max: float = 0.0
    tf_mean: float = 0.0
    tf_sum: float = 0.0
    tf_variance: float = 0.0
    idf_min: float = 0.0
    idf_max: float = 0.0
    idf_mean: float = 0.0
    idf_sum: float = 0.0
    idf_variance: float = 0.0
    tf_idf_min: float = 0.0
    tf_idf_max: float = 0.0
    tf_idf_mean: float = 0.0
    tf_idf_sum: float = 0.0
    tf_idf_variance: float = 0.0
    cos_sim: float = 0.0
    covered_query_term_number: int = 0
    covered_query_term_ratio: float = 0.0
    char_len: int = 0
    term_len: int = 0
    total_query_terms: int = 0
    exact_match: int = 0
    match_ratio: float = 0.0
    starts_with_query: int = 0
    starts_with_query_ratio: float = 0.0
    query_earliness_score: float = 0.0
    is_scene_compliant: int = 0

class Corpus:
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.tfidf = TFIDF(corpus)
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus.values()])

    def compute_features(self, doc_id: str, query_terms: list[str]) -> TermBasedMetrics:
        v = TermBasedMetrics()

        doc_index = list(self.corpus.keys()).index(doc_id)
        v.bm25 = self.bm25.get_batch_scores(query_terms, [doc_index])[0]

        tfidf_results = [self.tfidf.get_tf_idf(doc_id, term) for term in query_terms]

        v.tf_min = min(r["tf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_max = max(r["tf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_sum = sum(r["tf"] for r in tfidf_results)
        v.tf_mean = v.tf_sum / len(tfidf_results) if tfidf_results else 0.0

        v.idf_min = min(r["idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.idf_max = max(r["idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.idf_sum = sum(r["idf"] for r in tfidf_results)
        v.idf_mean = v.idf_sum / len(tfidf_results) if tfidf_results else 0.0

        v.tf_idf_min = min(r["tf_idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_idf_max = max(r["tf_idf"] for r in tfidf_results) if tfidf_results else 0.0
        v.tf_idf_sum = sum(r["tf_idf"] for r in tfidf_results)
        v.tf_idf_mean = v.tf_idf_sum / len(tfidf_results) if tfidf_results else 0.0
        
        v.tf_variance = sum((r["tf"] - v.tf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0
        v.idf_variance = sum((r["idf"] - v.idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0 
        v.tf_idf_variance = sum((r["tf_idf"] - v.tf_idf_mean) ** 2 for r in tfidf_results) / len(tfidf_results) if tfidf_results else 0.0

        v.cos_sim = self.tfidf.get_cos_sim(doc_id, query_terms)

        v.covered_query_term_number = sum(1 for r in tfidf_results if r["tf"] > 0)
        v.covered_query_term_ratio = v.covered_query_term_number / len(query_terms) if len(query_terms) > 0 else 0

        # Get document text from tfidf to calculate lengths
        doc_text = self.corpus[doc_id]
        v.raw_text = doc_text
        v.char_len = len(doc_text)
        v.term_len = len(tokenize(doc_text))
        
        # Boolean features
        document_terms = tokenize(doc_text)
        matched_terms = set(query_terms) & set(document_terms)
        match_count = len(matched_terms)
        v.total_query_terms = len(query_terms)
        query_str = ' '.join(query_terms)
        v.exact_match = difflib.SequenceMatcher(None, query_str, doc_text.lower()).ratio()
        v.match_ratio = match_count / v.total_query_terms if v.total_query_terms > 0 else 0

        v.starts_with_query = int(document_terms[:len(query_terms)] == query_terms)
        v.starts_with_query_ratio = sum(
            1 for i, qt in enumerate(query_terms)
            if i < len(document_terms) and qt == document_terms[i]
        ) / len(query_terms) if len(query_terms) > 0 else 0.0

        # Query earliness score: how early does any query term appear in the text?
        indices = [doc_text.lower().find(term.lower()) for term in query_terms]
        indices = [index for index in indices if index >= 0]
        if indices:
            v.query_earliness_score = float(1.0 / math.log(min(indices) + math.e))
        
        return v


class QueryDocumentRelationVector:
    title: TermBasedMetrics = TermBasedMetrics()
    seeders: int = 0
    leechers: int = 0
    age: float = 0.0
    pos: int = 0
    size: int = 0
    entropy: float = 0
    diversity: float = 0
    content_age: float = 0
    temporal_gap: float = 0
    

    def mask(self, masked_features: list[str]):
        for feature in masked_features:
            if hasattr(self, feature):
                if feature == "title":
                    setattr(self, feature, TermBasedMetrics())
                else:
                    setattr(self, feature, 0)
    
    @property
    def seeder_leecher_ratio(self):
        return self.seeders / max(self.leechers, 1)

    @property
    def bm25_seeders(self):
        return self.title.bm25 * self.seeders

    @property
    def is_hot_release(self):
        file_age_hours = self.age / 3600
        return 1.0 if (self.seeder_leecher_ratio < 0.5 and file_age_hours < 24) else 0.0

    @property
    def is_stuck_torrent(self):
        file_age_hours = self.age / 3600
        return 1.0 if (self.seeder_leecher_ratio < 0.1 and file_age_hours > 720) else 0.0
    
    @property
    def uppercase_ratio(self):
        """Detects 'LOUD' spam filenames like 'FREE DOWNLOAD'."""
        if not self.title.raw_text: return 0.0
        # Count uppercase chars (ignoring non-alpha)
        upper = sum(1 for c in self.title.raw_text if c.isupper())
        total = sum(1 for c in self.title.raw_text if c.isalpha())
        return upper / max(total, 1)

    @property
    def non_alnum_ratio(self):
        """Detects decorative spam like '*** Title ***'."""
        if not self.title.raw_text: return 0.0
        special = sum(1 for c in self.title.raw_text if not c.isalnum() and not c.isspace())
        return special / len(self.title.raw_text)

    @property
    def _feature_values(self) -> dict[str, float]:
        """
        Map feature names to their computed values.
        Add/remove entries here when modifying FEATURE_DEFS.
        """
        return {
            "bm25": self.title.bm25,
            "tf_min": self.title.tf_min,
            "tf_max": self.title.tf_max,
            "tf_mean": self.title.tf_mean,
            "tf_sum": self.title.tf_sum,
            "tf_variance": self.title.tf_variance,
            "idf_min": self.title.idf_min,
            "idf_max": self.title.idf_max,
            "idf_mean": self.title.idf_mean,
            "idf_sum": self.title.idf_sum,
            "idf_variance": self.title.idf_variance,
            "tf_idf_min": self.title.tf_idf_min,
            "tf_idf_max": self.title.tf_idf_max,
            "tf_idf_mean": self.title.tf_idf_mean,
            "tf_idf_sum": self.title.tf_idf_sum,
            "tf_idf_variance": self.title.tf_idf_variance,
            "cos_sim": self.title.cos_sim,
            "covered_query_term_number": self.title.covered_query_term_number,
            "covered_query_term_ratio": self.title.covered_query_term_ratio,
            "char_len": self.title.char_len,
            "term_len": self.title.term_len,
            "total_query_terms": self.title.total_query_terms,
            "exact_match": self.title.exact_match,
            "match_ratio": self.title.match_ratio,
            "starts_with_query": self.title.starts_with_query,
            "starts_with_query_ratio": self.title.starts_with_query_ratio,
            "query_earliness_score": self.title.query_earliness_score,
            "seeders": self.seeders,
            "leechers": self.leechers,
            "seeder_leecher_ratio": self.seeder_leecher_ratio,
            "bm25_seeders": self.bm25_seeders,
            "is_hot_release": self.is_hot_release,
            "is_stuck_torrent": self.is_stuck_torrent,
            "entropy": self.entropy,
            "diversity": self.diversity,
            "content_age": self.content_age,
            "temporal_gap": self.temporal_gap,
            "size": self.size,
            "age": self.age,
            "pos": self.pos,
            "uppercase_ratio": self.uppercase_ratio,
            "non_alnum_ratio": self.non_alnum_ratio,
        }

    @property
    def features(self) -> list[float]:
        """Feature vector in the order defined by FEATURE_NAMES."""
        vals = self._feature_values
        return [vals[name] for name in FEATURE_NAMES]

    def __str__(self):
        return ' '.join(f'{i}:{val}' for i, val in enumerate(self.features))

class ClickThroughRecord:
    rel: float
    qid: int
    qdr: QueryDocumentRelationVector

    def __init__(self, rel=0.0, qid=0, qdr=None): 
        self.rel = rel
        self.qid = qid
        self.qdr = qdr

    def to_dict(self):
        return {
            'rel': self.rel,
            'qid': self.qid,
            'qdr': self.qdr
        }

    def __str__(self):
        return f'{self.rel} qid:{self.qid} {self.qdr}'
    
def split_dataset_by_qids(records, train_ratio=0.8, val_ratio=0.1):
    """
    Split records into train/validation/test sets based on query IDs.
    
    Args:
        records: list containing the records
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        
    Returns:
        tuple of (train_records, val_records, test_records) as lists of ClickThroughRecord objects
    """
    records_df = pd.DataFrame([record.to_dict() for record in records])
    qids = records_df['qid'].unique()
    
    # Calculate split sizes
    n_qids = len(qids)
    train_size = int(train_ratio * n_qids)
    val_size = int(val_ratio * n_qids)
    
    # Split qids into train/val/test
    train_qids = qids[:train_size]
    val_qids = qids[train_size:train_size+val_size]
    test_qids = qids[train_size+val_size:]
    
    # Filter records by qid
    train_records_df = records_df[records_df['qid'].isin(train_qids)]
    val_records_df = records_df[records_df['qid'].isin(val_qids)]
    test_records_df = records_df[records_df['qid'].isin(test_qids)]
    
    # Convert to ClickThroughRecord objects
    train_records = [ClickThroughRecord(**record) for _, record in train_records_df.iterrows()]
    val_records = [ClickThroughRecord(**record) for _, record in val_records_df.iterrows()]
    test_records = [ClickThroughRecord(**record) for _, record in test_records_df.iterrows()]
    
    return train_records, val_records, test_records


class FeatureScaler:
    """
    Scaler for LTR features with different strategies per feature type.

    Strategies:
    - skip: Features already in [0,1] range
    - minmax: MinMaxScaler for bounded features
    - exp_decay: log1p transform followed by MinMaxScaler
    - robust: RobustScaler (median/IQR based) for everything else
    """

    def __init__(self):
        self.params: dict[int, dict] = {}
        self.n_features = NUM_FEATURES

    def fit(self, X: np.ndarray):
        """Fit scaler parameters on training data."""
        for i in range(X.shape[1]):
            col = X[:, i].copy()

            if i in FEATURES_PER_QUERY:
                self.params[i] = {"strategy": "skip", "reason": "per_query"}

            elif i in FEATURES_BOUNDED:
                self.params[i] = {"strategy": "skip", "reason": "bounded"}

            elif i in FEATURES_EXP_DECAY:
                # Apply log1p first, then fit MinMax on transformed values
                col_transformed = np.log1p(col)
                min_val = float(col_transformed.min())
                max_val = float(col_transformed.max())
                self.params[i] = {
                    "strategy": "exp_decay",
                    "min": min_val,
                    "max": max_val
                }

            elif i in FEATURES_MINMAX:
                min_val = float(col.min())
                max_val = float(col.max())
                self.params[i] = {
                    "strategy": "minmax",
                    "min": min_val,
                    "max": max_val
                }

            else:
                # RobustScaler: center by median, scale by IQR, then MinMax to [0,1]
                median = float(np.median(col))
                q1 = float(np.percentile(col, 25))
                q3 = float(np.percentile(col, 75))
                iqr = q3 - q1

                # Apply robust scaling to get min/max for final MinMax step
                if iqr > 0:
                    col_robust = (col - median) / iqr
                else:
                    col_robust = col - median

                robust_min = float(col_robust.min())
                robust_max = float(col_robust.max())

                self.params[i] = {
                    "strategy": "robust",
                    "median": median,
                    "iqr": iqr,
                    "robust_min": robust_min,
                    "robust_max": robust_max
                }

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted parameters."""
        X_out = np.zeros_like(X, dtype=np.float64)

        for i in range(X.shape[1]):
            col = X[:, i].copy()
            p = self.params[i]

            if p["strategy"] == "skip":
                X_out[:, i] = col

            elif p["strategy"] == "exp_decay":
                col = np.log1p(col)
                range_val = p["max"] - p["min"]
                if range_val > 0:
                    X_out[:, i] = (col - p["min"]) / range_val
                else:
                    X_out[:, i] = 0.0

            elif p["strategy"] == "minmax":
                range_val = p["max"] - p["min"]
                if range_val > 0:
                    X_out[:, i] = (col - p["min"]) / range_val
                else:
                    X_out[:, i] = 0.0

            else:  # robust + minmax
                iqr = p["iqr"]
                if iqr > 0:
                    col = (col - p["median"]) / iqr
                else:
                    col = col - p["median"]

                # Apply MinMax on robust-scaled values
                range_val = p["robust_max"] - p["robust_min"]
                if range_val > 0:
                    X_out[:, i] = (col - p["robust_min"]) / range_val
                else:
                    X_out[:, i] = 0.0

        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def transform_single(self, features: list[float]) -> list[float]:
        """Transform a single feature vector (for inference)."""
        arr = np.array(features).reshape(1, -1)
        return self.transform(arr)[0].tolist()

    def save(self, path: str):
        """Save scaler parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load scaler parameters from JSON file."""
        scaler = cls()
        with open(path, 'r') as f:
            params = json.load(f)
        # JSON keys are strings, convert back to int
        scaler.params = {int(k): v for k, v in params.items()}
        return scaler


def normalize_features(ds_path: str):
    """
    Normalize features in the dataset using FeatureScaler.

    Applies different strategies per feature type:
    - Skip: features already in [0,1]
    - MinMaxScaler: entropy, diversity, content_age, temporal_gap, size
    - ExpDecay + MinMax: seeder_leecher_ratio, age, pos
    - RobustScaler: all other features

    Exports scaler parameters to feature_scaler.json for inference use.
    """
    x_train, y_train, query_ids_train = load_svmlight_file(os.path.join(ds_path, "train.txt"), query_id=True)
    x_test, y_test, query_ids_test = load_svmlight_file(os.path.join(ds_path, "test.txt"), query_id=True)
    x_vali, y_vali, query_ids_vali = load_svmlight_file(os.path.join(ds_path, "vali.txt"), query_id=True)

    X_train = x_train.toarray()
    X_test = x_test.toarray()
    X_vali = x_vali.toarray()

    # Fit scaler on training data only
    scaler = FeatureScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    X_vali_normalized = scaler.transform(X_vali)

    # Save scaler parameters for inference
    scaler_path = os.path.join(ds_path, "feature_scaler.json")
    scaler.save(scaler_path)
    print(f"Saved scaler parameters to {scaler_path}")

    # Write normalized datasets
    ds_normalized_path = os.path.join(ds_path, "_normalized")
    os.makedirs(ds_normalized_path, exist_ok=True)

    train_normalized_path = os.path.join(ds_normalized_path, "train.txt")
    with open(train_normalized_path, "w"):
        dump_svmlight_file(X_train_normalized, y_train, train_normalized_path, query_id=query_ids_train)

    test_normalized_path = os.path.join(ds_normalized_path, "test.txt")
    with open(test_normalized_path, "w"):
        dump_svmlight_file(X_test_normalized, y_test, test_normalized_path, query_id=query_ids_test)

    vali_normalized_path = os.path.join(ds_normalized_path, "vali.txt")
    with open(vali_normalized_path, "w"):
        dump_svmlight_file(X_vali_normalized, y_vali, vali_normalized_path, query_id=query_ids_vali)

def calc_mrr(ua: UserActivity) -> float:
    """Calculate Mean Reciprocal Rank for a single user activity"""
    return 1.0 / (ua.chosen_index + 1)

def mean_mrr(user_activities: list[UserActivity]) -> float:
    return np.mean([calc_mrr(ua) for ua in user_activities])

def mean_recall(user_activities: list[UserActivity], k: int) -> float:
    recalls = sum(1 if ua.chosen_index <= k else 0 for ua in user_activities)
    return recalls / len(user_activities)

def calc_ndcg(ua: UserActivity, k=None) -> float:
    """
    Calculate nDCG@k for a single user activity
    
    Args:
        ua: user activity
        k: number of top results to consider. If None, considers all results
    """
    true_relevance = [1 if res.infohash == ua.chosen_result.infohash else 0 for res in ua.results]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]
    
    if k is not None:
        true_relevance = true_relevance[:k]
        predicted_relevance = predicted_relevance[:k]
    
    return ndcg_score([true_relevance], [predicted_relevance])

def mean_ndcg(user_activities: list[UserActivity], k=None) -> float:
    return np.round(np.mean([calc_ndcg(ua, k) for ua in user_activities]), 3)

def calc_map(ua: UserActivity, k=None) -> float:
    """Calculate MAP for a single user activity"""
    true_relevance = [1 if res.infohash == ua.chosen_result.infohash else 0 for res in ua.results]
    predicted_relevance = [1/np.log2(i+2) for i in range(len(ua.results))]
    
    if k is not None:
        true_relevance = true_relevance[:k]
        predicted_relevance = predicted_relevance[:k]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return average_precision_score(true_relevance, predicted_relevance)

def mean_map(user_activities: list[UserActivity], k=None) -> float:
    return np.round(np.mean([calc_map(ua, k) for ua in user_activities]), 3)


@contextmanager
def timing():
    """Context manager that measures execution time and formats output as XhYmZs"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"Time taken: {hours}h{minutes}m{seconds}s")

def calculate_shannon_entropy(text):
    """
    Calculates the Shannon Entropy of a string.
    Formula: H(X) = - sum( p(x) * log2(p(x)) )
    """
    if not text:
        return 0
    
    # Calculate the frequency of each character
    freqs = Counter(text)
    total_len = len(text)
    
    # Calculate entropy
    entropy = 0
    for count in freqs.values():
        p_x = count / total_len
        entropy += - p_x * math.log2(p_x)
        
    return entropy

def calculate_trigram_diversity(text):
    """
    Measures repetitiveness. 
    Returns ratio of unique trigrams (3-char sequences) to total trigrams.
    Spam usually repeats words ('free', 'best'), lowering this score.
    """
    if len(text) < 3:
        return 1.0
        
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    unique_trigrams = len(set(trigrams))
    
    # 1.0 = Highly complex/unique
    # 0.5 or lower = Highly repetitive (Spam)
    return unique_trigrams / len(trigrams)

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