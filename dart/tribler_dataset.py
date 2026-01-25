"""
TriblerDataset class for loading Tribler LETOR format data
Handles 0-indexed features (unlike AOL which is 1-indexed)
"""
from fpdgd.data.LetorDataset import LetorDataset
import numpy as np
import math
from dart.common import FEATURES_GLOBAL

class TriblerDataset(LetorDataset):
    """
    Dataset class for Tribler data in LETOR format with 0-indexed features
    """

    def __init__(self, path, feature_size):
        """
        Initialize Tribler dataset

        Args:
            path: Path to LETOR format file
            feature_size: Number of features (31 for Tribler)
            binary_label: Threshold for binary relevance labels (0 = multi-level)
        """
        super().__init__(path, feature_size, True, 0, set(FEATURES_GLOBAL))

    def _load_data(self):
        """
        Load LETOR format data with 0-indexed features
        Overrides parent method to handle 0-indexed feature IDs
        """
        with open(self._path, "r") as fin:
            current_query = None
            for line in fin:
                cols = line.strip().split()
                query = cols[1].split(':')[1]

                if query == current_query:
                    docid = len(self._query_get_docids[query])
                    old_query = True
                else:
                    if current_query is not None and self._query_level_norm:
                        self._normalise(current_query)
                    old_query = False
                    docid = 0
                    current_query = query
                    self._docid_map[query] = {}
                    self._query_pos_docids[query] = []

                # Handle comments
                comments_part = line.split("#")
                if len(comments_part) == 2:
                    if query not in self._comments:
                        self._comments[query] = []
                    self._comments[query].append(comments_part[1].strip())

                # Parse relevance label
                relevance = float(cols[0])
                if relevance.is_integer():
                    relevance = int(relevance)
                if self._binary_label != 0:
                    relevance = 1 if relevance >= self._binary_label else 0

                # Parse features (0-indexed for Tribler)
                features = [0.0] * self._feature_size

                for i in range(2, len(cols)):
                    feature_id = cols[i].split(':')[0]

                    if not feature_id.isdigit():
                        if feature_id[0] == "#":
                            self._docid_map[query][docid] = cols[i][1:]
                        break

                    # Key difference: no subtraction for 0-indexed features
                    feature_id = int(feature_id)
                    feature_value = float(cols[i].split(':')[1])
                    if math.isnan(feature_value):
                        feature_value = 0.0

                    if feature_id < self._feature_size:
                        features[feature_id] = feature_value

                # Track positive documents
                if relevance > 0:
                    self._query_pos_docids[query].append(docid)

                # Store features and relevance
                if old_query:
                    self._query_docid_get_features[query][docid] = np.array(features)
                    self._query_get_docids[query].append(docid)
                    self._query_get_all_features[query] = np.vstack((self._query_get_all_features[query], features))
                    self._query_docid_get_rel[query][docid] = relevance
                    self._query_relevant_labels[query].append(relevance)
                else:
                    self._query_docid_get_features[query] = {docid: np.array(features)}
                    self._query_get_docids[query] = [docid]
                    self._query_get_all_features[query] = np.array([features])
                    self._query_docid_get_rel[query] = {docid: relevance}
                    self._query_relevant_labels[query] = [relevance]

            if self._query_level_norm:
                self._normalise(current_query)
