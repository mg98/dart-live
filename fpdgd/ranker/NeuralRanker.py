from fpdgd.ranker.AbstractRanker import AbstractRanker
import numpy as np
import copy

HIDDEN_SIZE = 32


class NeuralRanker(AbstractRanker):

    def __init__(self, num_features, learning_rate, learning_rate_decay=1, learning_rate_clip=0.01):
        super().__init__(num_features)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_clip = learning_rate_clip

        # Xavier initialization
        limit1 = np.sqrt(6.0 / (num_features + HIDDEN_SIZE))
        limit2 = np.sqrt(6.0 / (HIDDEN_SIZE + 1))

        self.W1 = np.random.uniform(-limit1, limit1, (num_features, HIDDEN_SIZE))
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.uniform(-limit2, limit2, (HIDDEN_SIZE, 1))
        self.b2 = np.zeros(1)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, features):
        z1 = features @ self.W1 + self.b1
        h = self._sigmoid(z1)
        scores = (h @ self.W2 + self.b2).ravel()
        return scores, h

    def get_scores(self, features):
        scores, _ = self._forward(features)
        return scores

    def _compute_param_gradients(self, doc_ind, doc_weights, feature_matrix):
        features = feature_matrix[doc_ind]
        z1 = features @ self.W1 + self.b1
        h = self._sigmoid(z1)

        w = doc_weights[:, None]

        dW2 = h.T @ w
        db2 = np.sum(doc_weights)

        dh = w @ self.W2.T
        dz1 = dh * h * (1.0 - h)

        dW1 = features.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return dW1, db1, dW2, db2

    def update(self, gradient):
        doc_ind, doc_weights, feature_matrix = gradient
        dW1, db1, dW2, db2 = self._compute_param_gradients(
            doc_ind, doc_weights, feature_matrix)

        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2

        if self.learning_rate > self.learning_rate_clip:
            self.learning_rate *= self.learning_rate_decay
        else:
            self.learning_rate = self.learning_rate_clip

    def assign_weights(self, weights):
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    def get_current_weights(self):
        return {
            'W1': copy.copy(self.W1),
            'b1': copy.copy(self.b1),
            'W2': copy.copy(self.W2),
            'b2': copy.copy(self.b2),
        }

    def get_query_result_list(self, dataset, query):
        docid_list = dataset.get_candidate_docids_by_query(query)
        feature_matrix = dataset.get_all_features_by_query(query)

        score_list = self.get_scores(feature_matrix)

        docid_score_list = zip(docid_list, score_list)
        docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)

        query_result_list = []
        for i in range(0, len(docid_list)):
            (docid, score) = docid_score_list[i]
            query_result_list.append(docid)
        return query_result_list

    def get_all_query_result_list(self, dataset, qids=None):
        if qids is None:
            qids = dataset.get_all_querys()

        query_result_list = {}

        for query in qids:
            docid_list = np.array(dataset.get_candidate_docids_by_query(query))
            docid_list = docid_list.reshape((len(docid_list), 1))
            feature_matrix = dataset.get_all_features_by_query(query)
            score_list = self.get_scores(feature_matrix)

            docid_score_list = np.column_stack((docid_list, score_list))
            docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

            query_result_list[query] = docid_score_list[:, 0]

        return query_result_list

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
