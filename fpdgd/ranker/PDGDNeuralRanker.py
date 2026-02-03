from fpdgd.ranker.NeuralRanker import NeuralRanker
import numpy as np


class PDGDNeuralRanker(NeuralRanker):
    def __init__(self, num_features, learning_rate, tau=1, learning_rate_decay=1,
                 gradient_clip_norm: float = 0):
        super().__init__(num_features, learning_rate, learning_rate_decay)
        self.tau = tau
        self.gradient_clip_norm = gradient_clip_norm

        self.sensitivity = 0
        self.enable_noise = False

    def enable_noise_and_set_sensitivity(self, enable, sensitivity):
        self.sensitivity = sensitivity
        self.enable_noise = enable

    def get_query_result_list(self, dataset, query, random=False):
        feature_matrix = dataset.get_all_features_by_query(query)
        docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        n_docs = docid_list.shape[0]

        k = np.minimum(10, n_docs)

        doc_scores = self.get_scores(feature_matrix)
        doc_scores += 18 - np.amax(doc_scores)
        ranking = self._recursive_choice(np.copy(doc_scores),
                                         np.array([], dtype=np.int32),
                                         k,
                                         random)

        return ranking, doc_scores

    def _recursive_choice(self, scores, incomplete_ranking, k_left, random):
        n_docs = scores.shape[0]

        scores[incomplete_ranking] = np.amin(scores)
        scores += 18 - np.amax(scores)
        exp_scores = np.exp(scores / self.tau)

        exp_scores[incomplete_ranking] = 0
        probs = exp_scores / np.sum(exp_scores)

        safe_n = np.sum(probs > 10 ** (-4) / n_docs)
        safe_k = np.minimum(safe_n, k_left)

        if random:
            next_ranking = np.random.choice(np.arange(n_docs),
                                            replace=False,
                                            size=safe_k)
        else:
            next_ranking = np.random.choice(np.arange(n_docs),
                                            replace=False,
                                            p=probs,
                                            size=safe_k)

        ranking = np.concatenate((incomplete_ranking, next_ranking))
        k_left = k_left - safe_k

        if k_left > 0:
            return self._recursive_choice(scores, ranking, k_left, random)
        else:
            return ranking

    def update_to_clicks(self, click_label, ranking, doc_scores, feature_matrix, last_exam=None, return_gradients=False):
        if last_exam is None:
            clicks = np.array(click_label == 1)

            n_docs = ranking.shape[0]
            n_results = 10
            cur_k = np.minimum(n_docs, n_results)

            included = np.ones(cur_k, dtype=np.int32)

            if not clicks[-1]:
                included[1:] = np.cumsum(clicks[::-1])[:0:-1]

            neg_ind = np.where(np.logical_xor(clicks, included))[0]
            pos_ind = np.where(clicks)[0]
        else:
            if last_exam == 10:
                neg_ind = np.where(click_label[:last_exam] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]
            else:
                neg_ind = np.where(click_label[:last_exam + 1] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]

        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg

        if n_pairs == 0:
            if return_gradients:
                return None
            return

        pos_r_ind = ranking[pos_ind]
        neg_r_ind = ranking[neg_ind]

        pos_scores = doc_scores[pos_r_ind]
        neg_scores = doc_scores[neg_r_ind]

        log_pair_pos = np.tile(pos_scores, n_neg)
        log_pair_neg = np.repeat(neg_scores, n_pos)

        pair_trans = 18 - np.maximum(log_pair_pos, log_pair_neg)
        exp_pair_pos = np.exp(log_pair_pos + pair_trans)
        exp_pair_neg = np.exp(log_pair_neg + pair_trans)

        pair_denom = (exp_pair_pos + exp_pair_neg)
        pair_w = np.maximum(exp_pair_pos, exp_pair_neg)
        pair_w /= pair_denom
        pair_w /= pair_denom
        pair_w *= np.minimum(exp_pair_pos, exp_pair_neg)

        pair_w *= self._calculate_unbias_weights(pos_ind, neg_ind, doc_scores, ranking)

        reshaped = np.reshape(pair_w, (n_neg, n_pos))
        pos_w = np.sum(reshaped, axis=0)
        neg_w = -np.sum(reshaped, axis=1)

        all_w = np.concatenate([pos_w, neg_w])
        all_ind = np.concatenate([pos_r_ind, neg_r_ind])

        if return_gradients:
            return (all_ind, all_w, feature_matrix)
        else:
            self._update_to_documents(all_ind, all_w, feature_matrix)

    def _clip_gradient(self, dW1, db1, dW2, db2):
        if self.gradient_clip_norm > 0:
            flat = np.concatenate([dW1.ravel(), db1, dW2.ravel(), [db2]])
            grad_norm = np.linalg.norm(flat)
            if grad_norm > self.gradient_clip_norm:
                scale = self.gradient_clip_norm / grad_norm
                return dW1 * scale, db1 * scale, dW2 * scale, db2 * scale
        return dW1, db1, dW2, db2

    def _apply_param_update(self, dW1, db1, dW2, db2):
        dW1, db1, dW2, db2 = self._clip_gradient(dW1, db1, dW2, db2)

        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2
        self.learning_rate *= self.learning_rate_decay

        if self.enable_noise:
            for param in (self.W1, self.b1, self.W2, self.b2):
                norm = np.linalg.norm(param)
                if norm > self.sensitivity:
                    param *= self.sensitivity / norm

    def get_update_gradients(self, doc_ind, doc_weights, feature_matrix):
        return (doc_ind, doc_weights, feature_matrix)

    def update_to_gradients(self, gradients):
        if gradients is None:
            return
        doc_ind, doc_weights, feature_matrix = gradients
        dW1, db1, dW2, db2 = self._compute_param_gradients(
            doc_ind, doc_weights, feature_matrix)
        self._apply_param_update(dW1, db1, dW2, db2)

    def _update_to_documents(self, doc_ind, doc_weights, feature_matrix):
        dW1, db1, dW2, db2 = self._compute_param_gradients(
            doc_ind, doc_weights, feature_matrix)
        self._apply_param_update(dW1, db1, dW2, db2)

    def federated_averaging_weights(self, feedbacks):
        assert len(feedbacks) > 0
        feedbacks = [(m.gradient, m.parameters, m.n_interactions) for m in feedbacks]
        total_interactions = 0
        params = None
        for feedback in feedbacks:
            client_interactions = feedback[2]
            client_params = feedback[1]
            if params is None:
                params = {k: client_interactions * v for k, v in client_params.items()}
            else:
                for k in params:
                    params[k] += client_interactions * client_params[k]
            total_interactions += client_interactions
        for k in params:
            params[k] /= total_interactions
        self.assign_weights(params)

    def async_federated_averaging_weights(self, feedback):
        _, client_weights, _, client_staleness = feedback
        staleness_weight = 1.0 / (client_staleness + 1.0)
        current = self.get_current_weights()
        for k in current:
            current[k] = current[k] * (1 - staleness_weight) + client_weights[k] * staleness_weight
        self.assign_weights(current)

    def _calculate_unbias_weights(self, pos_ind, neg_ind, doc_scores, ranking):
        ranking_prob = self._calculate_observed_prob(pos_ind, neg_ind,
                                                     doc_scores, ranking)
        flipped_prob = self._calculate_flipped_prob(pos_ind, neg_ind,
                                                    doc_scores, ranking)
        return flipped_prob / (ranking_prob + flipped_prob)

    def _calculate_flipped_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)
        pair_i = np.arange(n_pairs)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        flipped_rankings = np.tile(ranking[None, :], [n_pairs, 1])
        flipped_rankings[pair_i, pos_pair_i] = ranking[neg_pair_i]
        flipped_rankings[pair_i, neg_pair_i] = ranking[pos_pair_i]

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        flipped_log = doc_scores[flipped_rankings]

        safe_log = np.tile(doc_scores[None, None, :], [n_pairs, n_results, 1])

        results_ij = np.tile(results_i[None, 1:], [n_pairs, 1])
        pair_ij = np.tile(pair_i[:, None], [1, n_results - 1])
        mask = np.zeros((n_pairs, n_results, n_docs))
        mask[pair_ij, results_ij, flipped_rankings[:, :-1]] = True
        mask = np.cumsum(mask, axis=1).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=2)
        safe_log -= safe_max[:, :, None] - 18
        flipped_log -= safe_max - 18
        flipped_exp = np.exp(flipped_log)

        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0
        safe_denom = np.sum(safe_exp, axis=2)
        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = (flipped_exp / safe_denom)[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def _calculate_observed_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        safe_log = np.tile(doc_scores[None, :], [n_results, 1])

        mask = np.zeros((n_results, n_docs))
        mask[results_i[1:], ranking[:-1]] = True
        mask = np.cumsum(mask, axis=0).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=1)
        safe_log -= safe_max[:, None] - 18
        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0

        ranking_log = doc_scores[ranking] - safe_max + 18
        ranking_exp = np.exp(ranking_log)

        safe_denom = np.sum(safe_exp, axis=1)
        ranking_prob = ranking_exp / safe_denom

        tiled_prob = np.tile(ranking_prob[None, :], [n_pairs, 1])

        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = tiled_prob[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def set_tau(self, tau):
        self.tau = tau

    def update_from_click(self, feature_matrix: np.ndarray, selected_idx: int, max_results: int = 10) -> bool:
        n_results = feature_matrix.shape[0]

        if n_results == 0 or selected_idx < 0 or selected_idx >= n_results:
            return False

        if selected_idx >= max_results:
            return False

        feature_matrix = feature_matrix[:max_results]
        n_results = feature_matrix.shape[0]

        scores = self.get_scores(feature_matrix)

        score_min, score_max = scores.min(), scores.max()
        score_range = score_max - score_min
        if score_range > 18:
            scores = (scores - score_min) / score_range * 18

        ranking = np.arange(n_results, dtype=np.int32)

        click_labels = np.zeros(n_results, dtype=np.float64)
        click_labels[selected_idx] = 1.0

        gradient = self.update_to_clicks(
            click_labels,
            ranking,
            scores,
            feature_matrix,
            return_gradients=True
        )
        self.update_to_gradients(gradient)
        return True
