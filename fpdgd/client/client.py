from typing import NamedTuple
import numpy as np
import copy
from dart.evl_tool import query_ndcg_at_k
from fpdgd.data.LetorDataset import LetorDataset
from dart.dp import gamma_noise

# The message that each client send to the server:
# 1.updated parameters from client
# 2.volume of data that client use for each update
ClientMessage = NamedTuple("ClientMessage",[("gradient", np.ndarray), ("parameters", np.ndarray), ("n_interactions", int)])

# Metric values (ndcg@k, mrr@k) of each client averaged on whole batch (computed by relevance label)
ClientMetric = NamedTuple("ClientMetric", [("mean_mrr", float), ("mrr_list", list)])

class RankingClient:
    """
    emulate clients
    """
    def __init__(self, dataset: LetorDataset, init_model, seed: int, sensitivity, epsilon, enable_noise, n_clients, personalization_lambda=1.0):
        """
        :param dataset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param seed: random seed used to generate queries for client
        :param sensitivity: set global sensitivity of ranking model
        :param epsilon: privacy budget
        :param enable_noise: use differential privacy noise or not
        :param n_clients: number of clients
        :param personalization_lambda: weight for client model in weighted average (1.0 = no server update, 0.0 = full server update)
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.query_set = {str(qid) for qid in dataset.get_all_querys()}
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients
        self.personalization_lambda = personalization_lambda

    def update_model(self, server_model) -> None:
        """
        Update the client-side model using weighted average with server model
        :param server_model: The server model to blend with client model
        :return: None
        """
        if self.personalization_lambda == 0.0:
            # Complete server model adoption
            self.model = copy.deepcopy(server_model)
        elif self.personalization_lambda == 1.0:
            # No server model update (keep client model as is)
            pass
        else:
            # Weighted average: lambda * client_weights + (1 - lambda) * server_weights
            client_weights = self.model.get_current_weights()
            server_weights = server_model.get_current_weights()
            
            # Perform weighted averaging
            new_weights = (self.personalization_lambda * client_weights + 
                          (1.0 - self.personalization_lambda) * server_weights)
            
            # Set the new weights on the client model
            self.model.assign_weights(new_weights)

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        
        # Check if we have enough queries available
        if len(self.query_set) < n_interactions:
            print(f"Not enough queries available, returning None")
            # Not enough queries available, return None
            return None, None
        
        # Take the first n_interactions queries from the dataset (treat as queue)
        queries_to_process = self.query_set[:n_interactions]
        # Remove processed queries from the dataset
        self.query_set = self.query_set[n_interactions:]
        
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        for i in range(n_interactions): # run in batches
            qid = queries_to_process[i]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for j in range(0, ranking_result.shape[0]):
                docid = ranking_result[j]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[j] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = query_ndcg_at_k(self.dataset,ranking_result, qid, 10)
            per_interaction_client_ndcg.append(online_ndcg)

            g = self.model.update_to_clicks(ranking_relevance, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
                gradients += g

        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)

    def client_ranker_update_queries(self, query_ids: list[str], multi_update=True):
        """
        Process specific query IDs for ranking model training
        :param query_ids: List of specific query IDs to process
        :param multi_update: Whether to update model after each query or batch update
        :return: ClientMessage and ClientMetric
        """
        per_interaction_client_mrr = []
        
        # Check if all requested queries are available in the client's query set
        available_queries = set(self.query_set)
        requested_queries = set(query_ids)
        
        assert requested_queries.issubset(available_queries), f"Requested queries {requested_queries} are not available in the client's query set {available_queries}"
        
        # # Remove the processed queries from the client's query set
        # self.query_set = np.array([q for q in self.query_set if q not in requested_queries])
        
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        for qid in query_ids:
            ranking_result, scores = self.model.get_query_result_list(self.dataset, str(qid))
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for j in range(0, ranking_result.shape[0]):
                docid = ranking_result[j]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[j] = relevance
            # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance))

            g = self.model.update_to_clicks(ranking_relevance, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
                gradients += g

        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)
            updated_weights += noise

        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=len(query_ids)), ClientMetric(mean_mrr=mean_client_mrr, mrr_list=per_interaction_client_mrr)
