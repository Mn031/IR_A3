"""
method_a_zone_scoring.py
========================
Method A: Boolean Weighted Zone Scoring with Grid Search Optimization.
Learns real-valued base weights for four document fields (Title, Author, Bibliography, Body)
using Boolean matching scores as input and human relevance judgments as ground truth.
Fully compatible with PyTerrier pipelines via pt.Transformer.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
import pyterrier as pt

from data_parser import Query, Qrel, tokenize
from indexer import CranfieldCorpusStats


class BooleanWeightedZoneScorer(pt.Transformer):
    """
    PyTerrier-compatible Transformer that scores and ranks documents using
    learned Boolean Weighted Zone Scoring across four zones:
      - Title
      - Author
      - Bibliography
      - Body text
    """

    def __init__(
        self,
        stats: CranfieldCorpusStats,
        weights: np.ndarray,
        zone_names: Optional[List[str]] = None,
        scoring_mode: str = "fraction"
    ):
        """
        :param stats: CranfieldCorpusStats instance holding preprocessed corpus tokens.
        :param weights: 1D array of shape (4,) containing weights for [Title, Author, Bibliography, Body].
        :param scoring_mode: 'fraction' (proportion of query terms in zone) or 'binary' (1 if any term in zone).
        """
        super().__init__()
        self.stats = stats
        self.weights = np.array(weights, dtype=np.float32)
        self.zone_names = zone_names or ['title', 'author', 'bibliography', 'body']
        self.scoring_mode = scoring_mode
        self.doc_ids = np.array(self.stats.doc_ids)

    def _compute_query_matrix(self, query_tokens: List[str]) -> np.ndarray:
        """
        Computes the (N, 4) Boolean matching matrix for a given query against all documents.
        """
        q_set = set(query_tokens)
        q_len = len(q_set) if len(q_set) > 0 else 1.0
        n_docs = len(self.doc_ids)
        matrix = np.zeros((n_docs, 4), dtype=np.float32)

        for i, doc_id in enumerate(self.doc_ids):
            zones = self.stats.zone_tokens[doc_id]
            for j, z in enumerate(self.zone_names):
                overlap = len(q_set.intersection(zones[z]))
                if self.scoring_mode == "fraction":
                    matrix[i, j] = overlap / q_len
                else:  # binary
                    matrix[i, j] = 1.0 if overlap > 0 else 0.0

        return matrix

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input topics/queries DataFrame into ranked document results.
        Supports both direct topic retrieval (scoring all collection documents)
        and candidate re-ranking if 'docno' is already present.
        """
        rows = []

        # Case 1: Topic retrieval (ranking full collection for each query)
        if 'docno' not in topics_or_res.columns:
            for _, row in topics_or_res.iterrows():
                qid = str(row['qid'])
                query_text = row['query']
                tokens = tokenize(query_text, remove_stopwords=True)
                if not tokens:
                    tokens = tokenize(query_text, remove_stopwords=False)

                M = self._compute_query_matrix(tokens)
                scores = M @ self.weights

                # Rank all documents with score > 0 first, followed by score == 0
                ranked_indices = np.argsort(-scores)

                for rank, idx in enumerate(ranked_indices):
                    doc_id = self.doc_ids[idx]
                    sc = float(scores[idx])
                    rows.append({
                        'qid': qid,
                        'docno': doc_id,
                        'score': sc,
                        'rank': rank
                    })

            res_df = pd.DataFrame(rows)
            return res_df

        # Case 2: Re-ranking candidate documents
        for qid, qdf in topics_or_res.groupby('qid'):
            q_text = qdf.iloc[0]['query'] if 'query' in qdf.columns else ""
            q_tokens = tokenize(q_text, remove_stopwords=True)
            q_set = set(q_tokens)
            q_len = len(q_set) if len(q_set) > 0 else 1.0

            cand_docs = qdf['docno'].tolist()
            c_scores = []
            for doc_id in cand_docs:
                zones = self.stats.zone_tokens.get(doc_id, {'title': set(), 'author': set(), 'bibliography': set(), 'body': set()})
                z_scores = np.zeros(4, dtype=np.float32)
                for j, z in enumerate(self.zone_names):
                    overlap = len(q_set.intersection(zones[z]))
                    z_scores[j] = (overlap / q_len) if self.scoring_mode == "fraction" else (1.0 if overlap > 0 else 0.0)
                c_scores.append(float(z_scores @ self.weights))

            order = np.argsort(-np.array(c_scores))
            for rank, idx in enumerate(order):
                rows.append({
                    'qid': qid,
                    'docno': cand_docs[idx],
                    'score': c_scores[idx],
                    'rank': rank
                })

        return pd.DataFrame(rows)


class GridSearchZoneOptimizer:
    """
    Grid search algorithm to learn optimal base weights for the 4 document zones
    using human relevance judgments on the training set.
    """

    def __init__(
        self,
        stats: CranfieldCorpusStats,
        step: float = 0.05,
        target_metric: str = "f1_10",
        scoring_mode: str = "fraction"
    ):
        """
        :param stats: CranfieldCorpusStats instance.
        :param step: Grid resolution step (e.g., 0.05 yields 1,771 simplex weight combinations).
        :param target_metric: Optimization objective ('f1_10', 'p_10', or 'map').
        :param scoring_mode: 'fraction' or 'binary'.
        """
        self.stats = stats
        self.step = step
        self.target_metric = target_metric
        self.scoring_mode = scoring_mode
        self.zone_names = ['title', 'author', 'bibliography', 'body']
        self.doc_ids = np.array(self.stats.doc_ids)

    def generate_grid_weights(self) -> List[np.ndarray]:
        """
        Generates all non-negative weights (w1, w2, w3, w4) such that sum(wi) == 1.0.
        """
        weights: List[np.ndarray] = []
        steps = int(round(1.0 / self.step)) + 1

        for w1 in np.linspace(0, 1, steps):
            for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) / self.step)) + 1):
                for w3 in np.linspace(0, 1 - w1 - w2, int(round((1 - w1 - w2) / self.step)) + 1):
                    w4 = max(0.0, round(1.0 - w1 - w2 - w3, 4))
                    weights.append(np.array([w1, w2, w3, w4], dtype=np.float32))

        return weights

    def fit(
        self,
        train_queries: List[Query],
        train_qrels: List[Qrel]
    ) -> Tuple[np.ndarray, float, Dict[str, any]]:
        """
        Performs grid search to find the optimal weights that maximize the chosen metric on training queries.
        Returns: (best_weights, best_score, history_info)
        """
        # Map ground truth relevance per query
        qrels_map: Dict[str, Set[str]] = {}
        for qr in train_qrels:
            if qr.label == 1:
                if qr.qid not in qrels_map:
                    qrels_map[qr.qid] = set()
                qrels_map[qr.qid].add(qr.docno)

        # Precompute (N, 4) matching matrix for each training query
        q_len_docs = len(self.doc_ids)
        matrices: List[np.ndarray] = []
        train_rels: List[Set[str]] = []

        scorer = BooleanWeightedZoneScorer(self.stats, np.ones(4), scoring_mode=self.scoring_mode)

        for q in train_queries:
            q_toks = q.tokens if q.tokens else tokenize(q.query, remove_stopwords=False)
            M = scorer._compute_query_matrix(q_toks)
            matrices.append(M)
            train_rels.append(qrels_map.get(q.qid, set()))

        candidate_weights = self.generate_grid_weights()

        best_score = -1.0
        best_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        for w in candidate_weights:
            metric_scores: List[float] = []

            for i, M in enumerate(matrices):
                scores = M @ w
                rel = train_rels[i]
                n_rel = len(rel)
                if n_rel == 0:
                    continue

                # Top 10 indices
                top10_idx = np.argpartition(-scores, 10)[:10]
                top10_docs = set(self.doc_ids[top10_idx])
                hits = len(top10_docs.intersection(rel))

                p10 = hits / 10.0
                r10 = hits / n_rel
                f1_10 = (2 * p10 * r10) / (p10 + r10) if (p10 + r10) > 0 else 0.0

                if self.target_metric == "f1_10":
                    metric_scores.append(f1_10)
                elif self.target_metric == "p_10":
                    metric_scores.append(p10)
                elif self.target_metric == "map":
                    # Order top 100 for MAP approximation
                    top_idx = np.argsort(-scores)[:50]
                    ap = 0.0
                    h = 0
                    for rank, idx in enumerate(top_idx):
                        if self.doc_ids[idx] in rel:
                            h += 1
                            ap += h / (rank + 1)
                    metric_scores.append(ap / n_rel)

            avg_score = float(np.mean(metric_scores)) if metric_scores else 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_weights = w

        history = {
            'total_candidates': len(candidate_weights),
            'target_metric': self.target_metric,
            'scoring_mode': self.scoring_mode,
            'best_weights': best_weights,
            'best_training_score': best_score
        }

        return best_weights, best_score, history

