"""
method_b_svm_ltr.py
===================
Method B: Learning Real-Valued Weights using Support Vector Machines (Linear SVM).
Extracts 22 query-document features via PyTerrier-compatible pipelines:
  1. query term number
  2. query term ratio
  3. stream length
  4. idf
  5. sum of term frequency
  6. min of term frequency
  7. max of term frequency
  8. mean of term frequency
  9. variance of term frequency
  10. sum of stream length normalized term frequency
  11. min of stream length normalized term frequency
  12. max of stream length normalized term frequency
  13. mean of stream length normalized term frequency
  14. variance of stream length normalized term frequency
  15. sum of tf*idf
  16. min of tf*idf
  17. max of tf*idf
  18. mean of tf*idf
  19. variance of tf*idf
  20. boolean model
  21. vector space model
  22. BM25
Learns real-valued weights using Linear SVM and human relevance judgments.
"""

from typing import List, Dict, Set, Tuple, Optional
import math
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pyterrier as pt

from data_parser import Query, Qrel, tokenize
from indexer import CranfieldCorpusStats


FEATURE_NAMES: List[str] = [
    "query term number",
    "query term ratio",
    "stream length",
    "idf",
    "sum of term frequency",
    "min of term frequency",
    "max of term frequency",
    "mean of term frequency",
    "variance of term frequency",
    "sum of stream length normalized term frequency",
    "min of stream length normalized term frequency",
    "max of stream length normalized term frequency",
    "mean of stream length normalized term frequency",
    "variance of stream length normalized term frequency",
    "sum of tf*idf",
    "min of tf*idf",
    "max of tf*idf",
    "mean of tf*idf",
    "variance of tf*idf",
    "boolean model",
    "vector space model",
    "BM25"
]


class FeatureExtractor:
    """
    Computes all 22 required features for query-document pairs.
    """

    def __init__(self, stats: CranfieldCorpusStats, k1: float = 1.2, b: float = 0.75):
        self.stats = stats
        self.k1 = k1
        self.b = b

    def extract_features(self, query_tokens: List[str], doc_id: str) -> np.ndarray:
        """
        Extracts the exact 22 features for a single (query, document) pair.
        Returns a 1D NumPy array of shape (22,).
        """
        q_len = len(query_tokens)
        if q_len == 0:
            return np.zeros(22, dtype=np.float32)

        doc_tf = self.stats.doc_tfs.get(doc_id, Counter())
        doc_len = self.stats.doc_lens.get(doc_id, 1)

        # Matching terms and frequencies
        matching_terms = [t for t in query_tokens if t in doc_tf]
        unique_matching = set(matching_terms)
        tfs = [doc_tf.get(t, 0) for t in query_tokens]
        matching_tfs = [doc_tf[t] for t in matching_terms]

        # Normalized TFs
        norm_tfs = [tf / doc_len for tf in tfs]
        matching_norm_tfs = [doc_tf[t] / doc_len for t in matching_terms]

        # IDFs and TF-IDF
        idfs = [self.stats.get_idf(t) for t in query_tokens]
        tfidfs = [tfs[i] * idfs[i] for i in range(q_len)]
        matching_tfidfs = [doc_tf[t] * self.stats.get_idf(t) for t in matching_terms]

        # 1. query term number
        f1 = float(q_len)
        # 2. query term ratio
        f2 = float(len(matching_terms) / q_len)
        # 3. stream length
        f3 = float(doc_len)
        # 4. idf (sum of IDF for distinct matching query terms)
        f4 = float(sum(self.stats.get_idf(t) for t in unique_matching))

        # 5. sum of term frequency
        f5 = float(sum(tfs))
        # 6. min of term frequency (among matching terms, or 0)
        f6 = float(min(matching_tfs)) if matching_tfs else 0.0
        # 7. max of term frequency
        f7 = float(max(tfs)) if tfs else 0.0
        # 8. mean of term frequency
        f8 = float(np.mean(tfs)) if tfs else 0.0
        # 9. variance of term frequency
        f9 = float(np.var(tfs)) if tfs else 0.0

        # 10. sum of stream length normalized term frequency
        f10 = float(sum(norm_tfs))
        # 11. min of stream length normalized term frequency
        f11 = float(min(matching_norm_tfs)) if matching_norm_tfs else 0.0
        # 12. max of stream length normalized term frequency
        f12 = float(max(norm_tfs)) if norm_tfs else 0.0
        # 13. mean of stream length normalized term frequency
        f13 = float(np.mean(norm_tfs)) if norm_tfs else 0.0
        # 14. variance of stream length normalized term frequency
        f14 = float(np.var(norm_tfs)) if norm_tfs else 0.0

        # 15. sum of tf*idf
        f15 = float(sum(tfidfs))
        # 16. min of tf*idf
        f16 = float(min(matching_tfidfs)) if matching_tfidfs else 0.0
        # 17. max of tf*idf
        f17 = float(max(tfidfs)) if tfidfs else 0.0
        # 18. mean of tf*idf
        f18 = float(np.mean(tfidfs)) if tfidfs else 0.0
        # 19. variance of tf*idf
        f19 = float(np.var(tfidfs)) if tfidfs else 0.0

        # 20. boolean model (coordinate term overlap)
        unique_matching = set(matching_terms)
        f20 = float(len(unique_matching))

        # 21. vector space model (cosine similarity)
        q_tf = Counter(query_tokens)
        q_norm = math.sqrt(sum((q_tf[t] * self.stats.get_idf(t)) ** 2 for t in q_tf))
        d_norm = self.stats.doc_vsm_norms.get(doc_id, 1.0)
        dot_prod = sum((q_tf[t] * self.stats.get_idf(t)) * (doc_tf[t] * self.stats.get_idf(t)) for t in unique_matching)
        f21 = float(dot_prod / (q_norm * d_norm)) if (q_norm > 0 and d_norm > 0) else 0.0
        # Guard against minor floating point precision overshooting 1.0
        f21 = min(1.0, max(0.0, f21))

        # 22. BM25 score
        k1 = self.k1
        b = self.b
        avgdl = self.stats.avgdl
        f22 = float(sum(
            self.stats.get_idf(t) * (doc_tf[t] * (k1 + 1.0)) /
            (doc_tf[t] + k1 * (1.0 - b + b * (doc_len / avgdl)))
            for t in unique_matching
        ))

        return np.array([
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22
        ], dtype=np.float32)

    def get_pyterrier_transformer(self) -> pt.Transformer:
        """
        Returns a PyTerrier transformer that computes the 'features' column
        for candidate documents in a DataFrame.
        """
        def _calc_row_features(row):
            query_text = row['query'] if 'query' in row else ""
            q_toks = tokenize(query_text, remove_stopwords=True)
            if not q_toks:
                q_toks = tokenize(query_text, remove_stopwords=False)
            doc_id = str(row['docno'])
            return self.extract_features(q_toks, doc_id)

        return pt.apply.doc_features(_calc_row_features)


class LinearSVMScorer(BaseEstimator):
    """
    Scikit-learn compatible Linear SVM estimator that outputs continuous real-valued
    decision scores (w * x + b) for PyTerrier's LTR re-ranking pipeline.
    """

    def __init__(self, C: float = 0.01, max_iter: int = 10000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LinearSVC(
            C=self.C,
            class_weight='balanced',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        self.raw_weights: Optional[np.ndarray] = None
        self.raw_intercept: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits StandardScaler and LinearSVC on training features and relevance labels.
        Derives the effective unscaled raw weights and intercept.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Calculate effective unscaled weights: w_raw_j = w_std_j / std_j
        scale = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
        self.raw_weights = self.model.coef_[0] / scale
        self.raw_intercept = float(
            self.model.intercept_[0] - np.sum(self.model.coef_[0] * self.scaler.mean_ / scale)
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Outputs continuous real-valued decision scores for document ranking.
        """
        if self.raw_weights is not None:
            return X @ self.raw_weights + self.raw_intercept
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    @classmethod
    def from_pretrained_weights(cls, raw_weights: np.ndarray, raw_intercept: float):
        """
        Constructs a LinearSVMScorer directly from pre-learned weights and intercept.
        """
        scorer = cls()
        scorer.raw_weights = np.array(raw_weights, dtype=np.float32)
        scorer.raw_intercept = float(raw_intercept)
        return scorer

    def get_learned_weights_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the learned standardized and raw weights for all 22 features.
        """
        records = []
        for name, w_std, w_raw in zip(FEATURE_NAMES, self.model.coef_[0], self.raw_weights):
            records.append({
                'feature': name,
                'standardized_weight': float(w_std),
                'raw_weight': float(w_raw)
            })
        records.append({
            'feature': 'Intercept',
            'standardized_weight': float(self.model.intercept_[0]),
            'raw_weight': float(self.raw_intercept)
        })
        return pd.DataFrame(records)


class SVMLearningToRank:
    """
    Orchestrates candidate generation, feature extraction, SVM training,
    and PyTerrier LTR test query re-ranking.
    """

    def __init__(self, stats: CranfieldCorpusStats, candidate_retriever: any, C: float = 0.01):
        self.stats = stats
        self.candidate_retriever = candidate_retriever
        self.feature_extractor = FeatureExtractor(stats)
        self.scorer = LinearSVMScorer(C=C)
        self.is_fitted = False

    def build_training_dataset(
        self,
        train_queries: List[Query],
        train_qrels: List[Qrel]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates candidate documents using the first-stage retriever,
        augments with ground-truth positives, and computes the 22 features.
        """
        qrels_dict: Dict[str, Set[str]] = {}
        for qr in train_qrels:
            if qr.label == 1:
                if qr.qid not in qrels_dict:
                    qrels_dict[qr.qid] = set()
                qrels_dict[qr.qid].add(qr.docno)

        queries_df = pd.DataFrame([{'qid': q.qid, 'query': q.query} for q in train_queries])
        cand_df = self.candidate_retriever.transform(queries_df)

        cand_grouped = cand_df.groupby('qid')
        X_list = []
        y_list = []

        q_map = {q.qid: q for q in train_queries}

        for q in train_queries:
            qid = q.qid
            q_toks = q.tokens if q.tokens else tokenize(q.query, remove_stopwords=False)
            seen_docs = set()

            if qid in cand_grouped.groups:
                group = cand_grouped.get_group(qid)
                for doc_id in group['docno']:
                    doc_id_str = str(doc_id)
                    seen_docs.add(doc_id_str)
                    feat = self.feature_extractor.extract_features(q_toks, doc_id_str)
                    label = 1 if doc_id_str in qrels_dict.get(qid, set()) else 0
                    X_list.append(feat)
                    y_list.append(label)

            # Ensure all ground-truth relevant documents are included in the training set
            for rel_doc in qrels_dict.get(qid, set()):
                if rel_doc not in seen_docs:
                    feat = self.feature_extractor.extract_features(q_toks, rel_doc)
                    X_list.append(feat)
                    y_list.append(1)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int32)
        return X, y

    def fit(self, train_queries: List[Query], train_qrels: List[Qrel]) -> LinearSVMScorer:
        """
        Trains the Linear SVM model on extracted query-document features.
        """
        X_train, y_train = self.build_training_dataset(train_queries, train_qrels)
        self.scorer.fit(X_train, y_train)
        self.is_fitted = True
        return self.scorer

    def get_pyterrier_pipeline(self) -> pt.Transformer:
        """
        Constructs the complete PyTerrier LTR pipeline:
          cand_retriever >> doc_features >> pt.ltr.apply_learned_model(scorer)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before constructing PyTerrier pipeline.")

        feat_trans = self.feature_extractor.get_pyterrier_transformer()
        ltr_trans = pt.ltr.apply_learned_model(self.scorer)
        return self.candidate_retriever >> feat_trans >> ltr_trans

    def rank_test_queries(self, test_queries: List[Query]) -> pd.DataFrame:
        """
        Applies candidate retrieval and SVM re-ranking to test queries.
        """
        queries_df = pd.DataFrame([{'qid': q.qid, 'query': q.query} for q in test_queries])
        pipeline = self.get_pyterrier_pipeline()
        ranked_res = pipeline.transform(queries_df)
        # Ensure clean rank indices
        ranked_res = ranked_res.sort_values(['qid', 'score'], ascending=[True, False]).reset_index(drop=True)
        ranked_res['rank'] = ranked_res.groupby('qid').cumcount()
        return ranked_res

    @classmethod
    def from_pretrained(
        cls,
        stats: CranfieldCorpusStats,
        candidate_retriever: pt.Transformer,
        raw_weights: np.ndarray,
        raw_intercept: float
    ):
        """
        Instantiates SVMLearningToRank with pre-learned weights for zero-leakage inference.
        """
        ltr = cls(stats=stats, candidate_retriever=candidate_retriever)
        ltr.scorer = LinearSVMScorer.from_pretrained_weights(raw_weights, raw_intercept)
        ltr.is_fitted = True
        return ltr

    def save(self, filepath: str):
        """Saves the trained LinearSVMScorer to a joblib file."""
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.scorer, filepath)

    def load(self, filepath: str):
        """Loads a pre-trained LinearSVMScorer from a joblib file."""
        import joblib
        self.scorer = joblib.load(filepath)
        self.is_fitted = True

