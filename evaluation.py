"""
evaluation.py
=============
Evaluation module for measuring ranked retrieval performance on test queries
and ground truth relevance judgments.
Calculates Precision, Recall, and F1 score at multiple cutoffs (k=5, 10, 20),
along with MAP and MRR. Exports test query rankings in standard TREC format.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
from data_parser import Qrel


class RetrievalEvaluator:
    """
    Computes Information Retrieval metrics for ranked query results against qrels.
    """

    def __init__(self, qrels: List[Qrel], cutoffs: Optional[List[int]] = None):
        """
        :param qrels: Ground truth relevance judgments.
        :param cutoffs: Rank cutoffs for P@k, R@k, F1@k (default: [5, 10, 20]).
        """
        self.cutoffs = cutoffs or [5, 10, 20]
        self.qrels_map: Dict[str, Set[str]] = {}
        for qr in qrels:
            if qr.label == 1:
                if qr.qid not in self.qrels_map:
                    self.qrels_map[qr.qid] = set()
                self.qrels_map[qr.qid].add(qr.docno)

    def evaluate_dataframe(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates a DataFrame with columns ['qid', 'docno', 'score', 'rank'].
        Returns macro-averaged metrics across all evaluated queries.
        """
        metrics = {f'P@{k}': [] for k in self.cutoffs}
        metrics.update({f'R@{k}': [] for k in self.cutoffs})
        metrics.update({f'F1@{k}': [] for k in self.cutoffs})
        metrics['MAP'] = []
        metrics['MRR'] = []

        grouped = results_df.groupby('qid')

        for qid, qdf in grouped:
            qid_str = str(qid)
            rel_docs = self.qrels_map.get(qid_str, set())
            n_rel = len(rel_docs)
            if n_rel == 0:
                continue

            qdf_sorted = qdf.sort_values('rank')
            doc_list = [str(d) for d in qdf_sorted['docno'].tolist()]

            # Precision@k, Recall@k, F1@k
            for k in self.cutoffs:
                top_k = set(doc_list[:k])
                hits = len(top_k.intersection(rel_docs))
                p_k = hits / k
                r_k = hits / n_rel
                f1_k = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0

                metrics[f'P@{k}'].append(p_k)
                metrics[f'R@{k}'].append(r_k)
                metrics[f'F1@{k}'].append(f1_k)

            # Average Precision (AP)
            hits = 0
            ap = 0.0
            rr = 0.0
            for rank_idx, doc_id in enumerate(doc_list):
                if doc_id in rel_docs:
                    hits += 1
                    ap += hits / (rank_idx + 1)
                    if rr == 0.0:
                        rr = 1.0 / (rank_idx + 1)

            ap /= n_rel
            metrics['MAP'].append(ap)
            metrics['MRR'].append(rr)

        return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in metrics.items()}

    def compare_models(
        self,
        model_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Evaluates multiple models and builds a comparative DataFrame.
        """
        all_metrics = {}
        for model_name, res_df in model_results.items():
            all_metrics[model_name] = self.evaluate_dataframe(res_df)

        comparison_df = pd.DataFrame(all_metrics).T
        return comparison_df

    @staticmethod
    def save_results_trec(
        results_df: pd.DataFrame,
        output_filepath: str,
        system_tag: str = "LTR_Model",
        max_docs_per_query: int = 100
    ) -> None:
        """
        Saves query ranking results in standard TREC format:
          <query_id> Q0 <doc_id> <rank> <score> <system_tag>
        """
        lines = []
        for qid, qdf in results_df.groupby('qid'):
            qdf_sorted = qdf.sort_values('rank').head(max_docs_per_query)
            for rank, (_, row) in enumerate(qdf_sorted.iterrows()):
                line = f"{row['qid']} Q0 {row['docno']} {rank} {row['score']:.6f} {system_tag}\n"
                lines.append(line)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Saved {len(lines)} ranking lines to: {output_filepath}")

