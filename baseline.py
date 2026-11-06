"""
baseline.py
===========
Implements the standard BM25 retrieval baseline using the PyTerrier framework.
Serves as the rigorous benchmark for evaluating Learning-to-Rank methods.
"""

from typing import Optional
import pandas as pd
import pyterrier as pt


class BM25Baseline:
    """
    Standard Robertson-Spärck Jones BM25 retrieval model wrapped in PyTerrier.
    """

    def __init__(self, index_ref: any, num_results: int = 100):
        """
        Initializes PyTerrier BM25 retriever.
        :param index_ref: PyTerrier IndexRef object.
        :param num_results: Number of documents to retrieve per query (default: 100).
        """
        self.index_ref = index_ref
        self.num_results = num_results
        self.retriever = pt.terrier.Retriever(
            self.index_ref,
            wmodel='BM25',
            num_results=self.num_results
        )

    def retrieve(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes BM25 retrieval for a DataFrame of queries.
        :param queries_df: DataFrame with columns ['qid', 'query'].
        :return: DataFrame with columns ['qid', 'docno', 'score', 'rank'].
        """
        req_cols = ['qid', 'query']
        if not all(col in queries_df.columns for col in req_cols):
            raise ValueError(f"Input DataFrame must contain columns: {req_cols}")

        res = self.retriever.transform(queries_df[req_cols])
        # Ensure clean rank and score sorting per query
        res = res.sort_values(['qid', 'score'], ascending=[True, False]).reset_index(drop=True)
        res['rank'] = res.groupby('qid').cumcount()
        return res

    def get_pyterrier_transformer(self) -> pt.Transformer:
        """
        Returns the underlying PyTerrier transformer pipeline object.
        """
        return self.retriever

