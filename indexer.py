"""
indexer.py
==========
Manages PyTerrier inverted index creation, disk persistence, and collection statistics
for the Cranfield corpus. Precomputes token frequencies, document lengths, and IDF
dictionaries needed for high-speed feature extraction.
"""

import os
import shutil
import math
from collections import Counter
from typing import List, Dict, Set, Tuple
import numpy as np
import pyterrier as pt

from data_parser import Document, tokenize


def init_pyterrier() -> None:
    """
    Safely initialize PyTerrier JVM.
    """
    if not pt.java.started():
        pt.java.init()


class CranfieldCorpusStats:
    """
    High-performance corpus statistics and token-level index cache.
    Stores pre-tokenized fields, term frequencies, document lengths, and IDFs
    for fast feature extraction in Methods A and B.
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_map: Dict[str, Document] = {d.doc_id: d for d in documents}
        self.doc_ids: List[str] = [d.doc_id for d in documents]
        self.N: int = len(documents)

        # Pre-tokenize all documents and individual zones
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_tfs: Dict[str, Counter] = {}
        self.doc_lens: Dict[str, int] = {}

        self.zone_tokens: Dict[str, Dict[str, Set[str]]] = {}

        self.df_counts: Counter = Counter()

        for doc in documents:
            doc_id = doc.doc_id
            tokens = tokenize(doc.full_text, remove_stopwords=True)
            self.doc_tokens[doc_id] = tokens
            self.doc_tfs[doc_id] = Counter(tokens)
            self.doc_lens[doc_id] = len(tokens) if len(tokens) > 0 else 1
            self.df_counts.update(set(tokens))

            self.zone_tokens[doc_id] = {
                'title': set(tokenize(doc.title, remove_stopwords=True)),
                'author': set(tokenize(doc.author, remove_stopwords=True)),
                'bibliography': set(tokenize(doc.bibliography, remove_stopwords=True)),
                'body': set(tokenize(doc.body, remove_stopwords=True))
            }

        self.avgdl: float = float(np.mean(list(self.doc_lens.values())))

        # Precompute RSJ IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf_cache: Dict[str, float] = {}
        for term, df in self.df_counts.items():
            self.idf_cache[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        # Precompute VSM document vector Euclidean norms
        self.doc_vsm_norms: Dict[str, float] = {}
        for doc_id, tf_counter in self.doc_tfs.items():
            sq_sum = sum((tf * self.get_idf(t)) ** 2 for t, tf in tf_counter.items())
            self.doc_vsm_norms[doc_id] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0

    def get_idf(self, term: str) -> float:
        """
        Returns the Robertson-Spärck Jones IDF for a given term.
        Falls back to log(1 + (N - 0 + 0.5)/(0.5)) for unseen terms.
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = self.df_counts.get(term, 0)
        idf_val = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
        self.idf_cache[term] = idf_val
        return idf_val


class CranfieldIndexer:
    """
    Constructs and loads PyTerrier inverted index for the Cranfield documents.
    """

    @staticmethod
    def build_or_load_index(
        documents: List[Document],
        index_dir: str = "./terrier_cran_index",
        overwrite: bool = False
    ) -> Tuple[any, CranfieldCorpusStats]:
        """
        Builds a new PyTerrier IterDictIndexer or loads the existing index reference.
        Also returns the CranfieldCorpusStats instance for feature computation.
        """
        init_pyterrier()
        abs_index_dir = os.path.abspath(index_dir)
        properties_file = os.path.join(abs_index_dir, "data.properties")

        stats = CranfieldCorpusStats(documents)

        if not overwrite and os.path.exists(properties_file):
            index_ref = pt.IndexRef.of(properties_file)
            return index_ref, stats

        if os.path.exists(abs_index_dir):
            shutil.rmtree(abs_index_dir)
        os.makedirs(abs_index_dir, exist_ok=True)

        # Prepare records for PyTerrier IterDictIndexer
        doc_records = [
            {
                'docno': d.doc_id,
                'text': d.full_text,
                'title': d.title,
                'author': d.author,
                'bibliography': d.bibliography,
                'body': d.body
            }
            for d in documents
        ]

        indexer = pt.IterDictIndexer(
            abs_index_dir,
            meta={'docno': 20},
            overwrite=True
        )
        index_ref = indexer.index(doc_records)

        return index_ref, stats

