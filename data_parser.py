"""
data_parser.py
==============
Modular custom parser for the Cranfield text collection and relevance judgments.
Properly extracts distinct document fields (.T, .A, .B, .W), queries (.I, .W),
and relevance judgments (cranqrel) with a reproducible train/test split.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


# Standard IR English Stopwords
STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such',
    't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your'
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize input text into lowercase alphanumeric terms, optionally removing stopwords.
    """
    if not text:
        return []
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return words


@dataclass
class Document:
    doc_id: str
    title: str
    author: str
    bibliography: str
    body: str
    full_text: str = ""

    def __post_init__(self):
        if not self.full_text:
            parts = [p for p in [self.title, self.author, self.bibliography, self.body] if p]
            self.full_text = " ".join(parts).strip()


@dataclass
class Query:
    qid: str          # Normalized sequential ID (1..225) aligned with cranqrel
    orig_id: str      # Historical ID in cran.qry (.I 001, .I 002, .I 004, ...)
    query: str        # Query text from .W
    tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = tokenize(self.query, remove_stopwords=True)


@dataclass
class Qrel:
    qid: str
    docno: str
    grade: int
    label: int        # Binary relevance: 1 if relevant, 0 if non-relevant


class CranfieldParser:
    """
    Custom parser for the Cranfield text collection.
    Processes:
      - cran.all.1400: documents marked by .I, .T, .A, .B, .W
      - cran.qry: queries marked by .I and .W
      - cranqrel: relevance judgments (query_id, doc_id, grade)
    """

    @staticmethod
    def parse_documents(filepath: str) -> List[Document]:
        """
        Parses cran.all.1400 and extracts fields for each document:
          - .I : Document ID
          - .T : Title
          - .A : Author
          - .B : Bibliography
          - .W : Body text / abstract
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document file not found at: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into individual document chunks by .I marker
        doc_chunks = re.split(r'\n(?=\.I\s+\d+)', content)
        documents = []

        for chunk in doc_chunks:
            lines = chunk.strip().split('\n')
            if not lines or not lines[0].startswith('.I'):
                continue

            doc_id = lines[0].split()[1]
            fields = {'title': '', 'author': '', 'bibliography': '', 'body': ''}
            curr_tag = None
            curr_content: List[str] = []

            tag_map = {'.T': 'title', '.A': 'author', '.B': 'bibliography', '.W': 'body'}

            for line in lines[1:]:
                prefix = line[:2]
                if prefix in tag_map:
                    if curr_tag:
                        fields[curr_tag] = ' '.join(curr_content).strip()
                    curr_tag = tag_map[prefix]
                    curr_content = []
                else:
                    if curr_tag:
                        curr_content.append(line.strip())

            if curr_tag:
                fields[curr_tag] = ' '.join(curr_content).strip()

            doc = Document(
                doc_id=doc_id,
                title=fields['title'],
                author=fields['author'],
                bibliography=fields['bibliography'],
                body=fields['body']
            )
            documents.append(doc)

        return documents

    @staticmethod
    def parse_queries(filepath: str) -> List[Query]:
        """
        Parses cran.qry and extracts queries marked by .I and .W.
        Maps the queries sequentially (1 to 225) to align directly with cranqrel,
        while maintaining the original .I identifier.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Query file not found at: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        query_chunks = re.split(r'\n(?=\.I\s+\d+)', content)
        queries = []
        seq_idx = 1

        for chunk in query_chunks:
            lines = chunk.strip().split('\n')
            if not lines or not lines[0].startswith('.I'):
                continue

            orig_id = lines[0].split()[1]
            curr_content: List[str] = []
            in_w = False

            for line in lines[1:]:
                if line.startswith('.W'):
                    in_w = True
                    continue
                if in_w:
                    curr_content.append(line.strip())

            qtext = ' '.join(curr_content).strip()
            queries.append(Query(
                qid=str(seq_idx),
                orig_id=orig_id,
                query=qtext
            ))
            seq_idx += 1

        return queries

    @staticmethod
    def parse_qrels(filepath: str, relevant_grades: Optional[List[int]] = None) -> List[Qrel]:
        """
        Parses cranqrel relevance judgments.
        Format: <query_id> <doc_id> <relevancy_grade>
        Relevance scale:
          -1 : Source document from which query was formulated (highly relevant)
           1 : Complete answer to question
           2 : High degree of relevance
           3 : Useful background
           4 : Minimum interest
           5 : No interest (not included in file)
        By default, grades in [-1, 1, 2, 3, 4] are treated as binary relevant (1).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Qrel file not found at: {filepath}")

        if relevant_grades is None:
            relevant_grades = [-1, 1, 2, 3, 4]

        qrels = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    qid = parts[0]
                    docno = parts[1]
                    grade = int(parts[2])
                    label = 1 if grade in relevant_grades else 0
                    qrels.append(Qrel(
                        qid=qid,
                        docno=docno,
                        grade=grade,
                        label=label
                    ))

        return qrels

    @classmethod
    def train_test_split(
        cls,
        queries: List[Query],
        qrels: List[Qrel],
        test_ratio: float = 0.20,
        random_state: int = 42
    ) -> Tuple[List[Query], List[Query], List[Qrel], List[Qrel]]:
        """
        Splits queries and their corresponding relevance judgments into train and test sets.
        Default ratio is 80% Train, 20% Test with a fixed random_state for exact reproducibility.
        """
        train_queries, test_queries = train_test_split(
            queries,
            test_size=test_ratio,
            random_state=random_state
        )

        train_qids = {q.qid for q in train_queries}
        test_qids = {q.qid for q in test_queries}

        train_qrels = [qr for qr in qrels if qr.qid in train_qids]
        test_qrels = [qr for qr in qrels if qr.qid in test_qids]

        return train_queries, test_queries, train_qrels, test_qrels

    @staticmethod
    def to_pandas(
        queries: List[Query],
        qrels: List[Qrel]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts queries and qrels lists to PyTerrier-compatible pandas DataFrames.
        """
        queries_df = pd.DataFrame([
            {'qid': q.qid, 'query': q.query, 'orig_id': q.orig_id}
            for q in queries
        ])

        qrels_df = pd.DataFrame([
            {'qid': qr.qid, 'docno': qr.docno, 'label': qr.label}
            for qr in qrels
        ])

        return queries_df, qrels_df

