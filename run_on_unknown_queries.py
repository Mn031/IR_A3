"""
run_on_unknown_queries.py
=========================
Evaluation runner for unseen/unknown test queries.
Loads the trained BM25, Method A (learned zone weights), and Method B (trained Linear SVM LTR)
models, processes any given query file, and outputs ranked document lists in TREC format.

Usage:
    python run_on_unknown_queries.py --queries_file ./data/cran.qry --model all --output_file ./results/unseen_results.txt
"""

import os
import argparse
import numpy as np
import pandas as pd

from data_parser import CranfieldParser, tokenize
from indexer import CranfieldIndexer
from baseline import BM25Baseline
from method_a_zone_scoring import BooleanWeightedZoneScorer
from method_b_svm_ltr import SVMLearningToRank
from evaluation import RetrievalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run trained LTR models on unknown/unseen queries.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing Cranfield corpus files")
    parser.add_argument("--queries_file", type=str, default="./data/cran.qry", help="Path to input query file (.qry format)")
    parser.add_argument("--index_dir", type=str, default="./terrier_cran_index", help="PyTerrier index directory")
    parser.add_argument("--model", type=str, choices=['baseline', 'method_a', 'method_b', 'all'], default='all',
                        help="Which model to run ('baseline', 'method_a', 'method_b', or 'all')")
    parser.add_argument("--output_file", type=str, default="./results/evaluation_output.txt", help="Path to output ranked results file")
    parser.add_argument("--max_docs", type=int, default=100, help="Maximum documents to return per query")
    args = parser.parse_args()

    print(f"Loading documents from {args.data_dir}...")
    docs_file = os.path.join(args.data_dir, "cran.all.1400")
    if not os.path.exists(docs_file):
        import tarfile
        tar_path = "cran.tar.gz" if os.path.exists("cran.tar.gz") else os.path.join(args.data_dir, "cran.tar.gz")
        if os.path.exists(tar_path):
            print(f"  * Extracting {tar_path} into {args.data_dir}...")
            os.makedirs(args.data_dir, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(args.data_dir)

    documents = CranfieldParser.parse_documents(docs_file)

    print(f"Loading PyTerrier index and corpus statistics...")
    index_ref, stats = CranfieldIndexer.build_or_load_index(documents, index_dir=args.index_dir, overwrite=False)

    print(f"Parsing queries from {args.queries_file}...")
    queries = CranfieldParser.parse_queries(args.queries_file)
    queries_df = pd.DataFrame([{'qid': q.qid, 'query': q.query} for q in queries])
    print(f"Loaded {len(queries_df)} queries.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # 1. Baseline BM25
    baseline_model = BM25Baseline(index_ref=index_ref, num_results=args.max_docs)

    # 2. Method A with optimal learned weights
    # Optimal weights from training: Title=0.55, Author=0.00, Bib=0.05, Body=0.40
    best_weights = np.array([0.55, 0.00, 0.05, 0.40], dtype=np.float32)
    method_a_scorer = BooleanWeightedZoneScorer(stats=stats, weights=best_weights, scoring_mode="fraction")

    # 3. Method B (Linear SVM LTR)
    qrels_file = os.path.join(args.data_dir, "cranqrel")
    qrels = CranfieldParser.parse_qrels(qrels_file)
    train_queries, _, train_qrels, _ = CranfieldParser.train_test_split(queries, qrels, test_ratio=0.20, random_state=42)

    svm_ltr = SVMLearningToRank(stats=stats, candidate_retriever=baseline_model.get_pyterrier_transformer(), C=0.01)
    print("Training SVM LTR re-ranker on base training set...")
    svm_ltr.fit(train_queries, train_qrels)

    results_to_save = {}

    if args.model in ['baseline', 'all']:
        print("Executing BM25 Baseline retrieval...")
        res_base = baseline_model.retrieve(queries_df)
        results_to_save['BM25_Baseline'] = res_base

    if args.model in ['method_a', 'all']:
        print("Executing Method A (Boolean Weighted Zone Scoring)...")
        res_a = method_a_scorer.transform(queries_df)
        results_to_save['MethodA_ZoneScoring'] = res_a

    if args.model in ['method_b', 'all']:
        print("Executing Method B (Linear SVM LTR)...")
        res_b = svm_ltr.rank_test_queries(queries)
        results_to_save['MethodB_LinearSVM_LTR'] = res_b

    # Export results
    lines = []
    for tag, df in results_to_save.items():
        for qid, qdf in df.groupby('qid'):
            qdf_sorted = qdf.sort_values('rank').head(args.max_docs)
            for rank, (_, row) in enumerate(qdf_sorted.iterrows()):
                line = f"{row['qid']} Q0 {row['docno']} {rank} {row['score']:.6f} {tag}\n"
                lines.append(line)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Successfully saved {len(lines)} ranking lines to: {args.output_file}")


if __name__ == "__main__":
    main()

