"""
main.py
=======
Master execution script for Learning to Rank on the Cranfield Text Collection.
Implements the complete end-to-end pipeline:
  1. Data Parsing & Train/Test Split (80% Train, 20% Test)
  2. PyTerrier Indexing & Corpus Statistics
  3. Benchmark Baseline Model (BM25)
  4. Method A: Boolean Weighted Zone Scoring (Grid Search Base Weight Learning)
  5. Method B: Linear SVM Learning to Rank (22 Feature Extraction & Weight Learning)
  6. Comprehensive Metric Evaluation (Precision@k, Recall@k, F1@k, MAP, MRR)
  7. Exporting TREC-Formatted Test Query Results to Text Files
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from data_parser import CranfieldParser
from indexer import CranfieldIndexer, init_pyterrier
from baseline import BM25Baseline
from method_a_zone_scoring import BooleanWeightedZoneScorer, GridSearchZoneOptimizer
from method_b_svm_ltr import SVMLearningToRank, FEATURE_NAMES
from evaluation import RetrievalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Learning to Rank on Cranfield Corpus")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing raw Cranfield files")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="Train/test split ratio for queries (default: 0.20)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for train/test splitting (default: 42)")
    parser.add_argument("--index_dir", type=str, default="./terrier_cran_index", help="PyTerrier index directory")
    parser.add_argument("--grid_step", type=float, default=0.05, help="Grid search step size for Method A (default: 0.05)")
    parser.add_argument("--svm_c", type=float, default=0.01, help="Regularization parameter C for Linear SVM (default: 0.01)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save test query result files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("INFORMATION RETRIEVAL PROGRAMMING ASSIGNMENT III: LEARNING TO RANK")
    print("=" * 80)

    # ---------------------------------------------------------
    # Step 1: Data Parsing & Train/Test Splitting
    # ---------------------------------------------------------
    print("\n[Step 1] Parsing raw Cranfield files...")
    docs_file = os.path.join(args.data_dir, "cran.all.1400")
    queries_file = os.path.join(args.data_dir, "cran.qry")
    qrels_file = os.path.join(args.data_dir, "cranqrel")

    # Auto-extract cran.tar.gz if data directory is missing parsed files
    if not os.path.exists(docs_file):
        import tarfile
        tar_path = "cran.tar.gz" if os.path.exists("cran.tar.gz") else os.path.join(args.data_dir, "cran.tar.gz")
        if os.path.exists(tar_path):
            print(f"  * Extracting {tar_path} into {args.data_dir}...")
            os.makedirs(args.data_dir, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(args.data_dir)

    documents = CranfieldParser.parse_documents(docs_file)
    queries = CranfieldParser.parse_queries(queries_file)
    qrels = CranfieldParser.parse_qrels(qrels_file)

    print(f"  * Total documents extracted: {len(documents)}")
    print(f"  * Total queries extracted:   {len(queries)}")
    print(f"  * Total qrel judgments:     {len(qrels)} (Relevant judgments: {sum(qr.label for qr in qrels)})")

    print("\n[Step 2] Executing Train/Test Split...")
    train_queries, test_queries, train_qrels, test_qrels = CranfieldParser.train_test_split(
        queries=queries,
        qrels=qrels,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )

    train_pct = int(round((1.0 - args.test_ratio) * 100))
    test_pct = int(round(args.test_ratio * 100))
    print(f"  * Chosen Train/Test Split Ratio: {train_pct}% Train / {test_pct}% Test (random_state={args.random_state})")
    print(f"  * Training Queries: {len(train_queries)} | Training Qrels: {len(train_qrels)}")
    print(f"  * Testing Queries:  {len(test_queries)}  | Testing Qrels:  {len(test_qrels)}")

    train_queries_df, train_qrels_df = CranfieldParser.to_pandas(train_queries, train_qrels)
    test_queries_df, test_qrels_df = CranfieldParser.to_pandas(test_queries, test_qrels)

    # ---------------------------------------------------------
    # Step 2: Indexing & Collection Statistics
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing PyTerrier & Inverted Index...")
    index_ref, stats = CranfieldIndexer.build_or_load_index(
        documents=documents,
        index_dir=args.index_dir,
        overwrite=False
    )
    print(f"  * Inverted index loaded successfully at: {args.index_dir}")
    print(f"  * Corpus vocabulary size: {len(stats.df_counts)} unique terms")
    print(f"  * Corpus average document length (avgdl): {stats.avgdl:.2f} tokens")

    # ---------------------------------------------------------
    # Step 3: Baseline Model Evaluation (BM25)
    # ---------------------------------------------------------
    print("\n[Step 4] Running Baseline Model (Pure BM25)...")
    baseline_model = BM25Baseline(index_ref=index_ref, num_results=100)
    baseline_test_results = baseline_model.retrieve(test_queries_df)
    print(f"  * Baseline retrieved {len(baseline_test_results)} candidate pairs for {len(test_queries)} test queries.")

    # ---------------------------------------------------------
    # Step 4: Method A - Boolean Weighted Zone Scoring
    # ---------------------------------------------------------
    print("\n[Step 5] Running Method A: Boolean Weighted Zone Scoring...")
    print(f"  * Running grid search over Title, Author, Bibliography, Body zones (step={args.grid_step})...")
    optimizer = GridSearchZoneOptimizer(
        stats=stats,
        step=args.grid_step,
        target_metric="f1_10",
        scoring_mode="fraction"
    )

    best_weights, best_train_f1, history = optimizer.fit(
        train_queries=train_queries,
        train_qrels=train_qrels
    )

    print(f"  * Grid search explored {history['total_candidates']} candidate weight combinations.")
    print(f"  * Optimal learned base weights:")
    print(f"      - Title weight:        {best_weights[0]:.4f}")
    print(f"      - Author weight:       {best_weights[1]:.4f}")
    print(f"      - Bibliography weight: {best_weights[2]:.4f}")
    print(f"      - Body text weight:    {best_weights[3]:.4f}")
    print(f"  * Best Training Mean F1@10: {best_train_f1:.4f}")

    zone_scorer = BooleanWeightedZoneScorer(
        stats=stats,
        weights=best_weights,
        scoring_mode="fraction"
    )
    method_a_test_results = zone_scorer.transform(test_queries_df)
    print(f"  * Method A scored and ranked {len(method_a_test_results)} document pairs.")

    # ---------------------------------------------------------
    # Step 5: Method B - Linear SVM Learning to Rank (22 Features)
    # ---------------------------------------------------------
    print("\n[Step 6] Running Method B: Linear SVM Learning to Rank...")
    print("  * Extracting 22 features and training Linear SVM on training query-document pairs...")
    candidate_retriever = baseline_model.get_pyterrier_transformer()
    svm_ltr = SVMLearningToRank(
        stats=stats,
        candidate_retriever=candidate_retriever,
        C=args.svm_c
    )

    svm_scorer = svm_ltr.fit(train_queries=train_queries, train_qrels=train_qrels)
    weights_summary = svm_scorer.get_learned_weights_summary()

    print("\n  === Learned Real-Valued Weights for 22 Features (Linear SVM) ===")
    print(f"  {'#':<3} | {'Feature Name':<46} | {'Standardized Weight':<20} | {'Raw Weight':<15}")
    print("  " + "-" * 90)
    for idx, row in weights_summary.iterrows():
        f_name = row['feature']
        w_std = row['standardized_weight']
        w_raw = row['raw_weight']
        print(f"  {idx+1:<3} | {f_name:<46} | {w_std:<20.4f} | {w_raw:<15.6f}")

    print("\n  * Generating re-ranked results for test queries using PyTerrier LTR pipeline...")
    method_b_test_results = svm_ltr.rank_test_queries(test_queries=test_queries)
    print(f"  * Method B re-ranked {len(method_b_test_results)} candidate pairs.")

    # Save trained SVM model for fast unknown query evaluation
    svm_model_path = os.path.join(args.output_dir, "svm_model.joblib")
    svm_ltr.save(svm_model_path)
    print(f"  * Saved trained SVM LTR model to: {svm_model_path}")

    # ---------------------------------------------------------
    # Step 6: Comprehensive Metric Evaluation
    # ---------------------------------------------------------
    print("\n[Step 7] Evaluating Retrieval Performance on Test Queries (Consistent Depth: Top-100)...")
    evaluator = RetrievalEvaluator(qrels=test_qrels, cutoffs=[5, 10, 20])

    # Ensure consistent evaluation depth (top 100) across all compared models
    method_a_top100 = method_a_test_results.sort_values(['qid', 'rank']).groupby('qid').head(100).reset_index(drop=True)

    model_dict = {
        'BM25 Baseline': baseline_test_results,
        'Method A (Weighted Zone)': method_a_top100,
        'Method B (Linear SVM LTR)': method_b_test_results
    }

    comparison_table = evaluator.compare_models(model_dict)

    print("\n" + "=" * 95)
    print(f"SUMMARY EVALUATION RESULTS ON TEST SUBSET ({len(test_queries)} QUERIES, SPLIT RATIO: {test_pct}%)")
    print("=" * 95)
    metric_cols = ['P@5', 'P@10', 'P@20', 'R@5', 'R@10', 'R@20', 'F1@5', 'F1@10', 'F1@20', 'MAP', 'MRR']
    display_df = comparison_table[metric_cols]
    print(display_df.round(4).to_string())
    print("=" * 95)

    # ---------------------------------------------------------
    # Step 7: Export Test Query Results to Text Files
    # ---------------------------------------------------------
    print("\n[Step 8] Exporting Test Query Results to Text Files...")
    baseline_out = os.path.join(args.output_dir, "test_results_baseline.txt")
    method_a_out = os.path.join(args.output_dir, "test_results_method_a.txt")
    method_b_out = os.path.join(args.output_dir, "test_results_method_b.txt")
    all_out = os.path.join(args.output_dir, "test_results_all.txt")

    evaluator.save_results_trec(baseline_test_results, baseline_out, system_tag="BM25_Baseline", max_docs_per_query=100)
    evaluator.save_results_trec(method_a_test_results, method_a_out, system_tag="MethodA_ZoneScoring", max_docs_per_query=100)
    evaluator.save_results_trec(method_b_test_results, method_b_out, system_tag="MethodB_LinearSVM_LTR", max_docs_per_query=100)

    # Combined summary file
    with open(all_out, 'w', encoding='utf-8') as f:
        f.write("# CRANFIELD LEARNING TO RANK: TEST QUERY EVALUATION RESULTS\n")
        f.write(f"# Split Ratio: {train_pct}% Train ({len(train_queries)} queries) / {test_pct}% Test ({len(test_queries)} queries)\n\n")
        f.write("=== COMPARATIVE EVALUATION TABLE ===\n")
        f.write(display_df.round(4).to_string())
        f.write("\n\n=== METHOD A OPTIMAL ZONE WEIGHTS ===\n")
        f.write(f"Title: {best_weights[0]:.4f}, Author: {best_weights[1]:.4f}, Bibliography: {best_weights[2]:.4f}, Body: {best_weights[3]:.4f}\n")
        f.write(f"Best Training F1@10: {best_train_f1:.4f}\n\n")
        f.write("=== METHOD B LEARNED SVM FEATURE WEIGHTS ===\n")
        f.write(weights_summary.to_string())
        f.write("\n")

    print(f"  * Baseline results saved to: {baseline_out}")
    print(f"  * Method A results saved to: {method_a_out}")
    print(f"  * Method B results saved to: {method_b_out}")
    print(f"  * Complete summary saved to:  {all_out}")

    print("\nPipeline execution complete successfully!")


if __name__ == "__main__":
    main()

