"""
generate_plots.py
=================
Generates publication-quality charts (300 DPI) for the IR PA3 report:
  1. Precision@k Comparison (k=5, 10, 20)
  2. Recall@k Comparison (k=5, 10, 20)
  3. F1@k Comparison (k=5, 10, 20)
  4. MAP & MRR Benchmark Comparison
  5. Method A Zone Weight Allocation (Donut Chart)
  6. Method B 22-Feature Standardized Weights (Horizontal Bar Chart)
  7. SVM Regularization Sensitivity (C vs. MAP & F1@10)
  8. Precision-Recall Trade-off Curves
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('./.mpl_cache')
os.makedirs('./.mpl_cache', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set cohesive academic visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13

# Empirical Data Points from Experimental Runs
cutoffs = [5, 10, 20]
bm25_p = [0.4578, 0.3089, 0.1956]
method_a_p = [0.4000, 0.2644, 0.1678]
method_b_p = [0.3867, 0.2644, 0.1667]

bm25_r = [0.3698, 0.4812, 0.5864]
method_a_r = [0.3207, 0.4141, 0.5265]
method_b_r = [0.3259, 0.4272, 0.5295]

bm25_f1 = [0.3795, 0.3500, 0.2759]
method_a_f1 = [0.3338, 0.3033, 0.2416]
method_b_f1 = [0.3282, 0.3066, 0.2406]

c_bm25 = '#1E3A8A'   # Navy
c_meth_a = '#0D9488' # Teal
c_meth_b = '#EA580C' # Orange/Rust

# -------------------------------------------------------------
# FIGURE 1: Precision@k Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
x = np.arange(len(cutoffs))
w = 0.25
rects1 = ax.bar(x - w, bm25_p, w, label='BM25 Baseline', color=c_bm25, edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, method_a_p, w, label='Method A: Weighted Zone', color=c_meth_a, edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + w, method_b_p, w, label='Method B: Linear SVM LTR', color=c_meth_b, edgecolor='black', alpha=0.9)

ax.set_ylabel('Precision@k')
ax.set_xlabel('Rank Cutoff (k)')
ax.set_title('Precision Comparison across Retrieval Models (Test Queries)')
ax.set_xticks(x)
ax.set_xticklabels([f'k = {k}' for k in cutoffs])
ax.set_ylim(0, 0.55)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

plt.tight_layout()
plt.savefig('./figures/precision_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 2: Recall@k Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
rects1 = ax.bar(x - w, bm25_r, w, label='BM25 Baseline', color=c_bm25, edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, method_a_r, w, label='Method A: Weighted Zone', color=c_meth_a, edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + w, method_b_r, w, label='Method B: Linear SVM LTR', color=c_meth_b, edgecolor='black', alpha=0.9)

ax.set_ylabel('Recall@k')
ax.set_xlabel('Rank Cutoff (k)')
ax.set_title('Recall Comparison across Retrieval Models (Test Queries)')
ax.set_xticks(x)
ax.set_xticklabels([f'k = {k}' for k in cutoffs])
ax.set_ylim(0, 0.70)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

plt.tight_layout()
plt.savefig('./figures/recall_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 3: F1@k Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
rects1 = ax.bar(x - w, bm25_f1, w, label='BM25 Baseline', color=c_bm25, edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, method_a_f1, w, label='Method A: Weighted Zone', color=c_meth_a, edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + w, method_b_f1, w, label='Method B: Linear SVM LTR', color=c_meth_b, edgecolor='black', alpha=0.9)

ax.set_ylabel('F1 Score@k')
ax.set_xlabel('Rank Cutoff (k)')
ax.set_title('F1 Score Comparison across Retrieval Models (Test Queries)')
ax.set_xticks(x)
ax.set_xticklabels([f'k = {k}' for k in cutoffs])
ax.set_ylim(0, 0.45)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, weight='bold')

plt.tight_layout()
plt.savefig('./figures/f1_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 4: MAP & MRR Benchmark Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
models = ['BM25 Baseline', 'Method A\n(Weighted Zone)', 'Method B\n(Linear SVM LTR)']
map_scores = [0.4291, 0.3473, 0.3926]
mrr_scores = [0.8310, 0.7145, 0.8346]

x_m = np.arange(len(models))
w_m = 0.35

b1 = ax.bar(x_m - w_m/2, map_scores, w_m, label='Mean Average Precision (MAP)', color='#3B82F6', edgecolor='black', alpha=0.9)
b2 = ax.bar(x_m + w_m/2, mrr_scores, w_m, label='Mean Reciprocal Rank (MRR)', color='#10B981', edgecolor='black', alpha=0.9)

ax.set_ylabel('Score')
ax.set_title('MAP and MRR Performance Comparison')
ax.set_xticks(x_m)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.95)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

for b in [b1, b2]:
    for bar in b:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, weight='bold')

plt.tight_layout()
plt.savefig('./figures/map_mrr_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 5: Method A Zone Weights Donut Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
labels = ['Title (.T)\nw = 0.55 (55%)', 'Text Body (.W)\nw = 0.40 (40%)', 'Bibliography (.B)\nw = 0.05 (5%)', 'Author (.A)\nw = 0.00 (0%)']
weights_zone = [0.55, 0.40, 0.05, 0.00]
colors_zone = ['#2563EB', '#059669', '#F59E0B', '#9CA3AF']
explode = (0.05, 0.05, 0.05, 0.1)

wedges, texts, autotexts = ax.pie(
    [0.55, 0.40, 0.05, 0.001], # small epsilon for author visual slice
    explode=explode,
    labels=['Title (55%)', 'Body (40%)', 'Bib (5%)', 'Author (0%)'],
    autopct='%1.0f%%',
    pctdistance=0.75,
    startangle=140,
    colors=colors_zone,
    wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1)
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
    autotext.set_fontsize(9)

ax.set_title('Method A: Optimal Learned Document Zone Weights\n(Simplex Grid Search, Best Train F1@10 = 0.2976)', fontsize=11)
plt.tight_layout()
plt.savefig('./figures/zone_weights_donut.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 6: Method B 22-Feature Standardized Weights Bar Chart
# -------------------------------------------------------------
features_list = [
    ("query term number", -0.1648),
    ("query term ratio", -0.2844),
    ("stream length", 0.0369),
    ("idf", -0.0507),
    ("sum of term frequency", 0.0044),
    ("min of term frequency", 0.0631),
    ("max of term frequency", -0.0213),
    ("mean of term frequency", 0.1949),
    ("variance of term frequency", -0.1227),
    ("sum norm tf", 0.0839),
    ("min norm tf", 0.0434),
    ("max norm tf", 0.1270),
    ("mean norm tf", -0.0787),
    ("variance norm tf", -0.1050),
    ("sum of tf*idf", 0.0334),
    ("min of tf*idf", -0.2059),
    ("max of tf*idf", -0.2610),
    ("mean of tf*idf", 0.0447),
    ("variance of tf*idf", 0.1780),
    ("boolean model", -0.0191),
    ("vector space model", 0.2171),
    ("BM25", 0.2059),
]

features_list_sorted = sorted(features_list, key=lambda x: x[1])
names = [x[0] for x in features_list_sorted]
vals = [x[1] for x in features_list_sorted]
bar_colors = ['#DC2626' if v < 0 else '#16A34A' for v in vals]

fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, vals, align='center', color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.88)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8.5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Standardized Weight in Linear SVM (w_std)')
ax.set_title('Method B: Learned Real-Valued Weights across 22 IR Features\n(Linear SVM with Balanced Class Weights, C=0.01)')
ax.set_xlim(-0.32, 0.25)

for bar in bars:
    w_val = bar.get_width()
    offset = 0.008 if w_val >= 0 else -0.008
    ha = 'left' if w_val >= 0 else 'right'
    ax.annotate(f'{w_val:+.4f}', xy=(w_val + offset, bar.get_y() + bar.get_height()/2),
                va='center', ha=ha, fontsize=7.5, weight='bold')

plt.tight_layout()
plt.savefig('./figures/svm_feature_weights.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 7: Regularization Parameter C Sensitivity
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
c_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
c_labels = ['1e-4', '1e-3', '1e-2\n(Selected)', '1e-1', '1.0']
p10_vals = [0.2644, 0.2600, 0.2600, 0.2600, 0.2600]
map_vals = [0.3871, 0.3849, 0.3766, 0.3738, 0.3736]
mrr_vals = [0.8115, 0.8174, 0.8193, 0.8193, 0.8193]

ax.plot(range(len(c_values)), map_vals, marker='o', linewidth=2, color='#2563EB', label='MAP')
ax.plot(range(len(c_values)), mrr_vals, marker='s', linewidth=2, color='#10B981', label='MRR')
ax.plot(range(len(c_values)), p10_vals, marker='^', linewidth=2, color='#F59E0B', label='P@10')

ax.set_xticks(range(len(c_values)))
ax.set_xticklabels(c_labels)
ax.set_xlabel('Linear SVM Regularization Parameter (C)')
ax.set_ylabel('Evaluation Score')
ax.set_title('Sensitivity of Linear SVM to Regularization Parameter C')
ax.set_ylim(0.20, 0.90)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='center right')

for idx in range(len(c_values)):
    ax.annotate(f'{map_vals[idx]:.4f}', xy=(idx, map_vals[idx]), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=8)
    ax.annotate(f'{mrr_vals[idx]:.4f}', xy=(idx, mrr_vals[idx]), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('./figures/svm_c_sensitivity.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# FIGURE 8: Precision-Recall Curve Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
ax.plot(bm25_r, bm25_p, marker='o', linewidth=2.5, color=c_bm25, label='BM25 Baseline')
ax.plot(method_a_r, method_a_p, marker='s', linewidth=2.5, color=c_meth_a, label='Method A: Weighted Zone')
ax.plot(method_b_r, method_b_p, marker='^', linewidth=2.5, color=c_meth_b, label='Method B: Linear SVM LTR')

# Annotate points k=5, 10, 20
for r, p, k in zip(bm25_r, bm25_p, cutoffs):
    ax.annotate(f'k={k}', xy=(r, p), xytext=(5, 5), textcoords='offset points', fontsize=8, color=c_bm25, weight='bold')
for r, p, k in zip(method_a_r, method_a_p, cutoffs):
    ax.annotate(f'k={k}', xy=(r, p), xytext=(5, -12), textcoords='offset points', fontsize=8, color=c_meth_a, weight='bold')
for r, p, k in zip(method_b_r, method_b_p, cutoffs):
    ax.annotate(f'k={k}', xy=(r, p), xytext=(-25, 4), textcoords='offset points', fontsize=8, color=c_meth_b, weight='bold')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Characteristics across Rank Cutoffs (k=5, 10, 20)')
ax.set_xlim(0.28, 0.65)
ax.set_ylim(0.12, 0.50)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('./figures/precision_recall_curves.png', dpi=300)
plt.close()

print('All 8 publication-quality figures successfully generated in ./figures directory!')

