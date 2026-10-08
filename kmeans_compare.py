"""
kmeans_compare.py
=================
Stage-1 comparison of the two clustering approaches for awareness profiling:
K-means on the MCA coordinates (the clustering used in the original CIRF design) versus latent class
analysis (model-based clustering of the categorical responses). Reports within-cluster sum of squares
(elbow), silhouette, adjusted Rand index and normalised mutual information with the LCA solution.
Output: out/kmeans.json and out/fig9_kmeans.png
"""
import os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
d = pd.read_csv('out/clean_with_scores.csv')
ITEMS = ['heard', 'understanding_c', 'normal_ageing', 'community_view', 'usual_response', 'open_talk']
Z = pd.get_dummies(d[ITEMS].astype(str), prefix_sep='=').astype(float); Q = len(ITEMS); n = len(d)
P = Z.values / (n * Q); r = P.sum(1); c = P.sum(0); S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, sig, Vt = np.linalg.svd(S, full_matrices=False); lam = sig ** 2
keep = int((lam > 1 / Q).sum())                                   # dimensions above the 1/Q threshold
Fc = ((U * sig) / np.sqrt(r)[:, None])[:, :keep]
out = {'mca_dimensions_used': keep, 'by_k': {}}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(Fc)
    out['by_k'][k] = {'wcss': float(km.inertia_), 'silhouette': float(silhouette_score(Fc, km.labels_)),
                      'ari_vs_lca': float(adjusted_rand_score(d.lclass, km.labels_)), 'nmi_vs_lca': float(normalized_mutual_info_score(d.lclass, km.labels_))}
km4 = KMeans(n_clusters=4, n_init=50, random_state=0).fit(Fc); lab = km4.labels_
ct = pd.crosstab(d.lclass, lab)
out['crosstab_lca_rows_kmeans_cols'] = ct.values.tolist()
out['kmeans4_sizes_pct'] = (pd.Series(lab).value_counts(normalize=True).sort_index() * 100).round(1).tolist()
out['kmeans4_dai_means'] = [float(d.DAI[lab == j].mean()) for j in range(4)]
# best one-to-one matching agreement (Hungarian on the cross-table)
from scipy.optimize import linear_sum_assignment
ri, ci = linear_sum_assignment(-ct.values); out['kmeans4_matched_agreement_pct'] = float(ct.values[ri, ci].sum() / n * 100)
json.dump(out, open('out/kmeans.json', 'w'), indent=2)
print(json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in out['by_k'].items()}, indent=0))
print('dims', keep, 'sizes', out['kmeans4_sizes_pct'], 'agreement', round(out['kmeans4_matched_agreement_pct'], 1)); print(ct)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
ks = list(out['by_k'])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
axs[0].plot(ks, [out['by_k'][k]['wcss'] for k in ks], 'o-', color='#1f4e79'); axs[0].set_xlabel('Number of clusters (K-means)'); axs[0].set_ylabel('Within-cluster sum of squares')
ax2 = axs[0].twinx(); ax2.plot(ks, [out['by_k'][k]['silhouette'] for k in ks], 's--', color='#c55a11'); ax2.set_ylabel('Silhouette', color='#c55a11'); ax2.spines['top'].set_visible(False)
axs[0].set_title('(a) K-means: elbow and silhouette', fontsize=11)
axs[1].plot(ks, [out['by_k'][k]['ari_vs_lca'] for k in ks], 'o-', color='#548235', label='Adjusted Rand index')
axs[1].plot(ks, [out['by_k'][k]['nmi_vs_lca'] for k in ks], 's-', color='#7f6000', label='Normalised mutual information')
axs[1].set_ylim(0, 1); axs[1].set_xlabel('Number of clusters (K-means)'); axs[1].set_ylabel('Agreement with latent classes'); axs[1].legend(frameon=False)
axs[1].set_title('(b) Agreement with the four latent classes', fontsize=11)
fig.tight_layout(); fig.savefig('out/fig9_kmeans.png', dpi=300)
