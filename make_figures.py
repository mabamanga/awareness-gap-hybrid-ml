"""make_figures.py: figures for the awareness paper, from out/results.json and out/clean_with_scores.csv."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
R = json.load(open('out/results.json')); d = pd.read_csv('out/clean_with_scores.csv')
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
NAVY, ORANGE, GREEN, GREY, GOLD, PURPLE = '#1f4e79', '#c55a11', '#548235', '#7f7f7f', '#bf8f00', '#7030a0'
K = R['lca_best']['K']; PROF = R['lca_best']['profiles']
CLASS_COL = [ORANGE, GOLD, GREEN, NAVY, PURPLE][:K]

# Fig 1 framework
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=fc, ec='#404040', lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=9.6, fontweight='bold' if bold else 'normal')
def arr(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='#404040', lw=1.2))
box(0.1, 1.7, 1.9, 1.3, f"Community survey\nn = {R['cleaning']['n_analytic']}\n6 awareness items\n8 socio-demographic\nand access items", '#ededed', True)
box(2.5, 3.0, 2.3, 1.2, 'Stage 1 (unsupervised)\nMCA: Dementia\nAwareness Index', '#dfe9f5')
box(2.5, 1.7, 2.3, 1.2, 'Stage 1 (unsupervised)\nLatent class analysis\nvalidated by K-means', '#dfe9f5')
box(2.5, 0.4, 2.3, 1.2, 'Determinants\nCluster-robust\nregression', '#dfe9f5')
for y in (3.6, 2.3, 1.0): arr(2.0, 2.35, 2.5, y)
box(5.3, 2.35, 2.4, 1.3, 'Stage 2 (supervised)\nDetection of the least\naware: LR, RF, XGBoost', '#fbe5d6')
box(5.3, 0.75, 2.4, 1.3, 'Optimisation\nGreedy maximum\ncoverage of channels', '#fbe5d6')
arr(4.8, 2.3, 5.3, 3.0); arr(4.8, 1.0, 5.3, 1.4); arr(4.8, 2.3, 5.3, 1.4)
box(8.2, 1.3, 2.7, 2.2, 'Intervention design\nwho to reach,\nthrough which channel,\nwith which message', '#e2efda', True)
arr(7.7, 3.0, 8.2, 2.6); arr(7.7, 1.4, 8.2, 2.0)
box(5.3, 3.85, 2.4, 0.65, 'Thematic coding of\nopen-ended answers', '#fff2cc'); arr(7.7, 4.15, 8.6, 3.5)
fig.tight_layout(); fig.savefig('out/fig1_framework.png', dpi=300); plt.close()

# Fig 2 awareness items (stacked horizontal bars)
items = [('heard', 'Heard of Alzheimer\'s disease', ['Yes', 'No']),
         ('understanding', 'Self-rated understanding (1 to 5)', [5, 4, 3, 2, 1]),
         ('normal_ageing', 'AD is a normal part of ageing', ['No', 'Not sure', 'Yes']),
         ('community_view', 'Community view of AD', ['As a medical condition', 'As a normal aging problem', 'As a spiritual or traditional issue', 'I don’t know']),
         ('usual_response', 'Usual community response', ['Taken to hospital', 'Ignored / considered normal', 'Taken for traditional healing', 'I don’t know']),
         ('open_talk', 'Memory loss discussed openly', ['Yes', 'Sometimes', 'No'])]
fig, ax = plt.subplots(figsize=(11, 4.6))
pal = ['#1f4e79', '#5b9bd5', '#bdd7ee', '#f4b183', '#c55a11']
for i, (v, lab, cats) in enumerate(items):
    left = 0; vc = d[v].value_counts(normalize=True) * 100
    for j, c in enumerate(cats):
        w = vc.get(c, 0); col = pal[j] if len(cats) == 5 else [pal[0], pal[2], pal[3], pal[4]][j] if len(cats) == 4 else [pal[0], pal[2], pal[4]][j] if len(cats) == 3 else [pal[0], pal[4]][j]
        ax.barh(i, w, left=left, color=col, edgecolor='white')
        SHORT = {"As a medical condition": "Medical", "As a normal aging problem": "Normal ageing", "As a spiritual or traditional issue": "Spiritual", "I don’t know": "Don't know", "Taken to hospital": "Hospital", "Ignored / considered normal": "Ignored", "Taken for traditional healing": "Traditional"}
        if w > 4.5: ax.text(left + w / 2, i, f"{SHORT.get(c, c)}\n{w:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if j in (0,) or (len(cats) > 2 and j == len(cats) - 1) else "black")
        left += w
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[1] for x in items]); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Respondents (%)')
fig.tight_layout(); fig.savefig('out/fig2_items.png', dpi=300); plt.close()

# Fig 3 MCA category map (recomputed with the same algorithm)
ITEMS = ['heard', 'understanding_c', 'normal_ageing', 'community_view', 'usual_response', 'open_talk']
Z = pd.get_dummies(d[ITEMS].astype(str), prefix_sep='=').astype(float); Q = len(ITEMS); n = len(d)
P = Z.values / (n * Q); r = P.sum(1); c = P.sum(0); S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, sig, Vt = np.linalg.svd(S, full_matrices=False); G = (Vt.T * sig) / np.sqrt(c)[:, None]
F = (U * sig) / np.sqrt(r)[:, None]
sgn = 1 if np.corrcoef(F[:, 0], (d.heard == 'Yes'))[0, 1] > 0 else -1
fig, ax = plt.subplots(figsize=(9, 6.2))
ax.axhline(0, color=GREY, lw=.6); ax.axvline(0, color=GREY, lw=.6)
short = {'heard': 'Heard', 'understanding_c': 'Understanding', 'normal_ageing': 'Normal ageing?', 'community_view': 'Community view', 'usual_response': 'Response', 'open_talk': 'Open talk'}
cols_ = dict(zip(ITEMS, [NAVY, ORANGE, GREEN, PURPLE, GOLD, GREY])); placed = []
for j, name in enumerate(Z.columns):
    it, lv = name.split('=', 1)
    ax.scatter(sgn * G[j, 0], G[j, 1], color=cols_[it], s=28)
    placed.append((sgn * G[j, 0], G[j, 1], f'{short[it]}: {lv.replace("As a ", "").replace("Taken for ", "").replace("Taken to ", "").replace(" / considered normal", "")[:24]}', cols_[it]))
MANUAL = {'Response: I don’t know': (-0.99, 0.30), 'Community view: I don’t know': (-0.99, 0.19), 'Understanding: 1 (very low)': (-0.99, -0.34),
          'Heard: No': (-0.80, -0.45), 'Normal ageing?: Not sure': (-0.58, -0.30), 'Response: traditional healing': (-0.05, 0.27),
          'Response: hospital': (0.36, 0.55), 'Normal ageing?: No': (0.05, 1.0), 'Community view: medical condition': (0.72, 0.99)}
for x0, y0, t, col in placed:
    if t in MANUAL:
        ax.annotate(t, (x0, y0), xytext=MANUAL[t], fontsize=7.6, color=col, arrowprops=dict(arrowstyle='-', color=col, lw=0.5))
    else:
        ax.annotate(t, (x0, y0), fontsize=7.6, xytext=(4, 4), textcoords='offset points', color=col)
bz = R['mca']['benzecri_pct']
ax.set_xlabel(f'Dimension 1: awareness ({bz[0]:.1f}% of adjusted inertia)'); ax.set_ylabel(f'Dimension 2: framing, medical (+) vs normal ageing (-) ({bz[1]:.1f}%)')
fig.tight_layout(); fig.savefig('out/fig3_mca_map.png', dpi=300); plt.close()

# Fig 4 LCA profiles heatmap
CLASS_NAMES = ['Unaware', 'Aware, traditional\nframing', 'Aware,\nnormalising', 'Aware, medical\nframing']
ROWS = [('heard', 'Yes', 'Heard of AD'), ('understanding_c', '1 (very low)', 'Understanding: very low'), ('understanding_c', '4 to 5 (high)', 'Understanding: high (4 to 5)'),
        ('normal_ageing', 'No', 'AD is not normal ageing'), ('normal_ageing', 'Not sure', 'Normal ageing: not sure'), ('normal_ageing', 'Yes', 'AD is normal ageing'),
        ('community_view', 'As a medical condition', 'Community view: medical'), ('community_view', 'As a normal aging problem', 'Community view: normal ageing'),
        ('community_view', 'As a spiritual or traditional issue', 'Community view: spiritual'), ('community_view', 'I don’t know', "Community view: don't know"),
        ('usual_response', 'Taken to hospital', 'Response: hospital'), ('usual_response', 'Taken for traditional healing', 'Response: traditional healing'),
        ('usual_response', 'Ignored / considered normal', 'Response: ignored'), ('usual_response', 'I don’t know', "Response: don't know"),
        ('open_talk', 'Yes', 'Memory loss discussed openly')]
M_ = np.array([[PROF[str(k)]['items'][it].get(lv, 0) for k in range(K)] for it, lv, _ in ROWS])
fig, ax = plt.subplots(figsize=(8.6, 7.4))
im = ax.imshow(M_, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(M_.shape[0]):
    for k in range(K): ax.text(k, i, f'{M_[i, k]:.2f}', ha='center', va='center', fontsize=8.5, color='white' if M_[i, k] > .55 else 'black')
ax.set_xticks(range(K)); ax.set_xticklabels([f"{CLASS_NAMES[k]}\n({PROF[str(k)]['share_pct']:.1f}%)" for k in range(K)], fontsize=9)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[2] for r in ROWS], fontsize=9); ax.xaxis.tick_top()
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=.04, label='Class-conditional probability')
fig.tight_layout(); fig.savefig('out/fig4_lca_profiles.png', dpi=300); plt.close()

# Fig 5 DAI by education and residence
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
eo = ['No formal education', 'Primary', 'Secondary', 'Tertiary', 'Postgraduate']; ro = ['Rural area', 'Semi-urban area', 'Urban area']
for a, v, o, t in [(axs[0], 'education', eo, '(a) Education'), (axs[1], 'residence', ro, '(b) Residence')]:
    a.boxplot([d.DAI[d[v] == g] for g in o], tick_labels=[g.replace(' education', '').replace(' area', '') for g in o], showfliers=False)
    a.plot(range(1, len(o) + 1), [d.DAI[d[v] == g].mean() for g in o], 'D', color=ORANGE, ms=6, label='Mean')
    a.set_title(t, fontsize=11)
axs[0].set_ylabel('Dementia Awareness Index (0 to 100)'); axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig('out/fig5_dai_groups.png', dpi=300); plt.close()

# Fig 6 forest plot: odds of belonging to the low-awareness class
T = R['determinants']['lowclass_OR']; names = [k for k in T if k != 'Intercept']
fig, ax = plt.subplots(figsize=(8.5, 7.2))
for i, k in enumerate(names):
    e = T[k]; col = ORANGE if e['p'] < 0.05 else NAVY
    ax.plot([e['lo'], e['hi']], [i, i], color=col); ax.plot(e['est'], i, 's', color=col)
ax.axvline(1, color='black', lw=.8); ax.set_xscale('log'); ax.set_yticks(range(len(names))); ax.set_yticklabels([k.replace('_', ' ') for k in names], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlabel('Odds ratio for low-awareness class (95% CI, cluster-robust; log scale)')
fig.tight_layout(); fig.savefig('out/fig6_forest_lowclass.png', dpi=300); plt.close()

# Fig 7 coverage of the low-awareness class
C = R['coverage_lowclass']
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
ks = list(C['single']); axs[0].barh(range(len(ks)), [C['single'][k] for k in ks], color=NAVY)
for i, k in enumerate(ks): axs[0].text(C['single'][k] + 1, i, f"{C['single'][k]:.0f}%", va='center')
axs[0].set_yticks(range(len(ks))); axs[0].set_yticklabels(ks, fontsize=9); axs[0].invert_yaxis(); axs[0].set_xlim(0, 100); axs[0].set_xlabel('Low-awareness class reachable (%)')
axs[0].set_title('(a) Single channels', fontsize=11)
steps = C['greedy']; axs[1].bar(range(len(steps) + 1), [0] + [s['cumulative_pct'] for s in steps], color=[GREY] + [GREEN] * len(steps))
axs[1].set_xticks(range(len(steps) + 1)); axs[1].set_xticklabels(['None'] + [f"+ {s['add'].split(' (')[0]}" for s in steps], fontsize=9)
for i, s in enumerate(steps): axs[1].text(i + 1, s['cumulative_pct'] + 1.5, f"{s['cumulative_pct']:.0f}%", ha='center')
axs[1].set_ylim(0, 100); axs[1].set_ylabel('Cumulative coverage (%)'); axs[1].set_title('(b) Greedy maximum coverage', fontsize=11)
fig.tight_layout(); fig.savefig('out/fig7_coverage.png', dpi=300); plt.close()

# Fig 8 ML targeting summary (AUC and lift)
M = R['ml']['summary']; ms = list(M)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
axs[0].bar(ms, [M[m]['auc'][0] for m in ms], yerr=[M[m]['auc'][1] for m in ms], color=[NAVY, GREEN, ORANGE], capsize=4)
axs[0].axhline(0.5, color='black', ls='--', lw=.8); axs[0].set_ylim(0.45, 0.75); axs[0].set_ylabel('AUC (grouped CV, 20 repeats)'); axs[0].set_title('(a) Discrimination', fontsize=11)
qs = [0, 20, 30, 50, 100]
for m, col in zip(ms, [NAVY, GREEN, ORANGE]):
    axs[1].plot(qs, [0, M[m]['lift20'][0] * 100, M[m]['lift30'][0] * 100, M[m]['lift50'][0] * 100, 100], 'o-', color=col, label=m)
axs[1].plot([0, 100], [0, 100], 'k--', lw=.8, label='Random targeting')
axs[1].set_xlabel('Population contacted, highest predicted risk first (%)'); axs[1].set_ylabel('Low-awareness class reached (%)'); axs[1].legend(frameon=False, fontsize=9)
axs[1].set_title('(b) Cumulative gains', fontsize=11)
fig.tight_layout(); fig.savefig('out/fig8_targeting.png', dpi=300); plt.close()
print('figures done')
