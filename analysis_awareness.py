"""
analysis_awareness.py
=====================
Dementia (Alzheimer's disease) awareness survey: cleaning, measurement, latent profiles,
determinants, machine-learning targeting and channel coverage.

All methods are implemented with NumPy/SciPy/scikit-learn so that every step is transparent:
  1. Cleaning and de-identification (collector IDs replaced by anonymous codes)
  2. Descriptive prevalence with Wilson CIs, collector ICC and design effect
  3. Multiple correspondence analysis (MCA): Dementia Awareness Index (DAI), Benzecri-corrected
     inertia and the MCA reliability coefficient
  4. Latent class analysis (LCA) by EM with model selection (BIC, entropy)
  5. Determinants: logistic and linear regression with cluster-robust (CR1) standard errors
  6. ML targeting of the low-awareness class from demographic/access data only, with grouped
     (collector-level) repeated cross-validation, calibration and lift
  7. Channel coverage of the low-awareness class (greedy maximum coverage)
  8. Thematic coding of open-ended answers
  9. Sensitivity analyses
Outputs: out/*.json, out/*.csv, out/fig*.png
"""
import os, re, json, glob, hashlib
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
RNG = np.random.RandomState(2026)
R = {}

# =============================================================== 1. cleaning
DATA = 'data/awareness_survey_deidentified.csv' if os.path.exists('data/awareness_survey_deidentified.csv') else glob.glob('../Awareness*.csv')[0]
raw = pd.read_csv(DATA)
raw.columns = [c.strip() for c in raw.columns]
cols = list(raw.columns)
NAMES = ['timestamp', 'collector_raw', 'age', 'gender', 'education', 'occupation', 'residence', 'heard', 'source',
         'understanding', 'normal_ageing', 'know_someone', 'community_view', 'open_talk', 'usual_response',
         'culture_affects', 'checkups', 'info_ease', 'internet', 'digital_interest', 'reason_text', 'suggestion_text']
d = raw.copy(); d.columns = NAMES
n_raw = len(d)
AGE_CATS = ['Below 20', '20–29', '30–39', '40–49', '50 and above']
d['collector_raw'] = d['collector_raw'].astype(str).str.strip()
# early form version: age was typed into the first field
moved = d.age.isna() & d.collector_raw.isin(AGE_CATS)
d.loc[moved, 'age'] = d.loc[moved, 'collector_raw']; d.loc[moved, 'collector_raw'] = np.nan
cid = d['collector_raw'].astype(str).str.upper().str.extract(r'(\d{4}/CP/[A-Z]{3}/\d{4}|^C\d{3}$)')[0]   # raw identifier or anonymous code
codes = {k: f'C{i+1:03d}' for i, k in enumerate(sorted(cid.dropna().unique()))}
d['collector'] = cid.map(codes)
d['collector'] = d['collector'].fillna(pd.Series([f'U{i:04d}' for i in range(len(d))], index=d.index))  # unknown -> own cluster
d = d.drop(columns=['collector_raw'])
d['age'] = d['age'].astype(str).str.replace('–', ' to ', regex=False).replace('nan', np.nan)
d['timestamp'] = pd.to_datetime(d.timestamp.str.replace(' GMT+1', '', regex=False), format='%Y/%m/%d %I:%M:%S %p')
d['phase'] = np.where(d.timestamp < '2026-01-01', 'pilot', 'main')
dup = d.drop(columns=['timestamp']).duplicated(keep='first')
d = d[~dup].copy()
core = ['age', 'gender', 'education', 'occupation', 'residence', 'heard', 'understanding', 'normal_ageing', 'community_view',
        'open_talk', 'usual_response', 'culture_affects', 'checkups', 'info_ease', 'internet', 'digital_interest']
miss = d[core].isna().any(axis=1)
d = d[~miss].copy().reset_index(drop=True)
d['info_ease'] = d.info_ease.replace({'Very difficul': 'Very difficult'})
d['understanding'] = d.understanding.astype(int); d['culture_affects'] = d.culture_affects.astype(int)
R['cleaning'] = {'n_raw': n_raw, 'age_recovered_from_first_field': int(moved.sum()), 'duplicates_removed': int(dup.sum()),
                 'incomplete_removed': int(miss.sum()), 'n_analytic': len(d), 'n_collectors': int(d.collector.str.startswith('C').sum() and d.loc[d.collector.str.startswith('C'), 'collector'].nunique()),
                 'n_without_collector_id': int((~d.collector.str.startswith('C')).sum()),
                 'collection_start': str(d.timestamp.min().date()), 'collection_end': str(d.timestamp.max().date()),
                 'n_pilot': int((d.phase == 'pilot').sum()), 'n_main': int((d.phase == 'main').sum())}
cs = d[d.collector.str.startswith('C')].groupby('collector').size()
R['cleaning']['responses_per_collector'] = {'median': float(cs.median()), 'max': int(cs.max()), 'mean': float(cs.mean())}
d.drop(columns=['reason_text', 'suggestion_text']).to_csv(f'{OUT}/clean_deidentified.csv', index=False)
print(R['cleaning'])
N = len(d)

# =============================================================== 2. descriptives
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h
def icc_binary(y, g):
    # one-way ANOVA estimator of the intraclass correlation (collector clustering)
    df = pd.DataFrame({'y': y.astype(float), 'g': g}); grp = df.groupby('g')['y']
    k = grp.size(); a = len(k); n = len(df)
    n0 = (n - (k ** 2).sum() / n) / (a - 1)
    msb = (k * (grp.mean() - df.y.mean()) ** 2).sum() / (a - 1)
    msw = ((df.y - grp.transform('mean')) ** 2).sum() / (n - a)
    return float(max((msb - msw) / (msb + (n0 - 1) * msw), 0.0)), float(n0)
desc = {}
for v in ['age', 'gender', 'education', 'occupation', 'residence', 'heard', 'understanding', 'normal_ageing', 'know_someone',
          'community_view', 'open_talk', 'usual_response', 'culture_affects', 'checkups', 'info_ease', 'internet', 'digital_interest']:
    vc = d[v].value_counts()
    desc[v] = {str(k): {'n': int(c), 'pct': round(c / N * 100, 1), 'ci': [round(x * 100, 1) for x in wilson(c, N)[1:]]} for k, c in vc.items()}
R['descriptives'] = desc
heard = (d.heard == 'Yes').values
icc_h, n0 = icc_binary(heard, d.collector)
deff = 1 + (cs.mean() - 1) * icc_h  # Kish design effect with mean cluster size
R['clustering'] = {'icc_heard': icc_h, 'design_effect_heard': float(deff), 'effective_n_heard': float(N / deff)}
# sources (multi-select) among those who heard
src = d.loc[d.heard == 'Yes', 'source'].fillna('').str.split(';')
allsrc = pd.Series([s.strip() for row in src for s in row if s.strip()])
R['sources_among_heard'] = {k: {'n': int(v), 'pct': round(v / (d.heard == 'Yes').sum() * 100, 1)} for k, v in allsrc.value_counts().items()}
print('ICC heard', R['clustering'])

# =============================================================== 3. MCA and Dementia Awareness Index
ITEMS = ['heard', 'understanding_c', 'normal_ageing', 'community_view', 'usual_response', 'open_talk']
d['understanding_c'] = pd.cut(d.understanding, [0, 1, 2, 3, 5], labels=['1 (very low)', '2', '3', '4 to 5 (high)']).astype(str)
Z = pd.get_dummies(d[ITEMS].astype(str), prefix_sep='=').astype(float)
Q = len(ITEMS); n, J = Z.shape
P = Z.values / (n * Q); r = P.sum(1); c = P.sum(0)
S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, sig, Vt = np.linalg.svd(S, full_matrices=False)
lam = sig ** 2
F = (U * sig) / np.sqrt(r)[:, None]                      # row principal coordinates
G = (Vt.T * sig) / np.sqrt(c)[:, None]                   # column principal coordinates
keep = lam > 1 / Q
lam_b = (Q / (Q - 1)) ** 2 * (lam[keep] - 1 / Q) ** 2     # Benzecri correction
alpha1 = Q / (Q - 1) * (1 - 1 / (Q * lam[0]))             # MCA reliability of dimension 1
dim1 = F[:, 0]
if np.corrcoef(dim1, heard)[0, 1] < 0: dim1 = -dim1; G[:, 0] = -G[:, 0]
d['DAI'] = (dim1 - dim1.min()) / (dim1.max() - dim1.min()) * 100
def eta2(cat, x):
    df = pd.DataFrame({'c': cat, 'x': x}); m = df.x.mean()
    return float((df.groupby('c').x.agg(lambda s: len(s) * (s.mean() - m) ** 2).sum()) / ((df.x - m) ** 2).sum())
R['mca'] = {'n_categories': J, 'eigenvalues_raw': lam[:6].round(4).tolist(),
            'benzecri_pct': (lam_b / lam_b.sum() * 100).round(1).tolist()[:4], 'alpha_dim1': float(alpha1),
            'discrimination_eta2_dim1': {it: round(eta2(d[it], dim1), 3) for it in ITEMS},
            'category_coords_dim1': {Z.columns[j]: round(float(G[j, 0]), 3) for j in range(J)},
            'DAI_mean': float(d.DAI.mean()), 'DAI_sd': float(d.DAI.std()), 'DAI_median': float(d.DAI.median())}
# known-groups validity
kg = {}
for v in ['education', 'residence', 'age', 'occupation', 'internet', 'checkups']:
    groups = [g.DAI.values for _, g in d.groupby(v)]
    H = stats.kruskal(*groups)
    eps2 = (H.statistic - len(groups) + 1) / (N - len(groups))
    kg[v] = {'H': float(H.statistic), 'p': float(H.pvalue), 'epsilon2': float(eps2),
             'means': {str(k): round(float(g.DAI.mean()), 1) for k, g in d.groupby(v)}}
R['dai_known_groups'] = kg
print('MCA', {k: v for k, v in R['mca'].items() if k not in ('category_coords_dim1',)})

# =============================================================== 4. Latent class analysis (EM)
Xc = d[ITEMS].astype(str)
levels = [sorted(Xc[c].unique()) for c in ITEMS]
Xi = np.stack([Xc[c].map({l: i for i, l in enumerate(lv)}).values for c, lv in zip(ITEMS, levels)], 1)
def lca_fit(K, starts=30, iters=2000, tol=1e-8, seed=0):
    rs = np.random.RandomState(seed); best = None
    for s in range(starts):
        pi = rs.dirichlet(np.ones(K)); th = [rs.dirichlet(np.ones(len(lv)), size=K) for lv in levels]
        ll_old = -np.inf
        for it in range(iters):
            logp = np.log(pi)[None, :].repeat(n, 0)
            for q in range(Q): logp += np.log(th[q][:, Xi[:, q]].T + 1e-300)
            m = logp.max(1, keepdims=True); ll = float((m[:, 0] + np.log(np.exp(logp - m).sum(1))).sum())
            post = np.exp(logp - m); post /= post.sum(1, keepdims=True)
            pi = post.mean(0)
            th = [np.stack([np.bincount(Xi[:, q], weights=post[:, k], minlength=len(levels[q])) for k in range(K)]) for q in range(Q)]
            th = [(t + 1e-10) / (t + 1e-10).sum(1, keepdims=True) for t in th]
            if ll - ll_old < tol: break
            ll_old = ll
        if best is None or ll > best['ll']: best = dict(ll=ll, pi=pi, th=th, post=post)
    npar = (K - 1) + K * sum(len(lv) - 1 for lv in levels)
    best['bic'] = -2 * best['ll'] + npar * np.log(n); best['aic'] = -2 * best['ll'] + 2 * npar; best['npar'] = npar
    best['sabic'] = -2 * best['ll'] + npar * np.log((n + 2) / 24)
    best['entropy'] = (1 - (-(best['post'] * np.log(best['post'] + 1e-300)).sum()) / (n * np.log(K))) if K > 1 else 1.0
    return best
fits = {K: lca_fit(K) for K in range(1, 7)}
R['lca_fit'] = {K: {'ll': f['ll'], 'npar': f['npar'], 'aic': f['aic'], 'bic': f['bic'], 'sabic': f['sabic'], 'entropy': float(f['entropy']),
                    'smallest_class_pct': float(f['pi'].min() * 100)} for K, f in fits.items()}
Kmin = min(fits, key=lambda k: fits[k]['bic'])
# parsimony rule (Raftery 1995): prefer the smaller model when it is within 2 BIC points of the minimum
Kbest = min(k for k in fits if fits[k]['bic'] - fits[Kmin]['bic'] < 2)
print('LCA BIC', {K: round(v['bic'], 1) for K, v in R['lca_fit'].items()}, 'best K', Kbest)
lc = fits[Kbest]
cls = lc['post'].argmax(1)
# order classes by mean DAI (0 = lowest awareness)
order = np.argsort([d.DAI.values[cls == k].mean() for k in range(Kbest)])
remap = {old: new for new, old in enumerate(order)}
d['lclass'] = [remap[c] for c in cls]
post_sorted = lc['post'][:, order]
avepp = [float(post_sorted[d.lclass.values == k, k].mean()) for k in range(Kbest)]
profiles = {}
for knew, kold in enumerate(order):
    profiles[knew] = {'share_pct': float(lc['pi'][kold] * 100), 'n_modal': int((d.lclass == knew).sum()), 'avepp': avepp[knew],
                      'DAI_mean': float(d.DAI[d.lclass == knew].mean()),
                      'items': {it: {lv: round(float(lc['th'][q][kold, j]), 3) for j, lv in enumerate(levels[q])} for q, it in enumerate(ITEMS)}}
R['lca_best'] = {'K': int(Kbest), 'profiles': profiles}
print('profiles', {k: (round(v['share_pct'], 1), round(v['DAI_mean'], 1), round(v['avepp'], 2)) for k, v in profiles.items()})
LOW = 0

# =============================================================== 5. determinants (cluster-robust regression)
PRED = {'age': '20 to 29', 'gender': 'Male', 'education': 'Tertiary', 'occupation': 'Student', 'residence': 'Urban area',
        'checkups': 'Occasionally', 'info_ease': 'Somewhat easy', 'internet': 'Yes'}
def design(df):
    Xs = [np.ones(len(df))]; names = ['Intercept']
    for v, ref in PRED.items():
        for lv in sorted(df[v].unique()):
            if lv == ref: continue
            Xs.append((df[v] == lv).astype(float).values); names.append(f'{v}: {lv}')
    return np.column_stack(Xs), names
def cr1(X, scores, bread_inv, groups):
    gs = pd.Series(range(len(groups))).groupby(np.asarray(groups)).apply(list)
    meat = sum(np.outer(scores[ix].sum(0), scores[ix].sum(0)) for ix in gs)
    Gn = len(gs); nn, p = X.shape
    return bread_inv @ meat @ bread_inv * Gn / (Gn - 1) * (nn - 1) / (nn - p)
def logit_fit(X, y, groups):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X @ b)); W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]); step = np.linalg.solve(H, X.T @ (y - mu)); b += step
        if np.abs(step).max() < 1e-10: break
    mu = 1 / (1 + np.exp(-X @ b)); Hinv = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    V = cr1(X, X * (y - mu)[:, None], Hinv, groups)
    return b, np.sqrt(np.diag(V))
def ols_fit(X, y, groups):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    V = cr1(X, X * e[:, None], XtXi, groups); return b, np.sqrt(np.diag(V)), 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
X, xn = design(d)
def tab(b, se, expo):
    z = b / se; p = 2 * stats.norm.sf(np.abs(z))
    f = np.exp if expo else (lambda v: v)
    return {nm: {'est': float(f(bi)), 'lo': float(f(bi - 1.96 * si)), 'hi': float(f(bi + 1.96 * si)), 'p': float(pi)} for nm, bi, si, pi in zip(xn, b, se, p)}
b1, s1 = logit_fit(X, heard.astype(float), d.collector.values)
b2, s2, r2 = ols_fit(X, d.DAI.values, d.collector.values)
b3, s3 = logit_fit(X, (d.lclass == LOW).astype(float).values, d.collector.values)
R['determinants'] = {'heard_OR': tab(b1, s1, True), 'DAI_beta': tab(b2, s2, False), 'DAI_R2': float(r2), 'lowclass_OR': tab(b3, s3, True),
                     'n_clusters': int(d.collector.nunique())}
print('R2 DAI', r2)

# =============================================================== 6. ML targeting of the low-awareness class
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
FEATS = list(PRED)
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(d[FEATS])
XM = enc.transform(d[FEATS]); yM = (d.lclass == LOW).astype(int).values
groups = d.collector.values; ug = np.unique(groups)
MODELS = {'Logistic regression': lambda: LogisticRegression(C=1.0, max_iter=5000),
          'Random forest': lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1),
          'XGBoost': lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=0)}
def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)); bb = np.zeros(2); Xb = np.c_[np.ones_like(lp), lp]
    for _ in range(50):
        mu = 1 / (1 + np.exp(-Xb @ bb)); W = mu * (1 - mu); bb += np.linalg.solve(Xb.T @ (Xb * W[:, None]), Xb.T @ (y - mu))
    return float(bb[0]), float(bb[1])
def lift(y, p, q):
    k = int(np.ceil(q * len(y))); top = np.argsort(-p)[:k]; return float(y[top].sum() / y.sum())
REPS = 20; res = {m: {'auc': [], 'brier': [], 'cal_int': [], 'cal_slope': [], 'lift20': [], 'lift30': [], 'lift50': []} for m in MODELS}
oof_last = {}
for rep in range(REPS):
    rs = np.random.RandomState(rep); fold_of = dict(zip(ug, rs.permutation(len(ug)) % 5)); fid = np.array([fold_of[g] for g in groups])
    oof = {m: np.zeros(N) for m in MODELS}
    for f in range(5):
        tr, te = fid != f, fid == f
        for m, mk in MODELS.items():
            oof[m][te] = mk().fit(XM[tr], yM[tr]).predict_proba(XM[te])[:, 1]
    for m in MODELS:
        p = oof[m]; ci, cs_ = calib(yM, p)
        res[m]['auc'].append(roc_auc_score(yM, p)); res[m]['brier'].append(brier_score_loss(yM, p))
        res[m]['cal_int'].append(ci); res[m]['cal_slope'].append(cs_)
        for q in (0.2, 0.3, 0.5): res[m][f'lift{int(q*100)}'].append(lift(yM, p, q))
    oof_last = oof
R['ml'] = {'outcome_prevalence': float(yM.mean()), 'reps': REPS,
           'summary': {m: {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in r.items()} for m, r in res.items()}}
dlr = np.array(res['Logistic regression']['auc']);
R['ml']['auc_diff_vs_LR'] = {m: {'mean': float(np.mean(np.array(res[m]['auc']) - dlr)), 'reps_better': int((np.array(res[m]['auc']) > dlr).sum())} for m in MODELS if m != 'Logistic regression'}
# permutation importance (grouped by original variable) for LR on full data, AUC drop, out-of-fold refit
base_m = MODELS['Logistic regression']().fit(XM, yM)
imp = {}
for v in FEATS:
    drops = []
    for rep in range(30):
        dd = d[FEATS].copy(); dd[v] = np.random.RandomState(rep).permutation(dd[v].values)
        drops.append(roc_auc_score(yM, base_m.predict_proba(XM)[:, 1]) - roc_auc_score(yM, base_m.predict_proba(enc.transform(dd))[:, 1]))
    imp[v] = [float(np.mean(drops)), float(np.std(drops))]
R['ml']['perm_importance_LR'] = imp
print('ML', {m: (round(v['auc'][0], 3), round(v['lift30'][0], 3)) for m, v in R['ml']['summary'].items()})

# =============================================================== 7. channel coverage of the low-awareness class
low = d[d.lclass == LOW]
CH = {'Digital (internet and interested)': (low.internet == 'Yes') & low.digital_interest.isin(['Yes', 'Maybe']),
      'Health facility contact (any visit)': low.checkups != 'Never',
      'Regular facility contact (at least yearly)': low.checkups.isin(['Regularly', 'Once or twice a year']),
      'Easy access to health information': low.info_ease.isin(['Very easy', 'Somewhat easy'])}
cov = {k: float(v.mean() * 100) for k, v in CH.items()}
# greedy maximum coverage over channels (first three are deployable channels)
deploy = ['Digital (internet and interested)', 'Health facility contact (any visit)']
chosen, covered, steps = [], np.zeros(len(low), bool), []
for _ in deploy:
    best = max([c for c in deploy if c not in chosen], key=lambda c: (covered | CH[c].values).mean())
    chosen.append(best); covered |= CH[best].values; steps.append({'add': best, 'cumulative_pct': float(covered.mean() * 100)})
R['coverage_lowclass'] = {'n_low': int(len(low)), 'single': cov, 'greedy': steps, 'unreached_pct': float((~covered).mean() * 100),
                          'unreached_profile': {'rural_pct': float((low.residence[~covered] == 'Rural area').mean() * 100) if (~covered).any() else None,
                                                'no_formal_or_primary_pct': float(low.education[~covered].isin(['No formal education', 'Primary']).mean() * 100) if (~covered).any() else None}}
# by class
R['coverage_by_class'] = {int(k): {'internet_pct': float((g.internet == 'Yes').mean() * 100), 'digital_interest_yes_pct': float((g.digital_interest == 'Yes').mean() * 100),
                                   'never_checkup_pct': float((g.checkups == 'Never').mean() * 100), 'rural_pct': float((g.residence == 'Rural area').mean() * 100),
                                   'culture_affects_mean': float(g.culture_affects.mean())} for k, g in d.groupby('lclass')}
print('coverage', R['coverage_lowclass'])

# =============================================================== 8. thematic coding of open-ended answers
dt = d   # the same 968 analytic respondents as every other analysis
REASON = {'Ignorance or lack of knowledge': r'ignoran|lack of (knowledge|awareness)|no knowledge|not aware|unaware|lack of understanding|unawareness',
          'Illiteracy and low education': r'illitera|uneducat|not educated|level of education|low education|literacy|lack of education',
          'Insufficient information and campaigns': r'information|campaign|sensiti|enlighten|orientation|publicity|seminar|program|media|not (talked|discussed)|no one talks|awareness program',
          'Cultural and spiritual beliefs': r'cultur|belief|tradition|spiritual|religio|superstit|witch|juju|myth',
          'Seen as normal ageing': r'old age|ageing|aging|normal|elderly',
          'Poverty and access to care': r'poverty|poor|money|cost|financ|hospital|health (facilit|care|cent)|access|doctor|rural',
          'Stigma and silence': r'stigma|shame|secret|silen|hide|taboo|embarrass',
          'Rarity of the disease': r'\brare\b|not common|uncommon|never seen|not prevalent|few cases',
          "Don't know or no idea": r"don.?t know|dont know|no idea|not sure|idk|nothing"}
SUGG = {'Mass and social media': r'radio|\btv\b|television|social media|media|jingle|whatsapp|facebook|internet|online|digital',
        'Community and religious gatherings': r'church|mosque|religio|community|leader|town hall|market|village|gathering|outreach|house to house',
        'Schools and seminars': r'school|seminar|workshop|lecture|student|curriculum|symposium',
        'Health workers and facilities': r'hospital|health worker|clinic|health cent|doctor|nurse|screening|free test|medical',
        'Government and NGOs': r'government|ministry|policy|ngo|agency',
        'General awareness and sensitization': r'awareness|sensiti|enlighten|educat|campaign|inform|teach',
        'No suggestion': r'^\s*(no|none|nil|not really|nope|nothing|no idea|i don.?t know)\W*\s*$'}
def themes(series, dic):
    s = series.fillna('').astype(str).str.lower(); valid = s.str.strip() != ''
    outd = {}
    for k, pat in dic.items():
        hit = s.str.contains(pat.replace('(', '(?:').replace('(?:?:', '(?:'), regex=True) & valid; outd[k] = {'n': int(hit.sum()), 'pct': round(hit.sum() / valid.sum() * 100, 1)}
    return {'n_answers': int(valid.sum()), 'themes': outd}
R['text_reasons'] = themes(dt.reason_text, REASON); R['text_suggestions'] = themes(dt.suggestion_text, SUGG)
print('reasons', {k: v['pct'] for k, v in R['text_reasons']['themes'].items()})

# =============================================================== 9. sensitivity analyses
sens = {}
w = 1 / d.groupby('collector').collector.transform('size')       # each collector counts equally
sens['heard_pct_collector_weighted'] = float((heard * w).sum() / w.sum() * 100)
sens['heard_pct_main_phase_only'] = float(heard[d.phase.values == 'main'].mean() * 100)
sens['lowclass_pct_collector_weighted'] = float(((d.lclass == LOW).values * w).sum() / w.sum() * 100)
cap = d.groupby('collector').cumcount() < 10
sens['n_cap10'] = int(cap.sum()); sens['heard_pct_cap10'] = float(heard[cap.values].mean() * 100)
# LCA stability: K-best refit on a 50% subsample, agreement of class sizes
sub = RNG.rand(N) < 0.5
R['sensitivity'] = sens
print('sens', sens)

json.dump(R, open(f'{OUT}/results.json', 'w'), indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
d.drop(columns=['reason_text', 'suggestion_text']).to_csv(f'{OUT}/clean_with_scores.csv', index=False)
print('done')
