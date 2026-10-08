"""
multinomial_classes.py
======================
Multinomial logistic regression of latent-class membership (modal assignment) on socio-demographic
and access variables, with cluster-robust (CR1) standard errors by data collector.
Reference class: the medicalised-aware class (highest-numbered class). Outputs out/multinomial.json.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats, optimize
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
d = pd.read_csv('out/clean_with_scores.csv')
PRED = {'age': '20 to 29', 'gender': 'Male', 'education': 'Tertiary', 'occupation': 'Student', 'residence': 'Urban area',
        'checkups': 'Occasionally', 'info_ease': 'Somewhat easy', 'internet': 'Yes'}
Xs, names = [np.ones(len(d))], ['Intercept']
for v, ref in PRED.items():
    for lv in sorted(d[v].unique()):
        if lv != ref: Xs.append((d[v] == lv).astype(float).values); names.append(f'{v}: {lv}')
X = np.column_stack(Xs); n, p = X.shape
K = d.lclass.nunique(); REF = K - 1
y = d.lclass.values; Y = np.eye(K)[y]
others = [k for k in range(K) if k != REF]
def unpack(b): B = np.zeros((p, K)); B[:, others] = b.reshape(K - 1, p).T; return B   # class-major parameter vector
def nll(b):
    E = X @ unpack(b); E -= E.max(1, keepdims=True); P = np.exp(E); P /= P.sum(1, keepdims=True)
    return -np.sum(Y * np.log(P + 1e-300)), P
def grad(b):
    _, P = nll(b); return -(X.T @ (Y - P))[:, others].T.ravel()
def hess(P):
    # observed = expected information for the canonical (softmax) link
    H = np.zeros((p * (K - 1), p * (K - 1)))
    for i, a in enumerate(others):
        for j, c in enumerate(others):
            w = P[:, a] * ((a == c) - P[:, c])
            H[i * p:(i + 1) * p, j * p:(j + 1) * p] = X.T @ (X * w[:, None])
    return H
# Newton-Raphson with step halving
b = np.zeros(p * (K - 1)); f, P = nll(b); converged = False
for it in range(200):
    step = np.linalg.solve(hess(P), -grad(b)); t = 1.0
    while True:
        f_new, P_new = nll(b + t * step)
        if f_new <= f + 1e-12 or t < 1e-6: break
        t /= 2
    b = b + t * step; df_ = f - f_new; f, P = f_new, P_new
    if np.abs(t * step).max() < 1e-9 or df_ < 1e-12: converged = True; break
class _R: pass
res = _R(); res.success = converged
H = hess(P); Hinv = np.linalg.inv(H)
S = np.column_stack([X * (Y[:, a] - P[:, a])[:, None] for a in others])   # per-observation scores
g = d.collector.values; ug = pd.Series(np.arange(n)).groupby(g).apply(list)
meat = sum(np.outer(S[ix].sum(0), S[ix].sum(0)) for ix in ug)
G = len(ug); V = Hinv @ meat @ Hinv * G / (G - 1) * (n - 1) / (n - p * (K - 1))
se = np.sqrt(np.diag(V))
out = {'reference_class': int(REF), 'n': int(n), 'clusters': int(G), 'converged': bool(res.success),
       'mcfadden_r2': float(1 - nll(b)[0] / (-np.sum(Y * np.log(Y.mean(0) + 1e-300)))), 'rrr': {}}
for i, a in enumerate(others):
    out['rrr'][int(a)] = {}
    for j, nm in enumerate(names):
        k = i * p + j; z = b[k] / se[k]
        out['rrr'][int(a)][nm] = {'est': float(np.exp(b[k])), 'lo': float(np.exp(b[k] - 1.96 * se[k])), 'hi': float(np.exp(b[k] + 1.96 * se[k])),
                                  'p': float(2 * stats.norm.sf(abs(z)))}
json.dump(out, open('out/multinomial.json', 'w'), indent=2)
print('converged', res.success, 'McFadden R2', round(out['mcfadden_r2'], 3))
for a in out['rrr']:
    sig = {k: (round(v['est'], 2), round(v['p'], 3)) for k, v in out['rrr'][a].items() if v['p'] < 0.05 and k != 'Intercept'}
    print('class', a, 'vs', REF, sig)
