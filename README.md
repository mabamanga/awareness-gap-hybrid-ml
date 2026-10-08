# Bridging the Awareness Gap: A Hybrid Machine Learning Method for Awareness and Detection of Alzheimer's Disease in Low-awareness Communities

Data and code for the article of the same title (Moses, Bamanga, Ahmadu, Malgwi and Husseini; submitted to *Dementia: The International Journal of Social Research and Practice*). The repository reproduces every table, figure and number reported in the article from the de-identified survey data.

## Study in brief

A cross-sectional, enumerator-administered online survey of 973 adults in Lafia and its surrounding communities, Nasarawa State, Nigeria (968 after cleaning), measured awareness and perceptions of Alzheimer's disease, access to health information and socio-demographic characteristics. The hybrid machine learning method has three parts:

1. **Stage 1, unsupervised awareness profiling.** Multiple correspondence analysis (MCA) of six awareness items gives a Dementia Awareness Index; latent class analysis (LCA, fitted by EM) identifies awareness profiles, validated against K-means clustering of the MCA coordinates.
2. **Stage 2, supervised detection.** Logistic regression, random forest and XGBoost detect members of the least-aware profile from socio-demographic and access variables, using enumerator-grouped repeated cross-validation.
3. **Optimisation.** A greedy maximum-coverage algorithm selects outreach channels for the least-aware profile.

Determinants are estimated with logistic, linear and multinomial regression with cluster-robust (CR1) standard errors by enumerator, and open-ended answers are coded with a keyword dictionary.

## Repository structure

```
data/awareness_survey_deidentified.csv   de-identified survey responses (973 rows, 22 columns)
data/CODEBOOK.md                          variable descriptions
deidentify.py                             how the public file was produced from the raw export (raw file not distributed)
analysis_awareness.py                     cleaning, descriptives, MCA, LCA, regression, machine learning, coverage, text coding, sensitivity
multinomial_classes.py                    multinomial model of awareness profiles (cluster-robust)
kmeans_compare.py                         K-means validation of the latent classes (Figure 5)
make_figures.py                           Figures 1 to 4 and 6 to 8
run_all.py                                runs the full pipeline in order
out/                                      results (JSON and CSV) and figures produced by the scripts
```

## How to reproduce

Python 3.9 or later.

```
pip install -r requirements.txt
python run_all.py
```

The full run takes about 10 minutes on a laptop (the latent class models use 30 random starts and the machine-learning evaluation uses 20 repetitions of grouped cross-validation). All random seeds are fixed, so the results are identical on every run.

## Where each result comes from

| Article item | Produced by | Output |
|---|---|---|
| Table 1, Table 2 (descriptives) | `analysis_awareness.py` | `out/results.json` (`descriptives`) |
| MCA, Dementia Awareness Index, Figure 3 | `analysis_awareness.py`, `make_figures.py` | `results.json` (`mca`), `out/fig3_mca_map.png` |
| Table 3, Table 4, Figure 4 (latent classes) | `analysis_awareness.py`, `make_figures.py` | `results.json` (`lca_fit`, `lca_best`), `out/fig4_lca_profiles.png` |
| Figure 5 (K-means validation) | `kmeans_compare.py` | `out/kmeans.json`, `out/fig9_kmeans.png` |
| Table 5, Figure 6 (determinants) | `analysis_awareness.py`, `make_figures.py` | `results.json` (`determinants`), `out/fig6_forest_lowclass.png` |
| Table 6 (multinomial model) | `multinomial_classes.py` | `out/multinomial.json` |
| Table 7, Figure 7 (supervised detection) | `analysis_awareness.py`, `make_figures.py` | `results.json` (`ml`), `out/fig8_targeting.png` |
| Table 8, Figure 8 (coverage) | `analysis_awareness.py`, `make_figures.py` | `results.json` (`coverage_lowclass`), `out/fig7_coverage.png` |
| Table 9, Supplementary Table S1 (open-ended themes) | `analysis_awareness.py` | `results.json` (`text_reasons`, `text_suggestions`) |
| Sensitivity analyses | `analysis_awareness.py` | `results.json` (`sensitivity`) |

## Data protection

The raw export contained one identifier, the student matric number of the enumerator who administered each questionnaire. It has been replaced by an anonymous code (C001, C002, ...). Respondents' names and contact details were never collected. Free-text answers were screened and contain no names, telephone numbers, e-mail addresses or identification numbers. Collection times were reduced to the date. The raw export is not distributed.

## Ethics and funding

The study was approved by the Research Ethics Committee of the Faculty of Computing, Federal University of Lafia (FUL/FC/REC/2025/014). Participants gave electronic informed consent. The research was funded by the Tertiary Education Trust Fund (TETFund) through the Institution-Based Research (IBR) intervention at the Federal University of Lafia (FUL/REG/TETFund/002/VOL.VI/143).

## Licence

Code: MIT licence (see `LICENSE`). Data: Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the article when using the data or code.

## Repository

https://github.com/mabamanga/awareness-gap-hybrid-ml

## Contact

Mahmud Ahmad Bamanga (corresponding author), Department of Computer Science, Federal University of Lafia, Nigeria. mabamanga@gmail.com
