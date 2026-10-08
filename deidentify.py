"""
deidentify.py
=============
Creates data/awareness_survey_deidentified.csv from the raw Google Forms export (not distributed).
The only identifier in the export, the enumerator's student matric number, is replaced by an anonymous
code (C001, C002, ...) assigned in sorted order, exactly as in the analysis. In 49 pilot responses an
earlier form version stored the respondent's age group in this field; that value is kept because it is
not identifying and is needed by the cleaning step. Any other content of the field is removed.
Timestamps are kept to the date only.
"""
import os, glob, re
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
raw = pd.read_csv(glob.glob('../Awareness*.csv')[0])
col = raw.columns[1]
AGE = ['Below 20', '20–29', '30–39', '40–49', '50 and above']
val = raw[col].astype(str).str.strip()
mid = val.str.upper().str.extract(r'(\d{4}/CP/[A-Z]{3}/\d{4})')[0]
codes = {k: f'C{i+1:03d}' for i, k in enumerate(sorted(mid.dropna().unique()))}
new = mid.map(codes)
new = new.where(new.notna(), val.where(val.isin(AGE)))
out = raw.copy(); out[col] = new
out = out.rename(columns={col: 'Enumerator code'})
ts = out.columns[0]
out[ts] = out[ts].astype(str).str.replace(r'\s\d{1,2}:\d{2}:\d{2}\s[AP]M', ' 12:00:00 PM', regex=True)
os.makedirs('data', exist_ok=True)
out.to_csv('data/awareness_survey_deidentified.csv', index=False, encoding='utf-8')
assert not out.astype(str).apply(lambda c: c.str.contains(r'\d{4}/CP/', regex=True)).any().any()
print('written', len(out), 'rows;', len(codes), 'enumerator codes; matric numbers removed')
