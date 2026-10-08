"""Runs the full analysis pipeline in order; outputs are written to out/."""
import subprocess, sys
for script in ['analysis_awareness.py', 'multinomial_classes.py', 'kmeans_compare.py', 'make_figures.py']:
    print(f'=== {script}', flush=True)
    subprocess.run([sys.executable, script], check=True)
print('All results and figures are in out/')
