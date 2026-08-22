import numpy as np
import pandas as pd
from pathlib import Path
import json

df_lj = pd.read_csv('runs/lj_energy_results_full.csv')
pce_file2 = Path('runs/general_convergence_n40_native/general_convergence_summary.csv')
df_pce2 = pd.read_csv(pce_file2)

# Pick an easy protein that converges early
# e.g., low min_last5 or high tm
easy = df_pce2.sort_values('min_last5').iloc[0]['protein']
print(f"Easiest protein (fastest convergence): {easy}")

# Load its energies
act_p = next(Path('runs').glob(f"**/*{easy}*/esmfold_activations.pt"))
run_dir = act_p.parent

# I don't have the energies cached directly for all, let's load what we can.
# The user ran lj_energy_results_full.csv, which has E_first and E_final.
# Let's print out the row.
print(df_lj[df_lj['name'].str.contains(easy)])

