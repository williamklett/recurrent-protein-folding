#!/usr/bin/env python3
import csv
import json
import numpy as np
from pathlib import Path

# Need to look at the relation between convergence speed and energy drop rate.
# What data do we have?
# lj_energy_results.csv has: name, tm, E_first, E_final, delta_E, min_E, monotone_frac
# The user says "convergence charts" show proteins converging faster decrease energy slower.
# Let's read lj_energy_results.csv and see what's there.

# The user is making an observation from "the convergence charts" (PCE vs TM, etc)
# that faster converging proteins tend to decrease energy slower.
# We don't have "convergence speed" directly in lj_energy_results.csv, we have it in PCE maybe?
# Wait, PCE is the positive curvature energy (late, abrupt convergence = high PCE).
# Smooth, early convergence = low PCE.
# Let's merge lj_energy_results.csv with general_convergence_summary.csv or whatever has PCE.

data_lj = pd.read_csv('runs/lj_energy_results_full.csv')
pce_file = Path('runs/general_convergence_ood500_native_cw30_stream/general_convergence_summary.csv')

if pce_file.exists():
    data_pce = pd.read_csv(pce_file)
    # The 'name' in lj_energy is like 'benchmark_1AO6', in pce it might be '1AO6'.
    # Let's check how they match up.
    print(data_pce.head())
else:
    print("PCE summary not found at", pce_file)
