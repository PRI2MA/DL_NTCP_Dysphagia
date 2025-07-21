"""
Create overview of all experiments' results.csv and save as results_overview.csv.
"""
import os
import pandas as pd
import config

# Initialize variables
exp_root_dir = config.exp_root_dir
filename_results_csv = config.filename_results_csv

# Get all exp folders
all_exps = os.listdir(exp_root_dir)

# Combine results.csv's
df = pd.DataFrame()
for exp in all_exps:
    path_i = os.path.join(exp_root_dir, exp)
    try:
        df_i = pd.read_csv(os.path.join(path_i, filename_results_csv), sep=';', index_col=0).T
        df = pd.concat([df, df_i], axis=0)
    except:
        print('{path} has no {file}.'.format(path=os.path.join(exp_root_dir, exp), file=filename_results_csv))

# Save as csv file
df.to_csv(os.path.join(config.root_path, config.filename_results_overview_csv), sep=';')


