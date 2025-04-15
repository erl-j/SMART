#%%
import glob

src_dir = "soundfont_renders"

# find all direct subdirectories of src_dir
subdirs = [d for d in glob.glob(src_dir + "/*") if d != src_dir and d != src_dir + "/.DS_Store"]
print("Subdirectories found:")
for subdir in subdirs:
    print(subdir)

# soundfonts to icnlude
soundfonts = ["musescore", "fluid", "sgm", "yamaha", "grandeur"]
# exlclude subdirs that dont contain soundfonts
subdirs = [subdir for subdir in subdirs if any(sf in subdir for sf in soundfonts)]

# for each subdirectory, get parquet logs found in <dir>/rl_logs/0/logs.parquet. add column with subdir name
import pandas as pd
import os

def get_parquet_logs(subdir):
    logs_dir = os.path.join(subdir, "rl_logs", "0")
    logs_file = os.path.join(logs_dir, "logs.parquet")
    if os.path.exists(logs_file):
        df = pd.read_parquet(logs_file)
        df["subdir"] = subdir
        return df
    else:
        print(f"Logs file not found: {logs_file}")
        return None
    
dfs = []
for subdir in subdirs:
    df = get_parquet_logs(subdir)
    if df is not None:
        dfs.append(df)

# concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)
logs = df

#%%
# for each subdir, plot 
for col in logs.columns:
    if logs[col].apply(lambda x: isinstance(x, dict)).all():
        logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
        logs.drop(col, axis=1, inplace=True)

# plot distribution of normalized_rewards_CE for each subdir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# print columns
print("Columns found:")
for col in logs.columns:
    print(col)

#%%
# create a row for each score type
# make a 4 by 1 subplot
fig, axs = plt.subplots(4, 1, figsize=(10, 20))
# for each score type, make box plos
# %%

# add a column taking the mean of the aes_scores_ columns
logs['aes_scores_average'] = logs.filter(like='aes_scores_').mean(axis=1)

aes_full_names = {
    "CE": "content enjoyment",
    "CU": "content usefulness",
    "PQ": "production quality",
    "PC": "production complexity"
}

# create a latex table for the mean across all aes_scores_ columns per subdir (rows)

# Extract the soundfont name from the subdir path for better table labels
logs['soundfont'] = logs['subdir'].apply(lambda x: x.split('/')[-1])

# Identify all columns that start with "aes_scores_"
aes_score_cols = [col for col in logs.columns if col.startswith('aes_scores_')]

# Create a table with means of all aesthetic scores grouped by soundfont
mean_scores = logs.groupby('soundfont')[aes_score_cols].mean()

# Clean up column names for the table
# This removes the "aes_scores_" prefix for cleaner headers
mean_scores.columns = [col.replace('aes_scores_', '') for col in mean_scores.columns]

# Rename columns to full names
mean_scores.rename(columns=aes_full_names, inplace=True)

# sort soundfonts by average
mean_scores['average'] = mean_scores.mean(axis=1)
mean_scores.sort_values(by='average', ascending=True, inplace=True)

# Generate LaTeX table with proper formatting
latex_table = mean_scores.to_latex(float_format="%.3f")


# Print the LaTeX table
print("LaTeX Table of Mean Aesthetic Scores by Soundfont:")
print(latex_table)

# You can also save it to a file if needed
with open('aesthetic_scores_table.tex', 'w') as f:
    f.write(latex_table)
print("Table saved to 'aesthetic_scores_table.tex'")

# If you want to transpose the table (soundfonts as columns)
latex_table_transposed = mean_scores.T.to_latex(float_format="%.3f")
print("\nTransposed LaTeX Table (metrics as rows, soundfonts as columns):")
print(latex_table_transposed)
# %%
