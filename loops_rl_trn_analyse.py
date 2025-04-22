#%%
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import symusic
import muspy
import numpy as np
import IPython.display as ipd
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import seaborn as sns

run_path = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-ce-0.04-1-200"
#%%
logs = glob.glob(run_path + "/rl_logs/**/*.parquet", recursive=True)
logs = [pd.read_parquet(log) for log in logs]
logs = pd.concat(logs)
for col in logs.columns:
    if logs[col].apply(lambda x: isinstance(x, dict)).all():
        logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
        logs.drop(col, axis=1, inplace=True)
#%%

# use seaborn dark theme
sns.set_theme(style="darkgrid")
# hide grid lines

# set colours to Set1
# plot scatterplot of reward_step vs normalized_rewards_programs_iou
plt.figure(figsize=(8,4))
# have only horizontal grid lines
plt.grid(axis='x')
plt.scatter(logs["reward_step"], logs["aes_scores_CE"], alpha=0.2, s=5.0)
plt.xlabel("Iteration")
plt.ylabel("Predicted Content Enjoyment")
# add some spacing between title and figure
plt.gca().set_title("Predicted content enjoyment across SMART training iterations", fontsize=14, fontweight='bold', pad=15)

# plt.title("Predicted content enjoyment across SMART training iterations", fontsize=14, fontweight='bold')
plt.xlim(0, 200)
plt.ylim(1, 10)

# save to plots as png and pdf
plt.savefig("plots/predicted_content_enjoyment.png", dpi=300, bbox_inches='tight')
plt.savefig("plots/predicted_content_enjoyment.pdf", dpi=300, bbox_inches='tight')

# plt.grid()
plt.show()

#%%

# if normalized_rewards_programs_iou is missing, add it as 0
if "normalized_rewards_programs_iou" not in logs.columns:
    logs["normalized_rewards_programs_iou"] = 0

#%%    
midi = []
midi_paths = glob.glob( run_path + f"/midi/**/*.mid", recursive=True)
# create records with 
midi.extend([{"midi_path": m, "reward_step": int(m.split("/")[-2].split("_")[0]), "idx" : int(m.split("_")[-1].replace(".mid","")), "symusic" : symusic.Score(m), "muspy": muspy.read_midi(m) } for m in tqdm(midi_paths)])

midi = pd.DataFrame(midi)

#%%

# print logs.columns
logs = logs.merge(midi, on=["reward_step", "idx"], how="inner")
normalized_rewards = [col for col in logs.columns if "normalized_rewards" in col]

# %%
# print columns
# count logs
print(len(logs))
# %%
