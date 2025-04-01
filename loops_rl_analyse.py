#%%
import pandas as pd
import glob
from tqdm import tqdm

# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-16gens"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou-32samples-piano"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou-32samples-piano-random-tempo"
# run_path = "artefacts/loops-fluir3-2-iou-logstep-1e-5-beta=0.04-avg-aes-and-iou-32samples-mi-iou-0.25"
# run_path  = "artefacts/loops-fluir3-2-iou-logstep-1e-6-beta=0.04-avg-aes-and-iou-8samples-mi-iou-0.25-32k"
# run_path  = "artefacts/loops-fluir3-2-iou-logstep-1e-6-beta=0.04-avg-aes-and-iou-8samples-mi-iou-0.25-32k-ce-pq"
# run_path  = "artefacts/loops-touhou-2-iou-logstep-1e-6-beta=0.04-avg-aes-and-iou-8samples-mi-iou-0.25-32k-ce-pq"
# run_path = "artefacts/drgpo-loops-touhou-2-iou-logstep-1e-4-beta=0.04-4samples-mi-iou-0.25-32k-ce-pq-16its"
# run_path = "artefacts/piano-test-8-1e-4"
# run_path = "artefacts/new-piano-long-2-ypd"
# run_path = "artefacts/loops-long-2-ypd-house-softmax-t=1.0-scaled-6"
# run_path = "artefacts/jazz-piano-not-relative-20x-matrix"
# run_path = "artefacts/loops-matrix-iou"
# run_path = "artefacts/loops-matrix-noprompt-jazz_fusion-b=0.04"
# run_path = "artefacts/loops-matrix-noprompt-clap-judge"
# run_path = "/workspace/aestune/artefacts/loops-matrix-noprompt-clap-judge-0.25"
# run_path = "/workspace/aestune/artefacts/loops-matrix-noprompt-clap-judge-0.25-temp=0.8"
# run_path = "artefacts/loops-matrix-ds-t=1.0"
# run_path = "artefacts/loops-matrix-beta=0.08"
# run_path = "artefacts/loops-matrix-iou-bass_drums_keys"
# run_path = "artefacts/loops-matrix-iou-bass_drums_keys-2-gtr"
run_path = "artefacts/piano-clap-only-10s"
# load all logs

logs = glob.glob(run_path + "/rl_logs/**/*.parquet", recursive=True)

logs = [pd.read_parquet(log) for log in logs]
logs = pd.concat(logs)


# for all columns that are dicts, expand them and prepend dict to the key as name
for col in logs.columns:
    if logs[col].apply(lambda x: isinstance(x, dict)).all():
        logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
        logs.drop(col, axis=1, inplace=True)


# load all midi
midi_paths = glob.glob(run_path + "/midi/**/*.mid", recursive=True)

# create records with 
midi = [{"midi_path": m, "reward_step": int(m.split("/")[-2].split("_")[0]), "idx" : int(m.split("_")[-1].replace(".mid","")) } for m in tqdm(midi_paths)]



# if normalized_rewards_programs_iou is missing, add it as 0
if "normalized_rewards_programs_iou" not in logs.columns:
    logs["normalized_rewards_programs_iou"] = 0


#%%
# join logs and midi on reward_step and idx

midi = pd.DataFrame(midi)
logs = logs.merge(midi, on=["reward_step", "idx"], how="inner")

# %%
import symusic

logs["midi"] = logs["midi_path"].apply(lambda x: symusic.Score(x))
# print all columns
for col in logs.columns:
    print(col)

# print reward distribution
print(logs["reward"].describe())

# make scatter plot of steps vs reward
import matplotlib.pyplot as plt
plt.scatter(logs["reward_step"], logs["reward"], alpha=0.1)
plt.show()

# plot scatter plot of CE rewards
plt.scatter(logs["reward_step"], logs["normalized_rewards_CE"], alpha=0.1)
plt.show()


#%%
logs["n_notes"] = logs["midi"].apply(lambda x: x.note_num())

# plot number of notes over time
plt.scatter(logs["reward_step"], logs["n_notes"], alpha=0.02)
plt.show()
#%%

# now get response length in tokens
logs["completion_length"] = logs["completion"].apply(lambda x: len(x))

# plot response length over time
plt.scatter(logs["reward_step"], logs["completion_length"], alpha=0.01)
plt.show()

# plot number of notes over time
#%%
# plot clap_score_raw over tim

plt.scatter(logs["reward_step"], logs["normalized_rewards_clap_clf"], alpha=0.1)
plt.show()

# fit line
import numpy as np
from sklearn.linear_model import LinearRegression
X = logs["reward_step"].values.reshape(-1, 1)
y = logs["normalized_rewards_clap_clf"].values
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, alpha=0.1)
plt.plot(X, reg.predict(X), c="red")
plt.show()
#%%

# plot ce, cu, pc, pq mean across steps
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].mean(), c="red", label="CE")
plt.plot(logs.groupby("reward_step")["normalized_rewards_CU"].mean(), c="teal", label="CU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_programs_iou"].mean(), c="green", label="IOU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PC"].mean(), c="orange", label="PC")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PQ"].mean(), c="purple", label="PQ")
# plot average reward in blue
plt.plot(logs.groupby("reward_step")["reward"].mean(), c="blue", label="reward")
plt.legend()
plt.title("Mean reward across steps")
plt.show()


# fit line to reward vs step
import numpy as np
from sklearn.linear_model import LinearRegression
X = logs["reward_step"].values.reshape(-1, 1)
y = logs["normalized_rewards_CE"].values
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, alpha=0.1)
plt.plot(X, reg.predict(X), c="red")
plt.show()

#%%

# make a ridge plot of the rewards 
import seaborn as sns
import matplotlib.pyplot as plt
sns.kdeplot(data=logs, x="reward", hue="reward_step", fill=True, alpha=0.5)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Assuming logs is your DataFrame with reward and reward_step columns
# First, ensure reward_step is treated as numeric if it isn't already
logs['reward_step'] = pd.to_numeric(logs['reward_step'])

# Create a grid for the 3D plot
unique_steps = np.sort(logs['reward_step'].unique())
reward_range = np.linspace(logs['reward'].min(), logs['reward'].max(), 100)

# Create meshgrid for 3D plotting
X, Y = np.meshgrid(unique_steps, reward_range)

# Create empty array to hold histogram-based density values
density_matrix = np.zeros((len(reward_range), len(unique_steps)))

# Fill the density matrix using histograms instead of KDE
for i, step in enumerate(unique_steps):
    step_data = logs[logs['reward_step'] == step]['reward']
    if len(step_data) > 0:
        # Create histogram for this step
        hist, bin_edges = np.histogram(step_data, bins=50, 
                                      range=(logs['reward'].min(), logs['reward'].max()),
                                      density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate histogram values to our common grid
        density = np.interp(reward_range, bin_centers, hist, left=0, right=0)
        density_matrix[:, i] = density

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the contour
contour = ax.contour3D(X, Y, density_matrix, 20, cmap=cm.viridis)

# Add a color bar
fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

# Set labels
ax.set_xlabel('Reward Step (Time)')
ax.set_ylabel('Reward Value')
ax.set_zlabel('Density')
ax.set_title('3D Contour Plot of Reward Distribution Evolution')

# Adjust viewing angle for better visualization
ax.view_init(30, 45)

plt.tight_layout()
plt.show()



#%%

# lets make a heatmap of the reward distribution
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# create a heatmap of the reward distribution across time
# x axis is reward_step, y axis is reward, color is density
# first create a 2d histogram of the rewards2

hist, xedges, yedges = np.histogram2d(logs["reward_step"], logs["reward"], bins=(100,10), range=[[0, 100], [0, 1]])
# create a meshgrid
X, Y = np.meshgrid(xedges, yedges)
# plot the heatmap
plt.pcolormesh(X, Y, hist.T, cmap="viridis")
plt.colorbar()
plt.show()

# %%
