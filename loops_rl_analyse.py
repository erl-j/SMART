#%%
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import symusic
import muspy
# run_path = "artefacts/all_runs_2/piano-procedural/aes-0.04-1-100"
run_path = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-0.04-1-200/"
# run_path = "artefacts/all_runs_2/mil-dataset/pam-iou-0.04-1-100"
#%%
# load all logs
prelogs = pd.read_parquet(run_path + "/pre_eval/eval/rl_logs/0/logs.parquet")
print("Rows in pre eval: ", len(prelogs))

# print reward_weights
reward_weights = prelogs["reward_weights"].iloc[0]
print("Reward weights: ", reward_weights)
postlogs = pd.read_parquet(run_path + "/post_eval/eval/rl_logs/0/logs.parquet")
# add field to identify pre and post eval
prelogs["stage"] = "pre"
postlogs["stage"] = "post"
# concat pre and post logs
logs = pd.concat([prelogs, postlogs])
# for all columns that are dicts, expand them and prepend dict to the key as name
for col in logs.columns:
    if logs[col].apply(lambda x: isinstance(x, dict)).all():
        logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
        logs.drop(col, axis=1, inplace=True)

#%%

midi = []
for stage in ["pre", "post"]:
    midi_paths = glob.glob( run_path + f"/{stage}_eval/eval/midi/0/*.mid", recursive=True)
    # create records with 
    midi.extend([{"midi_path": m, "reward_step": int(m.split("/")[-2].split("_")[0]), "idx" : int(m.split("_")[-1].replace(".mid","")), "stage": stage, "symusic" : symusic.Score(m), "muspy": muspy.read_midi(m) } for m in tqdm(midi_paths)])

# join midi and logs on reward_step and idx
midi_df = pd.DataFrame(midi)
# join logs and midi on reward_step, stage and idx
logs = logs.merge(midi_df, on=["reward_step", "idx", "stage"], how="inner")

#%%


def get_pitches(sm):
    pitches = []
    for track in sm.tracks:
        for note in track.notes:
            pitches.append(note.pitch)
    return pitches

def get_interonset_ticks(sm):
    all_interonset_ticks = []
    for track in sm.tracks:
        if len(track.notes) <= 1:
            continue
        # Sort notes by start time to ensure correct interonset calculation
        sorted_notes = sorted(track.notes, key=lambda x: x.start)
        for note_idx in range(len(sorted_notes) - 1):
            # Get the current note
            note = sorted_notes[note_idx]
            # Get the next note
            next_note = sorted_notes[note_idx + 1]
            # Calculate the interonset time in ticks
            interonset_time = next_note.start - note.start
            # Append to the list
            all_interonset_ticks.append(interonset_time)
    return all_interonset_ticks

def get_note_durations(sm):
    note_durations = []
    for track in sm.tracks:
        for note in track.notes:
            note_durations.append(note.end - note.start)
    return note_durations

def get_intervals(sm):
    all_intervals = []
    for track in sm.tracks:
        if len(track.notes) <= 1:
            continue
        # Sort notes by start time to ensure meaningful intervals
        sorted_notes = sorted(track.notes, key=lambda x: x.start)
        for note_idx in range(len(sorted_notes) - 1):
            # Get the current note
            note = sorted_notes[note_idx]
            # Get the next note
            next_note = sorted_notes[note_idx + 1]
            # Calculate the interval in semitones
            interval = next_note.pitch - note.pitch
            # Append to the list
            all_intervals.append(interval)
    return all_intervals

def get_velocities(sm):
    velocities = []
    for track in sm.tracks:
        for note in track.notes:
            velocities.append(note.velocity)
    return velocities

def get_dynamic_range(sm):
    velocities = get_velocities(sm)
    if not velocities:  # Check if the list is empty
        return 0
    return max(velocities) - min(velocities)

def get_number_of_notes(sm):
    total_notes = 0
    for track in sm.tracks:
        total_notes += len(track.notes)
    return total_notes

def get_number_of_tempo_changes(sm):
    return len(sm.tempos)

def get_number_of_time_signature_changes(sm):
    return len(sm.time_signatures)

def get_duration_in_seconds(sm):
    sm = sm.to(symusic.TimeUnit("second"))
    return sm.end()


import numpy as np
# Helper function to try a calculation and return NaN if it fails
def try_or_nan(func):
    try:
        return func()
    except Exception:
        return np.nan

# compute metrics with error handling
logs["metric_num_notes"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: x.note_num()))
logs["metric_pitch_class_entropy"] = logs["muspy"].apply(lambda x: try_or_nan(lambda: muspy.pitch_class_entropy(x)))
logs["metric_polyphony"] = logs["muspy"].apply(lambda x: try_or_nan(lambda: muspy.polyphony(x)))
logs["metric_polyphony_rate"] = logs["muspy"].apply(lambda x: try_or_nan(lambda: muspy.polyphony_rate(x)))
logs["metric_scale_consistency"] = logs["muspy"].apply(lambda x: try_or_nan(lambda: muspy.scale_consistency(x)))
logs["metric_empty_beat_rate"] = logs["muspy"].apply(lambda x: try_or_nan(lambda: muspy.empty_beat_rate(x)))

logs["fts_pitches"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_pitches(x)))
logs["fts_interonset_ticks"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_interonset_ticks(x)))
logs["fts_note_durations"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_note_durations(x)))
logs["fts_intervals"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_intervals(x)))
logs["fts_velocities"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_velocities(x)))

logs["ft_dynamic_range"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_dynamic_range(x)))
logs["ft_number_of_notes"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_number_of_notes(x)))
logs["ft_number_of_tempo_changes"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_number_of_tempo_changes(x)))
logs["ft_number_of_time_signature_changes"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_number_of_time_signature_changes(x)))
logs["ft_duration_in_seconds"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_duration_in_seconds(x)))


#%%
# get distribution of each metric, find metrics by picking columns that start with "metric_"
metrics = [ col for col in logs.columns if col.startswith("metric_")]

for metric in metrics:
    plt.figure()
    for system in ["pre", "post"]:
        plt.hist(logs[logs["stage"] == system][metric], bins=50, alpha=0.5, label=system)
    plt.title(metric)
    plt.legend()
    plt.show()

#%%

fts = [ col for col in logs.columns if col.startswith("fts_")]
# these first have to be aggregated by joining all lists 
for fts_col in fts:
    # now we can plot the distribution
    plt.figure()
    for system in ["pre", "post"]:
        # aggregate the lists into a single list
        all_values = []
        for values in logs[logs["stage"] == system][fts_col]:
            all_values.extend(values)
        plt.hist(all_values, bins=50, alpha=0.5, label=system)
    plt.title(fts_col)
    plt.legend()
    plt.show()


#%%

ft = [col for col in logs.columns if col.startswith("ft_")]
# now plot the distribution of of the ft
# for each column that starts with "ft_" in a single plot. these do not need to be aggregated
for ft_col in ft:
    plt.figure()
    for system in ["pre", "post"]:
        plt.hist(logs[logs["stage"] == system][ft_col], bins=50, alpha=0.5, label=system)
    plt.title(ft_col)
    plt.legend()
    plt.show()


#%%
# for each "normalized_rewards" column, plot the distribution for pre and post eval in a single histogram
normalized_rewards = [col for col in logs.columns if "normalized_rewards" in col]
for rew in normalized_rewards:
    plt.figure()
    for system in ["pre", "post"]:
        plt.hist(logs[logs["stage"] == system][rew], bins=50, alpha=0.5, label=system, range=(0, 1))
    plt.title(rew)
    plt.legend()
    plt.show()



#%%

import muspy


plt.figure()
for system in ["pre", "post"]:
    plt.hist(logs[logs["stage"] == system]["pitch_class_entropy"], bins=50, alpha=0.5, label=system)
plt.title("pitch_class_entropy")
plt.legend()
plt.show()
#%%


#%%
# plot average rewards for pre and post eval


#%%

logs = glob.glob(run_path + "/rl_logs/**/*.parquet", recursive=True)

logs = [pd.read_parquet(log) for log in logs]
logs = pd.concat(logs)






# if normalized_rewards_programs_iou is missing, add it as 0
if "normalized_rewards_programs_iou" not in logs.columns:
    logs["normalized_rewards_programs_iou"] = 0

# join logs and midi on reward_step and idx

midi = pd.DataFrame(midi)
logs = logs.merge(midi, on=["reward_step", "idx"], how="inner")

# plot all normalized rewards over time
# get field that stra
normalized_rewards = [col for col in logs.columns if "normalized_rewards" in col]

for rew in normalized_rewards:
    plt.plot(logs.groupby("reward_step")[rew].mean(), label=rew)

# also include average reward
plt.plot(logs.groupby("reward_step")["reward"].mean(), label="reward")
plt.legend()
plt.show()

n_rewards = len(normalized_rewards)

# show normalized_rewards_pam_avg in a scatter plot
plt.scatter(logs["reward_step"], logs["normalized_rewards_pam_avg"], alpha=0.1, s=2)
plt.legend()
plt.show()
# fit a line to the data
import numpy as np
from sklearn.linear_model import LinearRegression
X = logs["reward_step"].values.reshape(-1, 1)
y = logs["normalized_rewards_pam_avg"].values
reg = LinearRegression().fit(X, y)
plt.scatter(X, y, alpha=0.1)
plt.plot(X, reg.predict(X), c="red")
plt.show()




# for 

# # for each reward, plot the distribution of the reward across steps using subplots 
# for step in logs["reward_step"].unique():
#     plt.figure()
#     for i, rew in enumerate(normalized_rewards):
#         plt.subplot(n_rewards, 1, i+1)
#         plt.hist(logs[logs["reward_step"] == step][rew], bins=10, range=(0,1))
#         plt.title(rew)
#     plt.show()








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


print(logs["normalized_rewards_programs_iou"].describe())
# plot ce, cu, pc, pq mean across steps
plt.plot(logs.groupby("reward_step")["normalized_rewards_CE"].mean(), c="red", label="CE")
plt.plot(logs.groupby("reward_step")["normalized_rewards_CU"].mean(), c="teal", label="CU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_programs_iou"].mean(), c="green", label="IOU")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PC"].mean(), c="orange", label="PC")
plt.plot(logs.groupby("reward_step")["normalized_rewards_PQ"].mean(), c="purple", label="PQ")
plt.plot(logs.groupby("reward_step")["normalized_rewards_clap_clf"].mean(), c="black", label="clap")
# plot average reward in blue
# plt.plot(logs.groupby("reward_step")["reward"].mean(), c="blue", label="reward")
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

# take logs of first step only
logs_first = logs#[logs["reward_step"] == 0]
for rew in ["CE", "CU", "PC", "PQ"]:
    # scatter plot of clap score clf vs ce
    plt.scatter(logs_first["normalized_rewards_clap_clf"], logs_first[f"normalized_rewards_{rew}"], alpha=0.5, s=1)
    # set x and y limits to 0,1
    plt.xlim(0,1)
    plt.ylim(0,1)
    # label axes
    plt.xlabel("clap clf")
    plt.ylabel(rew)
    plt.show()


# now same for clap score clf vs iou
# %%
