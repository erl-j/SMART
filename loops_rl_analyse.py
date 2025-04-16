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
run_path = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s"
# run_path = "artefacts/all_runs_4/piano-long-dataset/aes-avg-0.04-1.0-100-10s/"
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
# print prompt_and_completion_tokens for pre
pre_prompt_and_completion_tokens = prelogs["prompt_and_completion_tokens"].iloc[0]
print("Prompt and completion tokens: ", pre_prompt_and_completion_tokens)
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
        sorted_notes = sorted(track.notes, key=lambda x: (x.start,x.pitch,x.end, x.velocity))
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

def get_pitch_range(sm):
    pitches = get_pitches(sm)
    if not pitches:  # Check if the list is empty
        return 0
    return max(pitches) - min(pitches)

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
logs["ft_pitch_range"] = logs["symusic"].apply(lambda x: try_or_nan(lambda: get_pitch_range(x)))
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
zero_note_logs = logs[logs["metric_num_notes"] == 0]
nan_note_logs = logs[logs["metric_num_notes"].isna()]
# get idx of zero_note_logs
zero_note_logs_idx = zero_note_logs["idx"].tolist()
print("Indices of logs with 0 notes: ", zero_note_logs_idx)
print("Number of logs with 0 notes: ", len(zero_note_logs))
print("Number of logs with nan notes: ", len(nan_note_logs))
# only keep logs where number of notes is not 0 or nan
filtered_logs = logs[logs["metric_num_notes"].notna() & (logs["metric_num_notes"] != 0)]
print(len(filtered_logs))
print(len(logs))
# count how many notes are in the pre and post eval
pre_count = len(filtered_logs[filtered_logs["stage"] == "pre"])
post_count = len(filtered_logs[filtered_logs["stage"] == "post"])
print("Pre eval count: ", pre_count)
print("Post eval count: ", post_count)

#%%
# count how many have less than 10 notes
less_than_10_notes = filtered_logs[filtered_logs["metric_num_notes"] < 10]

print("Number of logs with less than 10 notes: ", len(less_than_10_notes))
#%%
# show and prompt_and_completion_tokens that yielded 0 or nan notes
#%%
#%%
# show and prompt_and_completion_tokens that yielded 0 or nan notes
#%%
# Define custom names for the pre and post systems
SYSTEM_NAMES = {
    "pre": "base",  # Change this to your preferred name for "pre"
    "post": "SMART"  # Change this to your preferred name for "post"
}

selected_metrics = {
    "metric_num_notes": {
        "title": "Number of Notes",
        "bins": 40,
        "range": (0, 100),
        "description": "Distribution of note count",
        "xaxis_label": "Note Count",
        "yaxis_label": "Number of Tracks"
    },
    "metric_polyphony_rate": {
        "title": "Polyphony Rate",
        "bins": 30,
        "range": (0, 1.0),
        "description": "Proportion of polyphonic notes",
        "xaxis_label": "Polyphony Rate",
        "yaxis_label": "Number of Tracks"
    },
    "metric_empty_beat_rate": {
        "title": "Empty Beat Rate",
        "bins": 25,
        "range": (0, 1.0),
        "description": "Proportion of beats with no notes",
        "xaxis_label": "Empty Beat Rate",
        "yaxis_label": "Number of Tracks"
    },

    "metric_scale_consistency": {
        "title": "Scale Consistency",
        "bins": 30,
        "range": (0.65, 1.0),
        "description": "Adherence to scale patterns",
        "xaxis_label": "Scale Consistency Score",
        "yaxis_label": "Number of Tracks"
    },
    # "fts_intervals": {
    #     "title": "Interval Distribution",
    #     "bins": 120,
    #     "range": (-60, 60),
    #     "description": "Distribution of melodic intervals",
    #     "xaxis_label": "Interval Size (semitones)",
    #     "yaxis_label": "Number of Notes"
    # },
    "fts_pitches": {
        "title": "Pitch Distribution",
        "bins": 80,  # Fixed: was 100-20
        "range": (20, 100),
        "description": "Distribution of MIDI pitch values",
        "xaxis_label": "MIDI Pitch",
        "yaxis_label": "Number of Notes"
    },
    "ft_pitch_range": {
        "title": "Pitch Range",
        "bins": 30,
        "range": (0, 70),
        "description": "Range between lowest and highest pitch",
        "xaxis_label": "Pitch Range (semitones)",
        "yaxis_label": "Number of Tracks"
    },

    "fts_velocities": {
        "title": "Velocity Distribution",
        "bins": 20,
        "range": (0, 127),
        "description": "Distribution of MIDI velocity values",
        "xaxis_label": "MIDI Velocity",
        "yaxis_label": "Number of Notes"
    },
    "ft_dynamic_range": {
        "title": "Velocity Range",
        "bins": 20,
        "range": (0, 127),
        "description": "Range of dynamics (loudness)",
        "xaxis_label": "Velocity Range",
        "yaxis_label": "Number of Tracks"
    },
}

# Simple, clean styling setup - using Set1 palette as requested
import matplotlib.pyplot as plt
import seaborn as sns

# Set a dark style with Set1 color palette and improved tick visibility
sns.set_theme(style="ticks", palette="Set1", font_scale=1.2)
plt.rcParams['axes.grid'] = False  # Turn off grid
plt.rcParams['font.size'] = 11
plt.rcParams['xtick.color'] = 'black'  # Make ticks more visible
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.major.width'] = 1.2  # Thicker ticks
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5.0   # Longer ticks
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['axes.linewidth'] = 1.2     # Thicker axes
# set background color to dark default
plt.rcParams['axes.facecolor'] = '#EAEAF2'  # Set axes background to white



# Create the figure
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Flatten the axes array for easier iteration
axs = axs.flatten()

for i, (metric_key, metric_info) in enumerate(selected_metrics.items()):
    ax = axs[i]
    # Determine whether to use density based on the metric prefix
    use_density = False  # Default to count (not density)
    
    for j, system in enumerate(["pre", "post"]):
        # Handle metrics that are stored as lists
        if isinstance(filtered_logs[filtered_logs["stage"] == system][metric_key].iloc[0], list):
            all_values = []
            for values in filtered_logs[filtered_logs["stage"] == system][metric_key]:
                all_values.extend(values)
            
            # Filter values to be within the specified range
            filtered_values = [v for v in all_values if metric_info["range"][0] <= v <= metric_info["range"][1]]
            
            # Plot histogram with appropriate settings
            ax.hist(filtered_values, 
                   bins=metric_info["bins"], 
                   range=metric_info["range"],
                   alpha=0.6, 
                   density=use_density,
                   edgecolor='black', 
                   linewidth=0.0,
                   label=f"{SYSTEM_NAMES[system]}")
        else:
            # Same for non-list metrics
            data = filtered_logs[filtered_logs["stage"] == system][metric_key]
            filtered_data = data[(data >= metric_info["range"][0]) & (data <= metric_info["range"][1])]
            
            ax.hist(filtered_data, 
                   bins=metric_info["bins"], 
                   range=metric_info["range"],
                   alpha=0.6, 
                   density=use_density,
                   edgecolor='black', 
                   linewidth=0.0,
                   label=f"{SYSTEM_NAMES[system]}")
    
    # Add titles and labels
    ax.set_title(metric_info["title"], fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_info["xaxis_label"], fontsize=12)
    ax.set_ylabel(metric_info["yaxis_label"], fontsize=12)
    
    # Add a legend with white background for better visibility
    ax.legend(loc="best", frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # Clean up the plot - remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)  # Make remaining spines more visible
    ax.spines['left'].set_linewidth(1.2)
    
    # Improve tick visibility and spacing
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.0, length=3)
    
    # Format y-axis with integers for counts and make ticks more visible
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    
    # Make sure to apply color to ticks for visibility in dark theme
    ax.tick_params(colors='black')

# Apply tight layout
plt.tight_layout()

# Save the figure
plt.savefig('plots/music_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plots/music_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Code for the second set of plots (aesthetic scores)
# Set the style to match the main plots with enhanced tick visibility


# Create a 4 split histograms for aes_scores_CE, aes_scores_CU, aes_scores_PC, aes_scores_PQ
plt.figure(figsize=(16, 4))
audiobox_full_names = {
    "CE": "Predicted Content Enjoyment",
    "CU": "Predicted Content Usefulness",
    "PC": "Predicted Production Complexity",
    "PQ": "Predicted Production Quality"
}

for i, rew in enumerate(["aes_scores_CE", "aes_scores_CU", "aes_scores_PC", "aes_scores_PQ"]):
    # plot in one row, no grid
    plt.subplot(1, 4, i+1)
    for j, system in enumerate(["pre", "post"]):
        plt.hist(logs[logs["stage"] == system][rew], 
                bins=32, 
                alpha=0.6, 
                label=SYSTEM_NAMES[system], 
                range=(1, 10),
                edgecolor='black',
                linewidth=0.0)
        
    # plt.gca().set_title(audiobox_full_names[rew.split("_")[-1]], fontsize=14, fontweight='bold')

    
    plt.xlabel(audiobox_full_names[rew.split("_")[-1]], fontsize=12)
    plt.ylabel("Number of Tracks", fontsize=12)
    plt.ylim(0, 1000)
    
    # Add legend with white background for better visibility
    plt.legend(loc="best", frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # Remove top and right spines, make bottom and left spines more visible
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.2)
    plt.gca().spines['left'].set_linewidth(1.2)
    
    # Improve tick visibility and spacing
    plt.gca().tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5, colors='black')
    
    # Reduce the number of ticks on the y axis but ensure they're visible
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

plt.tight_layout()
plt.savefig('plots/aes_scores_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plots/aes_scores_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
# %%

import symusic
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define MIDI directories
midi_dirs = {
    "base model": "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s/pre_eval/eval/midi/0",
    "w/ SMART 100 steps, beta=0.04": "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s/post_eval/eval/midi/0",
    "w/ SMART 1000 steps, beta=0.04": "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-1000-10s/post_eval/eval/midi/0",
    "w/ SMART 1000 steps, beta=0.00": "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.0-1.0-1000-10s/post_eval/eval/midi/0",
}

# Function to get piano roll
def get_pr(sm):
    sm = sm.copy().resample(12, 0)
    pr = sm.pianoroll(modes=["frame"],
                    pitch_range=(0, 128),
                    encode_velocity=False).sum(0).sum(0)
    return pr

# Store results
results = {}

# Compute data for all configurations
for key, midi_dir in midi_dirs.items():
    print(f"Processing {key}...")
    
    # Load logs
    logs_path = midi_dir.replace("midi", "rl_logs") + "/logs.parquet"
    logs = pd.read_parquet(logs_path)
    
    # Expand dictionary columns
    for col in logs.columns:
        if logs[col].apply(lambda x: isinstance(x, dict)).all():
            logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
            logs.drop(col, axis=1, inplace=True)
    
    # Find all MIDI files
    midi_files = glob.glob(midi_dir + "/*.mid")

    # sort midi_files
    midi_files = sorted(midi_files)

    skip_str = ["641.mid"]
    
    # Calculate piano rolls
    piano_rolls = []
    for i in tqdm(range(len(midi_files))):
        if any(s in midi_files[i] for s in skip_str):
            continue
        # print(f"Processing {midi_files[i]}...")
        midi = symusic.Score(midi_files[i])
        pr = get_pr(midi)
        if pr.shape[1] < 12*16:
            # only take first 32 frames
            continue
        piano_rolls.append(pr[:, :12*16])
    
    # Convert to numpy array and calculate average
    piano_rolls = np.array(piano_rolls)
    avg_piano_roll = np.mean(piano_rolls, axis=0)
    
    # Store results
    results[key] = {
        'logs': logs,
        'avg_piano_roll': avg_piano_roll
    }

# Now that we have all the data, we can plot in the next cell
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


font = "Arial"
# Enhanced plot settings for better tick visibility

# Create 4x2 subplot grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# First pass to find global min/max values for piano rolls
# and maximum count for histograms
piano_roll_min = float('inf')
piano_roll_max = float('-inf')
hist_max_count = 0

# Define model names for the columns
model_names = {
    "base model": "Base Model",
    "w/ SMART 100 steps, beta=0.04": "SMART, 100 steps\nβ=0.04",
    "w/ SMART 1000 steps, beta=0.04": "SMART, 1000 steps\nβ=0.04",
    "w/ SMART 1000 steps, beta=0.00": "SMART, 1000 steps\nβ=0.00"
}

# Define a nice color palette
colors = plt.cm.get_cmap('Set1').colors

for key, data in results.items():
    # Find min/max for piano rolls
    curr_min = data['avg_piano_roll'].min()
    curr_max = data['avg_piano_roll'].max()
    
    if curr_min < piano_roll_min:
        piano_roll_min = curr_min
    if curr_max > piano_roll_max:
        piano_roll_max = curr_max
    
    # Calculate histogram to find max count
    hist_counts, _ = np.histogram(data['logs']["aes_scores_CE"], bins=40, range=(1, 10))
    if hist_counts.max() > hist_max_count:
        hist_max_count = hist_counts.max()

# Plot histograms on top row and piano rolls on bottom row
for i, (key, data) in enumerate(results.items()):
    # Top row: CE histogram with specific subplot title
    sns.histplot(data['logs']["aes_scores_CE"], bins=32, binrange=(1, 10), 
                ax=axes[0, i], color=colors[i % len(colors)], 
                edgecolor='black', linewidth=0.0, alpha=0.7)
    
    # axes[0, i].set_title("Predicted Content Enjoyment", fontsize=14, fontfamily=font)
    axes[0, i].set_xlabel("Predicted Content Enjoyment", fontsize=12, fontfamily=font)
    axes[0, i].set_ylabel("Count", fontsize=12, fontfamily=font)
    
    # Set consistent y-axis limit across all histograms
    axes[0, i].set_ylim(0, hist_max_count * 1.05)  # Add 5% padding
    
    # Clean up the plot - remove grid and unnecessary spines
    axes[0, i].grid(False)
    axes[0, i].spines['top'].set_visible(False)
    axes[0, i].spines['right'].set_visible(False)
    axes[0, i].spines['bottom'].set_linewidth(1.2)
    axes[0, i].spines['left'].set_linewidth(1.2)
    
    # Enhance tick visibility
    axes[0, i].tick_params(axis='both', which='major', labelsize=10, width=1.2, length=5, colors='black')
    
    # Format y-axis with integers for counts and make ticks sparser
    axes[0, i].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    
    # Bottom row: Average piano roll
    # Note: using 1-avg_piano_roll for better visibility with gray colormap
    piano_roll = data['avg_piano_roll']
    num_ticks = piano_roll.shape[1]
    
    # Assuming standard MIDI resolution of 480 ticks per beat
    ticks_per_beat = 12
    
    im = axes[1, i].imshow(1-piano_roll, aspect='auto', cmap='gray', 
                          interpolation='none', origin='lower',
                          vmin=1-piano_roll_max, vmax=1-piano_roll_min,  # Adjust vmin/vmax for inverted values
                          extent=[0, num_ticks/ticks_per_beat, 0, 127])  # Convert x-axis to beats
    
    axes[1, i].set_title("Average Piano Roll", fontsize=14, fontfamily=font)
    axes[1, i].set_xlabel("Beats", fontsize=12, fontfamily=font)
    axes[1, i].set_ylabel("Pitch", fontsize=12, fontfamily=font)
    axes[1, i].set_ylim(21, 109)  # Restrict y-axis to 21-109 pitch range
    
    # Add frame to piano roll by making all spines visible
    for spine in axes[1, i].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)  # Increased from 1.0 for better visibility
        spine.set_color('black')  # Using black for better visibility
    
    # Add x and y ticks for the piano roll
    # For x-axis (beats): add major ticks every 4 beats, minor ticks every beat
    max_beats = num_ticks / ticks_per_beat
    axes[1, i].xaxis.set_major_locator(plt.MultipleLocator(4))
    axes[1, i].xaxis.set_minor_locator(plt.MultipleLocator(1))
    axes[1, i].tick_params(axis='x', which='major', length=6, width=1.5, colors='black')  # Thicker and black
    axes[1, i].tick_params(axis='x', which='minor', length=3, width=0.8, colors='white')  # Thicker and black
    
    # For y-axis (pitch): add major ticks at octaves (C notes: 24, 36, 48, 60, 72, 84, 96, 108)
    # and minor ticks at all notes
    octave_ticks = [24, 36, 48, 60, 72, 84, 96, 108]  # C octaves in MIDI
    axes[1, i].set_yticks(octave_ticks)
    axes[1, i].set_yticklabels([f"C{o-1}" for o in range(1, 9)])  # C0 to C8 notation
    
    # Add minor ticks for all notes
    all_notes = np.arange(21, 110)
    axes[1, i].set_yticks(all_notes, minor=True)
    axes[1, i].tick_params(axis='y', which='major', length=6, width=1.5, colors='black')  # Thicker and black
    axes[1, i].tick_params(axis='y', which='minor', length=3, width=0.8, colors='white')  # Thicker and black
    
    # Add light grid lines at octave positions
    axes[1, i].grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

# Apply tight layout
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at the top for column titles

# After tight_layout, add column titles directly above each column
for i, key in enumerate(results.keys()):
    # Get the position of the top axes in this column
    pos = axes[0, i].get_position()
    # Place title at the center of the column
    fig.text(pos.x0 + pos.width/2, 0.96, model_names.get(key, key), 
             ha='center', va='top', fontsize=13, fontweight='bold', fontfamily=font, color='black')  # Using black color

# Adjust figure to make room for column titles
plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4)  # More space between subplots

# Save high-resolution versions
plt.savefig('plots/midi_visualization_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plots/midi_visualization_comparison.png', dpi=300, bbox_inches='tight')

plt.show()
# %%
