#%%

import symusic
from tqdm import tqdm
# data format
checkpoint_name = "./artefacts/pianophi-kl=0.1"

# load the tokenizer
# %%
import glob
import pandas as pd

midi_paths = glob.glob(f"{checkpoint_name}/fs_renders/**/*.mid", recursive=True)

records = [
    {
    "path": midi_path,
    "reward": float(midi_path.split("/")[-1].split("_")[0].split("=")[1]),
    "step": int(midi_path.split("/")[-2]),
    "score" : symusic.Score(midi_path)
} for midi_path in tqdm(midi_paths)
]

df = pd.DataFrame(records)

def get_pitches(score):
    pitches = []
    for track in score.tracks:
        for note in track.notes:
            pitches.append(note.pitch)
    return pitches

def get_intervals(pitches):
    intervals = []
    for i in range(len(pitches)-1):
        intervals.append(pitches[i+1] - pitches[i])
    return intervals
            

df["pitches"] = df["score"].apply(get_pitches)
df["intervals"] = df["pitches"].apply(get_intervals)

# show scatterplot of reward across steps
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df, x="step", y="reward", alpha=0.25)
plt.show()

#%%

# show scatterplot of pitches for each step
def flatten_list(l):
    return [item for sublist in l for item in sublist]

step_pitches = df.groupby("step")["pitches"].apply(list)
step_pitches = step_pitches.reset_index()
step_pitches["pitches"] = step_pitches["pitches"].apply(flatten_list)

# show scatterplot of intervals across steps
step_intervals = df.groupby("step")["intervals"].apply(list)
step_intervals = step_intervals.reset_index()
step_intervals["intervals"] = step_intervals["intervals"].apply(flatten_list)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# For pitches heatmap
plt.figure(figsize=(14, 8))

# Create a matrix with pitches on y-axis and steps on x-axis
steps = sorted(df['step'].unique())
pitch_matrix = np.zeros((128, len(steps)))  # 128 MIDI pitches

for i, step in enumerate(steps):
    step_data = df[df['step'] == step]
    all_pitches = [pitch for sublist in step_data['pitches'] for pitch in sublist]
    
    # Count occurrences of each pitch at this step
    for pitch in all_pitches:
        if 0 <= pitch < 128:  # Ensure valid MIDI pitch range
            pitch_matrix[pitch, i] += 1

# Flip the matrix so low pitches are at the bottom
# We need to reverse the matrix along the y-axis
pitch_matrix = np.flipud(pitch_matrix)

plt.figure(figsize=(14, 8))
ax = sns.heatmap(pitch_matrix, cmap='viridis')
plt.ylabel('Pitch Value')
plt.xlabel('Time Step')
plt.title('Pitch Distribution Over Time')

# Generate pitch labels (from high to low since we flipped the matrix)
pitch_labels = np.arange(127, -1, -12)  # Every 12 steps (octaves), from high to low
pitch_positions = np.arange(0, 128, 12)  # Position on the y-axis
plt.yticks(pitch_positions, pitch_labels)

# Set x-ticks to match the step values
plt.xticks(np.arange(len(steps)), steps)
plt.tight_layout()
plt.show()

# For intervals heatmap
plt.figure(figsize=(14, 8))

# Determine the interval range from your data
all_intervals = [interval for sublist in df['intervals'] for interval in sublist]
min_interval = min(all_intervals) if all_intervals else -12
max_interval = max(all_intervals) if all_intervals else 12

# Create a range of intervals and sort them numerically
interval_values = sorted(list(set(all_intervals)))
interval_dict = {interval: idx for idx, interval in enumerate(interval_values)}
interval_matrix = np.zeros((len(interval_values), len(steps)))

for i, step in enumerate(steps):
    step_data = df[df['step'] == step]
    all_intervals = [interval for sublist in step_data['intervals'] for interval in sublist]
    
    # Count occurrences of each interval at this step
    for interval in all_intervals:
        if interval in interval_dict:  # Ensure interval exists in our dictionary
            interval_matrix[interval_dict[interval], i] += 1

# Flip the matrix to have negative intervals at the bottom and positive at the top
plt.figure(figsize=(14, 8))
ax = sns.heatmap(interval_matrix, cmap='magma')
plt.ylabel('Interval Value')
plt.xlabel('Time Step')
plt.title('Interval Distribution Over Time')
# Set y-ticks to match the actual interval values
plt.yticks(np.arange(len(interval_values)), interval_values)
# Set x-ticks to match the step values
plt.xticks(np.arange(len(steps)), steps)
plt.tight_layout()
plt.show()


#%%


# get durations over time
def get_durations(score):
    durations = []
    for track in score.tracks:
        for note in track.notes:
            durations.append(note.duration)
    return durations

df["durations"] = df["score"].apply(get_durations)
print(df.head())
step_durations = df.groupby("step")["durations"].apply(list)
step_durations = step_durations.reset_index()
step_durations["durations"] = step_durations["durations"].apply(flatten_list)


for step in step_durations["step"]:
    durations = step_durations[step_durations["step"] == step]["durations"].values[0]
    sns.scatterplot(x=step, y=durations, alpha=0.01, color="blue")
plt.title("Note durations over time")
plt.show()

def get_interonset_times(score):
    interonset_times = []
    for track in score.tracks:
        for i in range(len(track.notes)-1):
            interonset_times.append(track.notes[i+1].start - track.notes[i].start)
    return interonset_times

df["interonset_times"] = df["score"].apply(get_interonset_times)
step_interonset_times = df.groupby("step")["interonset_times"].apply(list)
step_interonset_times = step_interonset_times.reset_index()
step_interonset_times["interonset_times"] = step_interonset_times["interonset_times"].apply(flatten_list)

for step in step_interonset_times["step"]:
    interonset_times = step_interonset_times[step_interonset_times["step"] == step]["interonset_times"].values[0]
    sns.scatterplot(x=step, y=interonset_times, alpha=0.1, color="blue")
plt.title("Interonset times over time")
plt.show()

#%%
# get number of notes
def get_num_notes(score):
    num_notes = 0
    for track in score.tracks:
        num_notes += len(track.notes)
    return num_notes

df["num_notes"] = df["score"].apply(get_num_notes)
step_num_notes = df.groupby("step")["num_notes"].apply(list)
step_num_notes = step_num_notes.reset_index()

print(step_num_notes)
for step in step_num_notes["step"]:
    num_notes = step_num_notes[step_num_notes["step"] == step]["num_notes"].values[0]
    sns.scatterplot(x=step, y=num_notes, alpha=0.1, color="blue")
plt.title("Number of notes over time")
plt.show()


#%%

# get score duration 
def get_score_duration(score):
    score = score.copy()
    # switch to seconds
    score = score.to(symusic.TimeUnit.second)
    return score.end()
df["score_duration"] = df["score"].apply(get_score_duration)
step_score_duration = df.groupby("step")["score_duration"].apply(list)
step_score_duration = step_score_duration.reset_index()

for step in step_score_duration["step"]:
    score_duration = step_score_duration[step_score_duration["step"] == step]["score_duration"].values[0]
    sns.scatterplot(x=step, y=score_duration, alpha=0.1, color="blue")
plt.title("Score duration over time")
plt.show()


#%%
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Make sure we're using a non-interactive backend for animation
import matplotlib
matplotlib.use('Agg')

# Create animated histogram for pitches
fig_pitch, ax_pitch = plt.subplots(figsize=(10, 6))

steps = sorted(df['step'].unique())
max_count = 0

# Find the maximum count for scaling
for step in steps:
    step_data = df[df['step'] == step]
    all_pitches = [pitch for sublist in step_data['pitches'].tolist() for pitch in sublist]
    if all_pitches:
        counts, _ = np.histogram(all_pitches, bins=np.arange(0, 129))
        max_count = max(max_count, np.max(counts))

pitch_bars = None  # Reference to the bar container

def animate_pitch(frame):
    global pitch_bars
    ax_pitch.clear()
    
    step = steps[frame]
    step_data = df[df['step'] == step]
    all_pitches = [pitch for sublist in step_data['pitches'].tolist() for pitch in sublist]
    
    if all_pitches:
        # Create histogram data
        counts, edges = np.histogram(all_pitches, bins=np.arange(0, 129))
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Plot the bars
        pitch_bars = ax_pitch.bar(centers, counts, width=1, alpha=0.7, color='skyblue')
    
    ax_pitch.set_xlim(0, 127)
    ax_pitch.set_ylim(0, max_count * 1.1)
    ax_pitch.set_xlabel('Pitch Value')
    ax_pitch.set_ylabel('Frequency')
    ax_pitch.set_title(f'Pitch Distribution at Step {step}')
    ax_pitch.grid(alpha=0.3)
    
    return pitch_bars,

# Create the animation with explicit frames
pitch_anim = FuncAnimation(
    fig_pitch, 
    animate_pitch, 
    frames=range(len(steps)), 
    interval=500, 
    blit=False,  # Set to False which is more reliable
    repeat=True
)

# Display in notebook - this is the key part
plt.close(fig_pitch)  # Prevent double display
HTML(pitch_anim.to_jshtml())  # This should display the animation

#%%
# Similarly for intervals
fig_interval, ax_interval = plt.subplots(figsize=(10, 6))

all_intervals = [interval for sublist in df['intervals'].tolist() for interval in sublist]
min_interval = min(all_intervals) if all_intervals else -12
max_interval = max(all_intervals) if all_intervals else 12
max_count_interval = 0

# Find maximum count for intervals
for step in steps:
    step_data = df[df['step'] == step]
    step_intervals = [interval for sublist in step_data['intervals'].tolist() for interval in sublist]
    if step_intervals:
        counts, _ = np.histogram(step_intervals, bins=np.arange(min_interval, max_interval + 2))
        max_count_interval = max(max_count_interval, np.max(counts))

interval_bars = None  # Reference to the bar container

def animate_interval(frame):
    global interval_bars
    ax_interval.clear()
    
    step = steps[frame]
    step_data = df[df['step'] == step]
    step_intervals = [interval for sublist in step_data['intervals'].tolist() for interval in sublist]
    
    if step_intervals:
        # Create histogram data
        counts, edges = np.histogram(step_intervals, bins=np.arange(min_interval, max_interval + 2))
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Plot the bars
        interval_bars = ax_interval.bar(centers, counts, width=0.8, alpha=0.7, color='lightcoral')
    
    ax_interval.set_xlim(min_interval - 0.5, max_interval + 0.5)
    ax_interval.set_ylim(0, max_count_interval * 1.1)
    ax_interval.set_xlabel('Interval Value')
    ax_interval.set_ylabel('Frequency')
    ax_interval.set_title(f'Interval Distribution at Step {step}')
    ax_interval.grid(alpha=0.3)
    
    return interval_bars,

# Create the animation with explicit frames
interval_anim = FuncAnimation(
    fig_interval, 
    animate_interval, 
    frames=range(len(steps)), 
    interval=500, 
    blit=False,  # Set to False which is more reliable
    repeat=True
)

# Display in notebook
plt.close(fig_interval)  # Prevent double display
HTML(interval_anim.to_jshtml())
# %%



# %%
