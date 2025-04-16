#%%
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

# Set up the style for a clean, minimalist appearance
plt.style.use('seaborn-v0_8')  # Base style without the grid
sns.set_context("paper", font_scale=1.2)

# Turn off the grid
plt.rcParams['axes.grid'] = False

# Set seaborn theme to light grid
sns.set_theme(style="ticks")

# Set background to light gray colour
plt.rcParams['axes.facecolor'] = '#f0f0f0'

# Set nicer font
font = "Arial"
plt.rcParams['font.family'] = font
plt.rcParams['font.size'] = 11

# Create 4x2 subplot grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# First pass to find global min/max values for piano rolls
# and maximum count for histograms
piano_roll_min = float('inf')
piano_roll_max = float('-inf')
hist_max_count = 0

# Define model names for the columns
model_names = {
    "base_model": "Base Model",
    "w/ SMART 100 steps, beta=0.04": "SMART 100 steps\nβ=0.04",
    "w/ SMART 1000 steps, beta=0.04": "SMART 1000 steps\nβ=0.04",
    "w/ SMART 1000 steps, beta=0.00": "SMART 1000 steps\nβ=0.00"
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
    sns.histplot(data['logs']["aes_scores_CE"], bins=40, binrange=(1, 10), 
                ax=axes[0, i], color=colors[i % len(colors)], 
                edgecolor='black', linewidth=0.5, alpha=0.7)
    
    axes[0, i].set_title("Content Enjoyment", fontsize=14, fontweight='bold', fontfamily=font)
    axes[0, i].set_xlabel("Predicted Content Enjoyment", fontsize=12, fontfamily=font)
    axes[0, i].set_ylabel("Count", fontsize=12, fontfamily=font)
    
    # Set consistent y-axis limit across all histograms
    axes[0, i].set_ylim(0, hist_max_count * 1.05)  # Add 5% padding
    
    # Clean up the plot - remove grid and unnecessary spines
    axes[0, i].grid(False)
    axes[0, i].spines['top'].set_visible(False)
    axes[0, i].spines['right'].set_visible(False)
    
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
    
    axes[1, i].set_title("Average Piano Roll", fontsize=14, fontweight='bold', fontfamily=font)
    axes[1, i].set_xlabel("Beats", fontsize=12, fontfamily=font)
    axes[1, i].set_ylabel("Pitch", fontsize=12, fontfamily=font)
    axes[1, i].set_ylim(21, 109)  # Restrict y-axis to 21-109 pitch range
    
    # Clean up the piano roll plots too
    axes[1, i].grid(False)
    axes[1, i].spines['top'].set_visible(False)
    axes[1, i].spines['right'].set_visible(False)

# Apply tight layout
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at the top for column titles

# After tight_layout, add column titles directly above each column
for i, key in enumerate(results.keys()):
    # Get the position of the top axes in this column
    pos = axes[0, i].get_position()
    # Place title at the center of the column
    fig.text(pos.x0 + pos.width/2, 0.96, model_names.get(key, key), 
             ha='center', va='top', fontsize=13, fontweight='bold', fontfamily=font)

# Adjust figure to make room for column titles
plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4)  # More space between subplots

# Save high-resolution versions
plt.savefig('midi_visualization_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('midi_visualization_comparison.png', dpi=300, bbox_inches='tight')

plt.show()
# %%
