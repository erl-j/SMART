#%%
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import symusic
import muspy
# run_path = "artefacts/all_runs_2/piano-procedural/aes-0.04-1-100"
# run_path = "artefacts/all_runs_2/mil-dataset/pam-iou-0.04-1-100"
# load all logs


pre_run = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-0.04-1-200/pre_eval/eval/"
post_run = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-0.04-1-200/post_eval/eval/"

prelogs = pd.read_parquet(f"{pre_run}/rl_logs/0/logs.parquet")
print("Rows in pre eval: ", len(prelogs))

# print reward_weights
reward_weights = prelogs["reward_weights"].iloc[0]
print("Reward weights: ", reward_weights)
postlogs = pd.read_parquet(f"{post_run}/rl_logs/0/logs.parquet")
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


pre_audio_paths = glob.glob(f"{pre_run}/audio/0/*.wav", recursive=True)
post_audio_paths =  glob.glob(f"{post_run}/audio/0/*.wav", recursive=True)

# create dict of audio paths with stage and idx
audio_paths = []
for path in pre_audio_paths:
    idx = int(path.split("/")[-1].split("_")[-1].replace(".wav", ""))
    audio_paths.append({"audio_path": path, "stage": "pre", "idx": idx})

for path in post_audio_paths:
    idx = int(path.split("/")[-1].split("_")[-1].replace(".wav", ""))
    audio_paths.append({"audio_path": path, "stage": "post", "idx": idx})
# create dataframe from dict
audio_paths = pd.DataFrame.from_records(audio_paths)

print(logs.columns)
print(audio_paths.columns)
# now merge audio paths with logs
logs = pd.merge(logs, audio_paths, on=["idx", "stage"], how="left")
# %%

import torchaudio
from IPython.display import Audio, display
# for first 10 indices. play audio pre and post

audio, sr = torchaudio.load(logs["audio_path"].iloc[0])

display(Audio(audio.numpy(), rate=sr))
   
#%%

# print one row

#%%
import pandas as pd
import numpy as np
import torchaudio
import random
from IPython.display import Audio, display, HTML
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

print(len(logs))

# Set up the test
NUM_COMPARISONS = 20
test_results = []

# Prepare data
# We'll randomly select NUM_COMPARISONS indices that exist in both pre and post stages
valid_indices = set(logs[logs['stage'] == 'pre']['idx'].unique()) & set(logs[logs['stage'] == 'post']['idx'].unique())
print(f"Valid indices for comparison: {len(valid_indices)}")
if len(valid_indices) >= NUM_COMPARISONS:
    selected_indices = random.sample(list(valid_indices), NUM_COMPARISONS)
else:
    raise ValueError(f"Not enough valid indices for {NUM_COMPARISONS} comparisons. Only {len(valid_indices)} available.")

print(f"Selected {NUM_COMPARISONS} random indices for comparison: {selected_indices}")
print("For each comparison, listen to both samples and enter 'A' or 'B' for the one you prefer.")
print()

# Create a list to store the true order and user's choices
comparison_data = []

# Run the test
for i, idx in enumerate(selected_indices):
    # Get pre and post samples for this index
    pre_sample = logs[(logs['idx'] == idx) & (logs['stage'] == 'pre')].iloc[0]
    post_sample = logs[(logs['idx'] == idx) & (logs['stage'] == 'post')].iloc[0]
    
    # Randomly assign to A and B
    is_a_pre = random.choice([True, False])
    sample_a = pre_sample if is_a_pre else post_sample
    sample_b = post_sample if is_a_pre else pre_sample
    
    # Load the audio
    audio_a, sr_a = torchaudio.load(sample_a['audio_path'])
    audio_b, sr_b = torchaudio.load(sample_b['audio_path'])
    
    # Store the comparison data
    comparison_data.append({
        'comparison_number': i + 1,
        'idx': idx,
        'is_a_pre': is_a_pre,
        'a_path': sample_a['audio_path'],
        'b_path': sample_b['audio_path']
    })
    
    # Display the comparison
    print(f"Comparison {i+1} of {NUM_COMPARISONS} (Index: {idx})")
    print("Sample A:")
    display(Audio(audio_a.numpy(), rate=sr_a))
    print("Sample B:")
    display(Audio(audio_b.numpy(), rate=sr_b))
    sleep(1)  # Pause for 5 seconds to allow listening
    
    # User needs to manually enter their choice at this point
    choice = input("Which sample do you prefer? (A/B): ")
    while choice.upper() not in ['A', 'B']:
        choice = input("Please enter either 'A' or 'B': ")
    
    # Save the choice
    comparison_data[i]['choice'] = choice.upper()
    print()

    # clear ipython output
    clear_output(wait=True)

# Calculate results
pre_preferred = 0
post_preferred = 0

for result in comparison_data:
    if result['choice'] == 'A':
        if result['is_a_pre']:
            pre_preferred += 1
        else:
            post_preferred += 1
    elif result['choice'] == 'B':
        if result['is_a_pre']:
            post_preferred += 1
        else:
            pre_preferred += 1

# Show results
print("=== LISTENING TEST RESULTS ===")
print(f"Pre versions preferred: {pre_preferred}/{NUM_COMPARISONS} ({pre_preferred/NUM_COMPARISONS*100:.1f}%)")
print(f"Post versions preferred: {post_preferred}/{NUM_COMPARISONS} ({post_preferred/NUM_COMPARISONS*100:.1f}%)")
print()

# Create a bar chart of the results
plt.figure(figsize=(8, 5))
versions = ['Pre', 'Post']
preferences = [pre_preferred, post_preferred]
percentages = [p/NUM_COMPARISONS*100 for p in preferences]

bars = plt.bar(versions, percentages, color=['#66b3ff', '#ff9999'])

# Add labels and title
plt.xlabel('Version')
plt.ylabel('Preference Percentage (%)')
plt.title('Listening Test Preferences')
plt.ylim(0, 100)

# Add the preference count on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{preferences[i]}/{NUM_COMPARISONS}',
            ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Display detailed results table
results_table = []
for result in comparison_data:
    pre_or_post_chosen = 'Pre' if (
        (result['choice'] == 'A' and result['is_a_pre']) or 
        (result['choice'] == 'B' and not result['is_a_pre'])
    ) else 'Post'
    
    results_table.append({
        'Comparison': result['comparison_number'],
        'Index': result['idx'],
        'Sample A was': 'Pre' if result['is_a_pre'] else 'Post',
        'You chose': result['choice'],
        'You preferred': pre_or_post_chosen
    })

print("Detailed Results:")
print(pd.DataFrame(results_table))
# %%

# for 10 first indices show a and b audio side by side
for i, idx in enumerate(selected_indices[:10]):
    # Get pre and post samples for this index
    pre_sample = logs[(logs['idx'] == idx) & (logs['stage'] == 'pre')].iloc[0]
    post_sample = logs[(logs['idx'] == idx) & (logs['stage'] == 'post')].iloc[0]
    
    # Randomly assign to A and B
    is_a_pre = True
    sample_a = pre_sample if is_a_pre else post_sample
    sample_b = post_sample if is_a_pre else pre_sample
    
    # Load the audio
    audio_a, sr_a = torchaudio.load(sample_a['audio_path'])
    audio_b, sr_b = torchaudio.load(sample_b['audio_path'])
    
    # Display the comparison
    print(f"Comparison {i+1} of {NUM_COMPARISONS} (Index: {idx})")
    print("Sample A:")
    display(Audio(audio_a.numpy(), rate=sr_a))
    print("Sample B:")
    display(Audio(audio_b.numpy(), rate=sr_b))
    print()
# %%
