#%%
import symusic

# get midi
# midi_dir = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-1000-10s/post_eval/eval/midi/0"
# midi_dir = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.0-1.0-1000-10s/post_eval/eval/midi/0"
# midi_dir = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s/post_eval/eval/midi/0"
midi_dir = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s/pre_eval/eval/midi/0"
# midi_dir = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-200-10s/post_eval/eval/midi/0"s
import glob

# find all midi files
midi_files = glob.glob(midi_dir + "/*.mid")

def get_pr(sm):
    sm = sm.copy().resample(12,0)
    pr = sm.pianoroll(modes=["frame"],
                      pitch_range=(0,128),
                      encode_velocity=False,
    ).sum(0).sum(0)
    return pr

# show pr of first midi files
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# now plot average piano roll (first 64 frames)
piano_rolls = []
skip_index = 480
for i in tqdm(range(len(midi_files))):
    midi = symusic.Score(midi_files[i])
    pr = get_pr(midi)
    if pr.shape[1] < 12*16:
        # only take first 32 frames
        continue
    piano_rolls.append(pr[:, :12*16])
piano_rolls = np.array(piano_rolls)
# average over all piano rolls
avg_piano_roll = np.mean(piano_rolls, axis=0)
# plot average piano roll
plt.imshow(1-avg_piano_roll, aspect='auto', cmap='gray')
plt.title("Average Piano Roll")
plt.xlabel("Time")
plt.ylabel("Pitch")
# plt.colorbar(label="Velocity")
plt.show()

# %%

# show pr for 3 examples
for i in range(3):
    midi = symusic.Score(midi_files[i])
    pr = get_pr(midi)
    plt.imshow(pr, aspect='auto', cmap='gray')
    plt.title(midi_files[i])
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.colorbar(label="Velocity")
    plt.show()
