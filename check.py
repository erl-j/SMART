#%%
import soundfile as sf

audio_path = "artefacts/all_runs_2/mil-dataset/aes-0.04-1-10/pre_eval/eval/audio/0/reward=0.679375680287679_7.wav"

import glob

audio_paths = glob.glob("artefacts/all_runs_2/mil-dataset/aes-0.04-1-10/pre_eval/eval/audio/0/*.wav", recursive=True)

# %%
# load audio
import torchaudio

import torch
import matplotlib.pyplot as plt

for i in range(10):
    audio, sr = torchaudio.load(audio_paths[i])
    # plot audio
    plt.figure()
    plt.plot(audio[0])
    plt.title(audio_paths[i])
    plt.show()


# %%
