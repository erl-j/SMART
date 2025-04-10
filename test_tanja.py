#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import miditok
from util import preview_sm
from symusic import BuiltInSF3, Synthesizer
import IPython.display as ipd

checkpoint = "outputs/mt/lively-waterfall-47/checkpoint-25000"

from tokenisation import TanjaTokenizer, TanjaTokenizerConfig

# %%

tokenizer = TanjaTokenizer(
    TanjaTokenizerConfig(
        ticks_per_beat=96,
        coarse_ticks_per_beat=12,
        tempo_range=(60, 200),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        n_events=300,
    )
)

model = AutoModelForCausalLM.from_pretrained(checkpoint)


# %%
import torch
# bos tempo program pitch
prompt = ["BOS_None", "Tempo_87", "Pitch_Drum50"]
input_ids = torch.tensor(tokenizer.tokens_to_ids(prompt))[None,...]
position_ids = torch.tensor([0,2,1])[None,...]
# increment last position id by 1
print(position_ids)
print(position_ids)

# apply model.forward
out = model(
    input_ids=input_ids,
    position_ids=position_ids,
    return_dict=True,
    # attention_mask=torch.ones(input_ids.shape).to(input_ids.device),
    # use_cache=True
)
print(out.logits.shape)
vocab = tokenizer.vocab

probs = out.logits[0, -1, :].softmax(dim=-1)

print("\n")
# get top 10 probability tokens
top_10 = torch.topk(probs, 10)
for i in range(10):
    print(vocab[top_10.indices[i].item()], top_10.values[i].item())

#%%

# generate a sequence
out = model.generate(
    input_ids= input_ids,
    max_length=2048,
    do_sample=True,
    pad_token_id=tokenizer.token_to_idx["PAD_None"],
    bos_token_id=tokenizer.token_to_idx["BOS_None"],
    eos_token_id=tokenizer.token_to_idx["EOS_None"],
    num_return_sequences=1,
    temperature=1.0,
    use_cache=True
    # top_k=1,
    # top_p=0.95,
)

# decode the sequence
tokens = tokenizer.ids_to_tokens(out[0].tolist())

print(tokens)
sm = tokenizer.tokens_to_midi(tokens)

preview_sm(sm)

#%%

# replace drums
# take head (everything before the first track_None)
head_tokens = tokens[:tokens.index("Track_None")]
# take tail (everything after the first track_None)
tail_tokens = tokens[tokens.index("Track_None"):]

def split_list(lst, delimiter):
    """Split a list into a list of lists, using the delimiter as a separator.
    So [sep, a, b, sep, a, b ] -> [[a, b], [a, b]]
    """
    result = []
    current = []
    for item in lst:
        if item == delimiter:
            result.append(current)
            current = []
        else:
            current.append(item)
    if current:
        result.append(current)
    return result
tracks = split_list(tail_tokens, "Track_None")

# remove empty tracks
tracks = [tr for tr in tracks if len(tr) > 0]

for tr in tracks:
    print(tr)

program_tokens = head_tokens[2:]

print(f"found {len(program_tokens)} programs")
print(f"found {len(tracks)} tracks")

print(program_tokens)

head_head_tokens = head_tokens[:2]

#%%

program_to_replace = "Drums"

# get index replace program
program_idx = program_tokens.index(f"Program_{program_to_replace}")

program_tokens = [pr for i, pr in enumerate(program_tokens) if i != program_idx]
# remove index from tracks
tracks = [tr for i, tr in enumerate(tracks) if i != program_idx]

# new_program
new_program = "Drums"
# add new program to end
program_tokens.append(f"Program_{new_program}")

# add Track_None to beginning of each track
for i, tr in enumerate(tracks):
    tracks[i] = ["Track_None"] + tr

# add a new track with only "Track_None"
tracks.append(["Track_None"])

def flatten(lst):
    """Flatten a list of lists."""
    return [item for sublist in lst for item in sublist]

new_tokens = head_head_tokens + program_tokens + flatten(tracks)

# convert into ids

# convert to ids
ids = tokenizer.tokens_to_ids(new_tokens)

# convert to tensor
input_ids = torch.LongTensor(ids)

# generate
out = model.generate(
    input_ids=input_ids[None,...],
    max_length=2048,
    do_sample=True,
    pad_token_id=tokenizer.token_to_idx["PAD_None"],
    bos_token_id=tokenizer.token_to_idx["BOS_None"],
    eos_token_id=tokenizer.token_to_idx["EOS_None"],
    num_return_sequences=1,
    temperature=1.0,
    use_cache=True
)

# decode and sample
tokens = tokenizer.ids_to_tokens(out[0].tolist())
sm = tokenizer.tokens_to_midi(tokens)
preview_sm(sm)


# %%


SAMPLE_RATE = 41_000
MAX_AUDIO_DURATION = 32
AUDIO_SAVE_INTERVAL = 10

#%%
print(sm.tracks)
def set_drum_to_program(sm):
    # remove expression
    sm = sm.copy()
    for track in sm.tracks:
        if track.is_drum:
            track.program=0
            track.controls=[]
            track.pitch_bends=[]
            track.pedals = []
            for note in track.notes:
                note.duration = sm.tpq
    return sm

sm = set_drum_to_program(sm)

SF_PATH= {
    "musescore": str(BuiltInSF3.MuseScoreGeneral().path(download=True)), 
    "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
    "monalisa":"./soundfonts/Monalisa_GM_v2_105.sf2",
    "ephesus":"./soundfonts/Ephesus_GM_Version_1_00.sf2",
    "touhou" : "./soundfonts/Touhou.sf2",
    "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
    "fluidr3": "./soundfonts/FluidR3 GM.sf2",
}

#%%
import numpy as np
import subprocess
import os
from tempfile import NamedTemporaryFile
from scipy.io import wavfile
import os
import numpy as np
import subprocess
from tempfile import NamedTemporaryFile, mkdtemp
from scipy.io import wavfile
import shutil
import glob


#%%
    
midi_path = "test.mid"
sm.dump_midi(midi_path)

# %%
import numpy as np
import tinysoundfont

import numpy as np


renderer = MidiRenderer(SF_PATH["sgm"], samplerate=SAMPLE_RATE)

#%%
from tqdm import tqdm
audio = renderer.render(midi_path, duration_seconds=MAX_AUDIO_DURATION)
ipd.display(ipd.Audio(audio, rate=SAMPLE_RATE))

#%%
# render 100 times
for i in tqdm(range(32)):
    audio = renderer.render(midi_path, duration_seconds=MAX_AUDIO_DURATION)

#%%
import joblib


#%%
print(audio)
ipd.display(ipd.Audio(audio, rate=SAMPLE_RATE))
# %%
