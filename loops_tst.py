#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import miditok
from util import preview_sm
from symusic import BuiltInSF3, Synthesizer
import IPython.display as ipd

checkpoint = "outputs/mt/treasured-cosmos-19/checkpoint-325000"
# checkpoint = "outputs/mt/ruby-microwave-20/checkpoint-425000"

tokenizer_config = miditok.TokenizerConfig.load_from_json("./data/tokenizer_config.json")
tokenizer = miditok.REMI(tokenizer_config)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# print vocab to file
with open("vocab.txt", "w") as f:
    for token in tokenizer.vocab:
        f.write(f"{token}\n")

# %%
import torch

prompt = ["BOS_None", "Program_-1"]
input_ids = torch.tensor([tokenizer.vocab[token] for token in prompt])[None,...]

# generate a sequence
out = model.generate(
    # input_ids= input_ids,
    max_length=2048,
    do_sample=True,
    pad_token_id=tokenizer.vocab["PAD_None"],
    bos_token_id=tokenizer.vocab["BOS_None"],
    eos_token_id=tokenizer.vocab["EOS_None"],
    num_return_sequences=1,
    temperature=1.0,
    use_cache=True
    # top_k=1,
    # top_p=0.95,
)

#%%
# decode the sequence
tokens = tokenizer._ids_to_tokens(out[0].tolist())
print(tokens)
sm = tokenizer.decode(out[0])

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
