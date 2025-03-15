#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda:0"

checkpoint = "session-gpt/checkpoints/nospace/checkpoint-8000"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# save tokenizer
tokenizer.save_pretrained(checkpoint)
# generate
input_ids = tokenizer.encode("This tune is a ", return_tensors="pt").to(device)

out = model.generate(
    input_ids=input_ids,
    max_length=200,
    temperature=1.0,
    do_sample=True,
)
out_str = tokenizer.decode(out[0])

print(out_str)

# get everything after the @ and before any €
abc = out_str.split("@")[1].split("€")[0]
print(abc)

import os
os.makedirs("../session-gpt/artefacts/session_gt2d", exist_ok=True)
# write to .abc file
with open("artefacts/8000.abc", "w") as f:
    f.write(tokenizer.decode(out[0], skip_special_tokens=True))

print(abc)
# %%
import symusic

score = symusic.Score.from_abc(abc)

#%%
import miditok
import symusic
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
from symusic import Synthesizer, BuiltInSF3, dump_wav
from audiobox_aesthetics.infer import initialize_predictor

sample_rate = 44100
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)

synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)

#%%
audio = synth.render(score)

import matplotlib.pyplot as plt
import IPython.display as ipd
def play_audio(audio):
    plt.figure(figsize=(10, 2))
    plt.plot(audio[0])
    plt.show()
    ipd.display(ipd.Audio(audio[0], rate=44100))

play_audio(audio)


# %%
# render abc
