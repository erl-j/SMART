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
# run_path = "artefacts/all_runs_2/piano-procedural/aes-0.04-1-100"
# run_path = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-ce-0.04-1-200/"
run_path = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s"
# run_path = "artefacts/all_runs_2/mil-dataset/pam-iou-0.04-1-100"
# run_path = "artefacts/all_runs_3/irma-dataset/aes-ce-iou-pam-0.16-0.9-1000/"
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
import transformers
import miditok
from util import preview_sm, sm_seconds
import torch
from processors import TinySoundfontSynthProcessor
import muspy
import tempfile
import soundfile as sf

BASE_MODEL_PATH = "lucacasini/metamidipianophi3_6L_long"
model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
# set seed

SF_PATH = "soundfonts/Yamaha-C5-Salamander-JNv5_1.sf2"

SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 10

processor = TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION)

device = "cuda:2"
model.to(device)

# get 15 random samples from prelogs
sample_size = 15
sampled_prelogs = prelogs.sample(sample_size, random_state=0)

import os
os.makedirs("listening_test", exist_ok=True)

prompt_index = 0
for i, row in sampled_prelogs.iterrows():
    attempt = -1
    empty_beat_rate = 1
    duration = 0
    while empty_beat_rate > 0 and duration < 5:
        attempt += 1
        # print prompt_and_completion_tokens for pre
        prompt_token_ids = row["prompt"]
        print("Prompt and completion tokens: ", prompt_token_ids)
        # convert to tokens
        prompt_tokens = tokenizer._ids_to_tokens(prompt_token_ids)
        print("Prompt tokens: ", prompt_tokens)

        output = model.generate(
            input_ids=torch.tensor(prompt_token_ids)[None,...].to(device),
            max_length=512,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
        )

        print("Output tokens: ", output.shape)
        # convert to tokens
        output_tokens = tokenizer._ids_to_tokens(output[0].tolist())
        print("Output tokens: ", output_tokens)
        # convert to midi
        sm = tokenizer.decode(output.tolist())

        with tempfile.NamedTemporaryFile(suffix=".mid") as f:
            sm.dump_midi(f.name)
            mp = muspy.read_midi(f.name)
            # get empty beats 
            empty_beat_rate = muspy.empty_beat_rate(mp)
            print("Empty beat rate: ", empty_beat_rate)
        
        duration = sm_seconds(sm)
    # convert to audio
    records = [{"sm": sm, "sm_duration": duration}]
    out_records = processor(records)

    # save audio
    audio = out_records[0]["audio"]
    print("Audio shape: ", audio.shape)
    sf.write("listening_test/"+f"pre_index_{prompt_index}_prompt_{i}_{'_'.join(prompt_tokens)}_attempt_{attempt}".replace("/","d") + ".wav", audio.T, SAMPLE_RATE)
    # save audio to file
    prompt_index += 1

    
#%%

model = transformers.AutoModelForCausalLM.from_pretrained("artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-100-10s/checkpoint-100", trust_remote_code=True, torch_dtype="auto")
model = model.to(device)
prompt_index = 0
for i, row in sampled_prelogs.iterrows():
    attempt = -1
    empty_beat_rate = 1
    duration = 0
    while empty_beat_rate > 0 and duration < 5:
        attempt += 1
        # print prompt_and_completion_tokens for pre
        prompt_token_ids = row["prompt"]
        print("Prompt and completion tokens: ", prompt_token_ids)
        # convert to tokens
        prompt_tokens = tokenizer._ids_to_tokens(prompt_token_ids)
        print("Prompt tokens: ", prompt_tokens)

        output = model.generate(
            input_ids=torch.tensor(prompt_token_ids)[None,...].to(device),
            max_length=512,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
        )

        print("Output tokens: ", output.shape)
        # convert to tokens
        output_tokens = tokenizer._ids_to_tokens(output[0].tolist())
        print("Output tokens: ", output_tokens)
        # convert to midi
        sm = tokenizer.decode(output.tolist())

        with tempfile.NamedTemporaryFile(suffix=".mid") as f:
            sm.dump_midi(f.name)
            mp = muspy.read_midi(f.name)
            # get empty beats 
            empty_beat_rate = muspy.empty_beat_rate(mp)
            print("Empty beat rate: ", empty_beat_rate)
        
        duration = sm_seconds(sm)
    # convert to audio
    records = [{"sm": sm, "sm_duration": duration}]
    out_records = processor(records)

    # save audio
    audio = out_records[0]["audio"]
    print("Audio shape: ", audio.shape)
    sf.write("listening_test/"+f"post_index_{prompt_index}_prompt_{i}_{'_'.join(prompt_tokens)}_attempt_{attempt}".replace("/","d") + ".wav", audio.T, SAMPLE_RATE)
    # save audio to file
    prompt_index += 1


# %%
