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
import transformers
import miditok
import torch
import tempfile
import soundfile as sf
import os
from datasets import Dataset

# base_out_dir = "artefacts/demo_samples_4L_overoptimization"
base_out_dir = "artefacts/demo_samples_overoptimized_beta=0.04"
TEMPERATURE=1.0
device = "cuda:7"



# ckpt_path = "artefacts/all_runs_4/piano-long-dataset/ce-piano-0.04-1.0-100-constant/checkpoint-100"
# BASE_MODEL_PATH = "lucacasini/metamidipianophi3_6L_long"

# ckpt_path = "artefacts/all_runs_3/piano-long-dataset/aes-ce-0.04-1.0-200/checkpoint-200"
# ckpt_path = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-ce-0.04-1-200/checkpoint-200"
# ckpt_path = "artefacts/all_runs_5/piano-4l-dataset/aes-ce-0.04-1.0-200-linear-scale-rewards=True/checkpoint-200"
ckpt_path = "artefacts/all_runs_3/piano-4l-procedural-no-starting-note/aes-ce-0.04-1-1000/checkpoint-1000"
BASE_MODEL_PATH = "lucacasini/metamidipianophi3_4L"

tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
#%%
# procedural prompt path
run_path = "/".join(ckpt_path.split("/")[:-1])
# load all logs
prelogs = pd.read_parquet(run_path + "/pre_eval/eval/rl_logs/0/logs.parquet")
print("Rows in pre eval: ", len(prelogs))


prelogs["stage"] = "pre"
# concat pre and post logs
# for all columns that are dicts, expand them and prepend dict to the key as name


#%%
# Sample from logs for procedural prompts
sample_size = 15
prelog_sample = prelogs.sample(sample_size, random_state=42)

# turn into list of dict containing idx and prompt
sampled_prelogs = []
for i, row in prelog_sample.iterrows():
    prompt = row["prompt"]
    # convert to token ids
    # add to list
    sampled_prelogs.append({"idx": i, "prompt": prompt})

# pirnt sampled prelogs
print("Sampled prelogs: ", sampled_prelogs)
#%%
# Load dataset prompts
N_EVAL_PROMPTS = sample_size
max_prompt_length = 4

# model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")


tst_ds = Dataset.load_from_disk("data/dataset_mmd_piano/test")
tst_ds = tst_ds.shuffle()
tst_ds = tst_ds.select(range(sample_size))

max_prompt_length = 4

def extract_prompt(example):
    tokens = example["tokens"]
    # convert to tokens
    # tokens = tokenizer._ids_to_tokens(token_ids)
    # find first tempo token
    first_tempo_token = None
    for i, token in enumerate(tokens):
        if token.startswith("Tempo_"):
            first_tempo_token = token
            break
    # find first timesig token
    first_timesig_token = None
    for i, token in enumerate(tokens):
        if token.startswith("TimeSig_"):
            first_timesig_token = token
            break
    prompt_tokens = ["Bar_None"] + [first_timesig_token] + ["Position_0"] + [first_tempo_token]
    # convert to token ids
    prompt_token_ids = tokenizer._tokens_to_ids(prompt_tokens)
    # pad left with PAD tokens until max_prompt_length
    prompt_token_ids = prompt_token_ids
    return {"prompt": prompt_token_ids}

tst_ds = tst_ds.map(extract_prompt)
#%%
# Configure model and processors
SF_PATH = "soundfonts/Yamaha-C5-Salamander-JNv5_1.sf2"
SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 32
MAX_LENGTH = 2048
MIN_DURATION = 1
MAX_EMPTY_BEAT_RATE = 1.0

# Import after defining SF_PATH
from util import preview_sm, sm_seconds
from processors import TinySoundfontSynthProcessor
processor = TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION)

# Define output directories
procedural_dir = f"{base_out_dir}/procedural_prompts"
dataset_dir = f"{base_out_dir}/dataset_prompts"

# Create output directories
os.makedirs(procedural_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

#%%
# Load tokenizer
tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)

# Map dataset to extract prompts
tst_ds = tst_ds.map(extract_prompt)

#%%
def generate_and_save_samples(model, prompts, out_dir, prefix=""):
    """
    Generate samples for a list of prompts and save them to the specified directory.
    
    Args:
        model: The model to use for generation
        prompts: List of prompts (list of dicts with 'prompt' key)
        out_dir: Directory to save samples to
        prefix: Prefix for filenames
    """
    for i, row in enumerate(prompts):
        attempt = -1
        empty_beat_rate = 1
        duration = 0
        
        assert "prompt" in row, f"Prompt not found in row, key: {row.keys()}"
        # Get prompt token ids
        if isinstance(row, pd.Series):
            prompt_token_ids = row["prompt"]
        else:
            prompt_token_ids = row["prompt"]

        print("Prompt token ids: ", prompt_token_ids)
        
        # Try generating until we get a valid sample
        while empty_beat_rate > MAX_EMPTY_BEAT_RATE or duration < MIN_DURATION:
            attempt += 1
            if attempt > 20:
                print(f"Giving up after 10 attempts for prompt {i}")
                break
                
            # Print prompt information
            print("Prompt and completion tokens: ", prompt_token_ids)
            prompt_tokens = tokenizer._ids_to_tokens(prompt_token_ids)
            print("Prompt tokens: ", prompt_tokens)

            # Generate output
            output = model.generate(
                input_ids=torch.tensor(prompt_token_ids)[None,...].to(device),
                max_length=MAX_LENGTH,
                do_sample=True,
                num_return_sequences=1,
                temperature=TEMPERATURE,
            )

            print("Output tokens: ", output.shape)
            output_tokens = tokenizer._ids_to_tokens(output[0].tolist())
            print("Output tokens: ", output_tokens)
            
            # Convert to MIDI
            sm = tokenizer.decode(output.tolist())

            if sm.note_num() == 0:
                print("No notes in generated sequence")
                continue

            # Check quality metrics
            with tempfile.NamedTemporaryFile(suffix=".mid") as f:
                sm.dump_midi(f.name)
                mp = muspy.read_midi(f.name)
                empty_beat_rate = muspy.empty_beat_rate(mp)
                print("Empty beat rate: ", empty_beat_rate)
            
            duration = sm_seconds(sm)
            print(f"Duration: {duration}s")
        
        if attempt <= 20:
            # Convert to audio
            records = [{"sm": sm, "sm_duration": duration}]
            out_records = processor(records)

            # Save audio
            audio = out_records[0]["audio"]
            print("Audio shape: ", audio.shape)
            
            # Create a safe filename
            tokens_str = '_'.join([t.replace("/", "d") for t in prompt_tokens[:4]])
            filename = f"{prefix}_index_{i}_{tokens_str}_attempt_{attempt}.wav"
            filepath = os.path.join(out_dir, filename)
            
            sf.write(filepath, audio.T, SAMPLE_RATE)
            print(f"Saved to {filepath}")

#%%
# Function to load model either from base or finetuned checkpoint
def load_model(model_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype="auto"
    )
    return model.to(device)

#%%
# Generate samples for both models and both prompt types
model_configs = [
    {"name": "base", "path": BASE_MODEL_PATH},
    {"name": "finetuned", "path": ckpt_path}
]

#%%

#%%
# Generate samples for procedural prompts
for model_config in model_configs:
    print(f"Generating samples for {model_config['name']} model with procedural prompts")
    model = load_model(model_config["path"])
    generate_and_save_samples(
        model,
        sampled_prelogs,
        procedural_dir,
        prefix=f"{model_config['name']}"
    )

#%%
# # Generate samples for dataset prompts
# for model_config in model_configs:
#     print(f"Generating samples for {model_config['name']} model with dataset prompts")
#     model = load_model(model_config["path"])
#     generate_and_save_samples(
#         model,
#         tst_ds,
#         dataset_dir,
#         prefix=f"{model_config['name']}"
#     )

# %%

