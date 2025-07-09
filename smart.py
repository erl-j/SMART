#%%
import miditok
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from symusic import BuiltInSF3
from datasets import Dataset
import os
from tqdm import tqdm
import numpy as np
import random
from processors import RewardManager, MidiTokToSymusicProcessor, TinySoundfontSynthProcessor, AudioBoxAesRewardProcessor
#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

# set SEED 
SEED=1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

GRADIENT_ACCUMULATION_STEPS = 1
USE_BF16 = True
NUM_GENERATIONS=8

TEMPERATURE = 1.0
NUM_ITERATIONS = 1
SCALE_REWARDS = True

NUM_TRAIN_STEPS = 200
LEARNING_RATE = 1e-4
SCHEDULE_TYPE = "linear"
BETA = 0.04

MODEL = "piano-4l"
PROMPT_SOURCE = "procedural"
AUDIO_SAVE_INTERVAL = NUM_ITERATIONS*10
SAVE_STEPS = 20
N_EVAL_PROMPTS=100

BATCH_SIZE=64 if "piano" in MODEL else 32

N_PROMPTS = (NUM_TRAIN_STEPS * BATCH_SIZE // NUM_GENERATIONS) * 10

SAMPLE_RATE = 48_000
SOUNDFONT = "musescore"

REWARD_WEIGHTS = {
    "CE": 1.0,
}

# get latest checkpoint
OUTPUT_DIR = f"artefacts/runs/{MODEL}-{PROMPT_SOURCE}/A-{BETA}-{TEMPERATURE}-{NUM_TRAIN_STEPS}-{SCHEDULE_TYPE}-scale-rewards={SCALE_REWARDS}"

# warn if output dir exists and may be overwritten
if os.path.exists(OUTPUT_DIR):
    print(f"Warning: Output directory {OUTPUT_DIR} already exists and may be overwritten.")
    print("Type 'yes' to continue, or 'no' to abort.")
    response = input()
    if response != "yes":
        raise ValueError("Aborted by user.")
else:
    # remove the output dir if it exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# get more soundfonts here: https://huggingface.co/datasets/projectlosangeles/soundfonts4u
SF_PATH= {
        "musescore": str(BuiltInSF3.MuseScoreGeneral().path(download=True)), 
}[SOUNDFONT]


MAX_COMPLETION_LENGTH = 256
MAX_BEATS = 64
MAX_AUDIO_DURATION = 10

BASE_MODEL_PATH = "lucacasini/metamidipianophi3_4L"
model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
bar_token = tokenizer.vocab["Bar_None"]
position_zero_token = tokenizer.vocab["Position_0"]
timesignature_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("TimeSig_")]
tempo_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Tempo_")]
pitch_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Pitch_")]
velocity_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Velocity_")]
duration_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Duration_")]

match PROMPT_SOURCE:
    case "dataset":
        trn_ds = Dataset.load_from_disk("data/dataset_mmd_piano/train")
        trn_ds = trn_ds.shuffle()
        trn_ds = trn_ds.select(range(N_PROMPTS))

        tst_ds = Dataset.load_from_disk("data/dataset_mmd_piano/test")
        tst_ds = tst_ds.shuffle()
        tst_ds = tst_ds.select(range(N_EVAL_PROMPTS))

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
            prompt_token_ids = [tokenizer.vocab["PAD_None"]] * (max_prompt_length - len(prompt_token_ids)) + prompt_token_ids
            return {"prompt": prompt_token_ids}

        trn_ds = trn_ds.map(extract_prompt)
        tst_ds = tst_ds.map(extract_prompt)
        
    case "procedural":
        def gen():
            for i in range(N_PROMPTS):
                yield {"prompt": [bar_token, 
                                random.choice(timesignature_tokens), 
                                position_zero_token, 
                                random.choice(tempo_tokens), 
                                random.choice(pitch_tokens), 
                                random.choice(velocity_tokens),
                                random.choice(duration_tokens)]
                    }
        trn_ds = Dataset.from_generator(gen)
        tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
        max_prompt_length = len(trn_ds[0]["prompt"])
    case "procedural-no-starting-note":
        def gen():
            for i in range(N_PROMPTS):
                yield {"prompt": [bar_token, 
                                random.choice(timesignature_tokens), 
                                position_zero_token, 
                                random.choice(tempo_tokens), 
                                ]
                    }
        trn_ds = Dataset.from_generator(gen)
        tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
        max_prompt_length = len(trn_ds[0]["prompt"])
    case "no prompt":
        def gen():
            for i in range(N_PROMPTS):
                yield {"prompt": [tokenizer.vocab["BOS_None"]]}
        trn_ds = Dataset.from_generator(gen)
        tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
        max_prompt_length = len(trn_ds[0]["prompt"])
    case _:
        raise ValueError("Invalid prompt source for piano model")
    
reward_manager = RewardManager(
    processors = [
        MidiTokToSymusicProcessor(tokenizer, is_multitrack=False, max_beats=100),
        TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION),
        AudioBoxAesRewardProcessor(),
    ],
    reward_weights = REWARD_WEIGHTS,
    output_dir=OUTPUT_DIR
)

class DummyTokenizer():
    def __init__(self,tokenizer):
        if isinstance(tokenizer.vocab, list):
            self.pad_token_id = tokenizer.token_to_idx["PAD_None"]
            self.eos_token_id = tokenizer.token_to_idx["EOS_None"]
            self.bos_token_id = tokenizer.token_to_idx["BOS_None"]
        else:
            self.pad_token_id = tokenizer.vocab["PAD_None"]
            self.eos_token_id = tokenizer.vocab["EOS_None"]
            self.bos_token_id = tokenizer.vocab["BOS_None"]  
    def decode(self, x, **kwargs):
        return x
    def batch_decode(self, x, **kwargs):
        return x
    def save_pretrained(self, path):
        print(f"Calling save_pretrained with {path} (does nothing)")
    def __call__(self,  **kwargs):
        input_ids = torch.tensor( kwargs["text"])
        attention_mask = torch.where(
            input_ids != self.pad_token_id, 1, 0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
dummy_tokenizer = DummyTokenizer(tokenizer)
# %%
config = GRPOConfig(
    num_iterations=NUM_ITERATIONS,
    scale_rewards=SCALE_REWARDS,
    temperature=TEMPERATURE,
    output_dir=OUTPUT_DIR,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=max_prompt_length,
    max_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    report_to="wandb",
    logging_steps=1,
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=SAVE_STEPS,
    beta=BETA,
    bf16=USE_BF16,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    lr_scheduler_type=SCHEDULE_TYPE,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_manager,
    args =  config,
    train_dataset=trn_ds,
    processing_class=dummy_tokenizer,
    eval_dataset=tst_ds,

)
# save model
trainer.train()
trainer.save_model()
