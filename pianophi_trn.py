#%%
import miditok
import symusic
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
from symusic import Synthesizer, BuiltInSF3, dump_wav
from audiobox_aesthetics.infer import initialize_predictor
import torch.nn.functional as F
from datasets import load_dataset
from transformers import ClapModel, ClapProcessor
import os
import random
#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"
BATCH_SIZE=64
NUM_GENERATIONS=8

#%%
# audio rendering settings
SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 10
PIANO_PROGRAM = 0
SF_PATH = BuiltInSF3.MuseScoreGeneral().path(download=True)
# SF_PATH = "./soundfonts/SGM-V2.01-XG-2.04.sf2"
synth = Synthesizer(
    sf_path = SF_PATH, # the path to the soundfont
    sample_rate = SAMPLE_RATE, # the sample rate of the output wave, sample_rate is the default value
)
#%%
BASE_MODEL_PATH = "lucacasini/metamidipianophi3"
model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
OUTPUT_DIR = "artefacts/pianophi-test"

#%%
AUDIO_SAVE_INTERVAL = 10
#%%
def set_piano_program(sm):
    for track in sm.tracks:
        track.program = PIANO_PROGRAM
    return sm
#%%
bar_token = tokenizer.vocab["Bar_None"]
position_zero_token = tokenizer.vocab["Position_0"]
timesignature_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("TimeSig_")]
tempo_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Tempo_")]
pitch_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Pitch_")]
velocity_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Velocity_")]

print(f"Found {len(timesignature_tokens)} time signature tokens")
print(f"Found {len(tempo_tokens)} tempo tokens")
print(f"Found {len(pitch_tokens)} pitch tokens")
print(f"Found {len(velocity_tokens)} velocity tokens")

def gen():
    for i in range(1_000_000):
        yield {"prompt": [bar_token, 
                        random.choice(timesignature_tokens), 
                        position_zero_token, 
                        random.choice(tempo_tokens), 
                        random.choice(pitch_tokens), 
                        random.choice(velocity_tokens)]
            }
ds = Dataset.from_generator(gen)
#%%
aes_predictor = initialize_predictor()
def get_aes_scores(records):
    # prepare inputs. 
    predictor_inputs = [{"path": torch.tensor(record["audio"]).float(), "sample_rate": SAMPLE_RATE, "idx":i} for i, record in enumerate(records) if record["audio"] is not None]
    scores = aes_predictor.forward(predictor_inputs)
    # put back scores to records that have audio
    record_with_audio_index = 0
    for i, record in enumerate(records):
        if record["audio"] is not None:
            record["aes_scores"] = scores[record_with_audio_index]
            record_with_audio_index += 1
        else:
            record["aes_scores"] = None
    return records

reward_step = 0
def aes_reward(completions, return_records=False, **kwargs):
    global reward_step
    global logs
    # print input arguments
    prompts = torch.tensor(kwargs["prompts"])
    full_seqs = torch.cat([prompts, completions.cpu()], dim=1)
    sms = [tokenizer(full_seqs[i].cpu().numpy()[None,...] ) for i in range(full_seqs.shape[0])]
    sms = [set_piano_program(sm) for sm in sms]
    records = [{"completion": completions[i].cpu(), "sm": sms[i], "prompt":prompts[i], "prompt_and_completion":full_seqs[i] } for i in range(full_seqs.shape[0])]
    for record in records:
        try:
            record["audio"] = synth.render(record["sm"])
            if record["audio"].shape[1] > MAX_AUDIO_DURATION * SAMPLE_RATE:
                record["audio"] = record["audio"][:,:MAX_AUDIO_DURATION * SAMPLE_RATE]
        except Exception as e:
            print(f"Error rendering audio: {e}")
            record["audio"] = None
    records = get_aes_scores(records)
    records = [ 
        {**record, "rewards":
        sum([record["aes_scores"]["CE"], record["aes_scores"]["CU"], record["aes_scores"]["PC"], record["aes_scores"]["PQ"]]) 
        if record["aes_scores"] is not None else 0}
         for record in records
    ]
    if return_records:
        return records
    else:
        rewards = [record["rewards"] for record in records]
        print(f"average reward: {sum(rewards)/len(rewards)}")
        print(f"Rewards: {rewards}")
        print(f"reward step: {reward_step}")

        logs = []
        dont_log = ["audio", "sm"]
        for record in records:
            log = {**record}
            for key in dont_log:
                log.pop(key)
            logs.append(log)
        # save logs
        os.makedirs(f"{OUTPUT_DIR}/rl_logs/{reward_step}", exist_ok=True)
        # save as parquet
        ds = Dataset.from_list(logs)
        ds.to_parquet(f"{OUTPUT_DIR}/rl_logs/{reward_step}/logs.parquet")

        os.makedirs(f"{OUTPUT_DIR}/midi/{reward_step}", exist_ok=True)
        for i in range(len(sms)):
            sms[i].dump_midi(f"{OUTPUT_DIR}/midi/{reward_step}/reward={rewards[i]}_{i}.mid")

        if reward_step % AUDIO_SAVE_INTERVAL == 0:
            os.makedirs(f"{OUTPUT_DIR}/audio/{reward_step}", exist_ok=True)
            for i in range(len(sms)):
                try:
                    dump_wav( f"{OUTPUT_DIR}/audio/{reward_step}/reward={rewards[i]}_{i}.wav", records[i]["audio"], SAMPLE_RATE, use_int16=True)
                except Exception as e:
                    print(f"Error dumping wav: {e}")
        reward_step += 1
        return rewards
#%%
class DummyTokenizer():
    def __init__(self,tokenizer):
        self.pad_token_id = tokenizer.vocab["PAD_None"]
        self.eos_token_id = tokenizer.vocab["EOS_None"]
        self.bos_token_id = tokenizer.vocab["BOS_None"]  
    def decode(self, x, **kwargs):
        return x
    def batch_decode(self, x, **kwargs):
        return x
    def save_pretrained(self, path):
        print(f"Calling save_pretrained with {path} (does nothing)")
    def __call__(self, x, **kwargs):
        return {
            "input_ids": torch.tensor(x),
            "attention_mask": torch.ones_like(torch.tensor(x))
        }
dummy_tokenizer = DummyTokenizer(tokenizer)
# %%

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=250,
    max_prompt_length=6,
    num_train_epochs=10,
    learning_rate=1e-5,
    report_to="wandb",
    logging_steps=1,
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    beta=0.1
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=aes_reward,
    args =  config,
    train_dataset=ds,
    processing_class=dummy_tokenizer,

)
# save model
trainer.train()
trainer.save_model()



#%%
