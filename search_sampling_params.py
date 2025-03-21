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
# set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# now count the number of available GPUs
print(torch.cuda.device_count())

#%%
sample_rate = 48_000
MAX_AUDIO_DURATION = 10
#%%
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)
synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)
#%%
model = transformers.AutoModelForCausalLM.from_pretrained("lucacasini/metamidipianophi3", trust_remote_code=True, torch_dtype="auto")
tokenizer = miditok.REMI.from_pretrained("lucacasini/metamidipianophi3")
OUTPUT_DIR = "artefacts/pianophi-kl=0.2-prompted-longer-training"
PIANO_PROGRAM = 0

# write tokenizer vocab to file
import json
with open(f"piano_vocab.json", "w") as f:
    json.dump(tokenizer.vocab, f, indent=4)

test_midi_path = "artefacts/pianophi/fs_renders/0/reward=20.163776755332947_5.mid"


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

def gen():
    for i in range(1_000_000):
        yield {"prompt": [bar_token, 
                        random.choice(timesignature_tokens), 
                        position_zero_token, random.choice(tempo_tokens), 
                        random.choice(pitch_tokens), 
                        random.choice(velocity_tokens)]
            }
ds = Dataset.from_generator(lambda x: gen(1_000_000))

#%%

aes_predictor = initialize_predictor()

SAVE_INTERVAL = 10
reward_step = 0

def get_aes_scores(records):
    # prepare inputs. 
    predictor_inputs = [{"path": torch.tensor(record["audio"]).float(), "sample_rate": sample_rate, "idx":i} for i, record in enumerate(records) if record["audio"] is not None]
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

def aes_reward(completions, **kwargs):
    # print input arguments

    prompts = torch.tensor(kwargs["prompts"])

    full_seqs = torch.cat([prompts, completions.cpu()], dim=1)

    sms = [tokenizer(full_seqs[i].cpu().numpy()[None,...] ) for i in range(full_seqs.shape[0])]
    # set piano program
    sms = [set_piano_program(sm) for
        sm
        in sms
    ]
    records = [{"completion": full_seqs[i], "sm": sms[i]} for i in range(full_seqs.shape[0])]
    for record in records:
        try:
            record["audio"] = synth.render(record["sm"])
            # if audio is too long, crop it
            if record["audio"].shape[1] > MAX_AUDIO_DURATION * sample_rate:
                record["audio"] = record["audio"][:,:MAX_AUDIO_DURATION * sample_rate]
        except Exception as e:
            print(f"Error rendering audio: {e}")
            record["audio"] = None
    
    records = get_aes_scores(records)
    # take mean of CE, CU, PC, PQ
    rewards = [sum([record["aes_scores"]["CE"], record["aes_scores"]["CU"], record["aes_scores"]["PC"], record["aes_scores"]["PQ"]]) if record["aes_scores"] is not None else 0 for record in records]
    # rewards = [score["CE"] for score in scores]
    # audio_embed = get_clap_features(audio)
    print(f"average reward: {sum(rewards)/len(rewards)}")
    print(f"Rewards: {rewards}")
    global reward_step
    print(f"reward step: {reward_step}")
    if reward_step % SAVE_INTERVAL == 0:
        os.makedirs(f"{OUTPUT_DIR}/fs_renders/{reward_step}", exist_ok=True)
        for i in range(len(sms)):
            sms[i].dump_midi(f"{OUTPUT_DIR}/fs_renders/{reward_step}/reward={rewards[i]}_{i}.mid")
            try:
                dump_wav( f"{OUTPUT_DIR}/fs_renders/{reward_step}/reward={rewards[i]}_{i}.wav", records[i]["audio"], sample_rate, use_int16=True)
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

    def encode(self, x, **kwargs):
        # return {"input_ids": torch.tensor([x])}
        # return {"input_ids": torch.tensor([[1]])}
        return None

    def decode(self, x, **kwargs):
        # print(f"Called decode with {x}")
        return x
    def batch_decode(self, x, **kwargs):
        # print(f"Called batch_decode with {x}")
        return x
    
    def save_pretrained(self, path):
        print(f"Calling save_pretrained with {path} (does nothing)")

    def __call__(self, x, **kwargs):
        # print(f"Called __call__ with {x}")
        n_samples = len(x)
        input_ids = torch.tensor(x)
        attention_mask = torch.ones_like(input_ids)
        print(f"input_ids: {input_ids.shape}")
        print(f"attention_mask: {attention_mask.shape}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        # return {
        #         "input_ids": torch.tensor([[1] for sample in range(n_samples)]), 
        #         "attention_mask":torch.tensor([[1] for sample in range(n_samples)]),
        #     }
dummy_tokenizer = DummyTokenizer(tokenizer)

#%%
# look at effect of sampling params on rewards
# get a batch of 64 samples fast to test the model
# get one batch
# tst_batch = next(iter(ds.batch(64)))

# # generate with model
# outputs = model.generate(
#     input_ids=torch.tensor(batch["prompt"]),
#     max_length=250,
#     # num_return_sequences=64,
#     temperature=1.0,
#     output_scores=True,
#     output_hidden_states=True,
#     output_attentions=True,
# )
# rewards = aes_reward(outputs, prompts=batch["prompt"])

#%%
# get rewards

# %%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

BATCH_SIZE=64
N_GENERATIONS=100

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=250,
    max_prompt_length=6,
    num_train_epochs=10_000,
    learning_rate=1e-5,
    report_to="wandb",
    logging_steps=1,
    num_generations=BATCH_SIZE,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    beta=0.2
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=aes_reward,
    args =  config,
    train_dataset=ds,
    processing_class=dummy_tokenizer,

)
# save model
trainer.save_model()
trainer.train()
#%%
