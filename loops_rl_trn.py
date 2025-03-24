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
# set CUDA DEVICE to 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # print how many GPUs are available
# print(torch.cuda.device_count())
#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"
BATCH_SIZE=32
NUM_GENERATIONS=32
REWARD_WEIGHTS = {
    "CE": 1.0,
    "CU": 1.0,
    "PC": 1.0,
    "PQ": 1.0
}
TEMPERATURE = 0.9
MAX_COMPLETION_LENGTH = 2048
NUM_TRAIN_EPOCHS = 200
LEARNING_RATE = 1e-5
BETA = 0.04

BASE_MODEL_PATH = "outputs/mt/treasured-cosmos-19/checkpoint-75000"
TOKENIZER_CONFIG_PATH = "data/tokenizer_config.json"

OUTPUT_DIR = "artefacts/loops-fluidr3"

#%%
# audio rendering settings
SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 32
AUDIO_SAVE_INTERVAL = 10
SOUNDFONT = "fluidr3" 

SF_PATH= {"musescore": BuiltInSF3.MuseScoreGeneral().path(download=True), 
            "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
            "monalisa":"./soundfonts/Monalisa_GM_v2_105.sf2",
            "ephesus":"./soundfonts/Ephesus_GM_Version_1_00.sf2",
            "touhou" : "./soundfonts/Touhou.sf2",
            "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
            "fluidr3": "./soundfonts/FluidR3 GM.sf2",

            }[SOUNDFONT]

synth = Synthesizer(
    sf_path = SF_PATH, # the path to the soundfont
    sample_rate = SAMPLE_RATE, # the sample rate of the output wave, sample_rate is the default value
)
#%%

model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
tokenizer_config = miditok.TokenizerConfig.load_from_json(TOKENIZER_CONFIG_PATH)
tokenizer = miditok.REMI(tokenizer_config)

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
    for i in range(100_000):
        yield {"prompt": [tokenizer.vocab["BOS_None"]]
            }
ds = Dataset.from_generator(gen)
prompt_length = len(ds[0]["prompt"])
#%%
aes_predictor = initialize_predictor()
def get_aes_scores(records):
    """
    Calculate aesthetic scores for records that have audio data.
    
    Args:
        records: List of record dictionaries containing audio data
        
    Returns:
        Updated records with aesthetic scores added
    """
    # Prepare inputs for predictor (only for records with valid audio)
    predictor_inputs = [
        {
            "path": torch.tensor(record["audio"]).float(), 
            "sample_rate": SAMPLE_RATE, 
            "idx": i
        } 
        for i, record in enumerate(records) 
        if record["audio"] is not None
    ]
    
    # Get scores from aesthetic predictor
    scores = aes_predictor.forward(predictor_inputs)
    
    # Map scores back to original records
    record_with_audio_index = 0
    for record in records:
        if record["audio"] is not None:
            record["aes_scores"] = scores[record_with_audio_index]
            record_with_audio_index += 1
        else:
            record["aes_scores"] = None
            
    return records

global_reward_step = 0

def aes_reward(completions, return_records=False, **kwargs):
    """
    Calculate aesthetic rewards for audio sequences generated from completions.
    
    Args:
        completions: Tensor of completion sequences
        return_records: If True, return full records instead of just rewards
        **kwargs: Additional arguments, must contain 'prompts'
    
    Returns:
        Either a list of rewards or a list of complete record dictionaries
    """
    global global_reward_step
    
    # Extract prompts from kwargs
    prompts = torch.tensor(kwargs["prompts"])
    
    # Combine prompts and completions
    full_seqs = torch.cat([prompts, completions.cpu()], dim=1)

    print(full_seqs.shape)
    
    # Process sequences into structured music objects
    sms = [tokenizer(full_seqs[i].cpu().numpy()) for i in range(full_seqs.shape[0])]
    
    # Create records for each sequence
    records = [
        {
            "completion": completions[i].cpu(),
            "sm": sms[i],
            "prompt": prompts[i],
            "prompt_and_completion": full_seqs[i],
            "idx": i,
            "raw_rewards": {},
        } 
        for i in range(full_seqs.shape[0])
    ]
    
    # Render audio for each record
    for record in records:
        try:
            record["audio"] = synth.render(record["sm"])
            if record["audio"].shape[1] > MAX_AUDIO_DURATION * SAMPLE_RATE:
                record["audio"] = record["audio"][:, :MAX_AUDIO_DURATION * SAMPLE_RATE]
        except Exception as e:
            print(f"Error rendering audio: {e}")
            record["audio"] = None
    
    # Get aesthetic scores for each record
    records = get_aes_scores(records)

    # add aes scores to raw_rewards
    for record in records:
        if record["aes_scores"] is not None:
            record["raw_rewards"]["CE"] = record["aes_scores"]["CE"]
            record["raw_rewards"]["CU"] = record["aes_scores"]["CU"]
            record["raw_rewards"]["PC"] = record["aes_scores"]["PC"]
            record["raw_rewards"]["PQ"] = record["aes_scores"]["PQ"]

    # get weighted reward normalized by total weight
    for record in records:
        record["reward"] = sum([record["raw_rewards"].get(key, 0) * REWARD_WEIGHTS[key] for key in REWARD_WEIGHTS.keys()]) / sum(REWARD_WEIGHTS.values())
        record["reward_weights"] = REWARD_WEIGHTS
    # Return records if requested
    if return_records:
        return records
    
    # Otherwise process rewards and save data
    rewards = [record["reward"] for record in records]
    print(f"average reward: {sum(rewards)/len(rewards)}")
    print(f"Rewards: {rewards}")
    print(f"reward step: {global_reward_step}")
    
    # Prepare logs (exclude audio and sm fields)
    logs = []
    dont_log = ["audio", "sm"]
    for record in records:
        log = {**record}
        for key in dont_log:
            log.pop(key)
        logs.append(log)
    
    # Save logs as parquet
    os.makedirs(f"{OUTPUT_DIR}/rl_logs/{global_reward_step}", exist_ok=True)
    ds = Dataset.from_list(logs)
    ds.to_parquet(f"{OUTPUT_DIR}/rl_logs/{global_reward_step}/logs.parquet")
    
    # Save MIDI files
    os.makedirs(f"{OUTPUT_DIR}/midi/{global_reward_step}", exist_ok=True)
    for i in range(len(sms)):
        sms[i].dump_midi(f"{OUTPUT_DIR}/midi/{global_reward_step}/reward={rewards[i]}_{i}.mid")
    
    # Save audio files periodically
    if global_reward_step % AUDIO_SAVE_INTERVAL == 0:
        os.makedirs(f"{OUTPUT_DIR}/audio/{global_reward_step}", exist_ok=True)
        for i in range(len(sms)):
            try:
                dump_wav(
                    f"{OUTPUT_DIR}/audio/{global_reward_step}/reward={rewards[i]}_{i}.wav", 
                    records[i]["audio"], 
                    SAMPLE_RATE, 
                    use_int16=True
                )
            except Exception as e:
                print(f"Error dumping wav: {e}")
    
    # Increment reward step
    global_reward_step += 1
    
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
    temperature=TEMPERATURE,
    output_dir=OUTPUT_DIR,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=prompt_length,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    report_to="wandb",
    logging_steps=1,
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=NUM_TRAIN_EPOCHS,
    beta=BETA,
    bf16=True,
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
# %%