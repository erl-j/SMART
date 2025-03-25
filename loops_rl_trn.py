#%%
import miditok
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from symusic import BuiltInSF3
from datasets import Dataset, load_dataset
from audiobox_aesthetics.infer import initialize_predictor
import torch.nn.functional as F
import os
from util import crop_sm, sm_beats_per_second
import tempfile
from tqdm import tqdm
import numpy as np
import glob
from loops_util import prepare_input
from symusic import dump_wav
#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

# set SEED 
SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE=32
USE_BF16 = True
NUM_GENERATIONS=32
REWARD_WEIGHTS = {
    "CE": 0.25,
    "CU": 0.25,
    "PC": 0.25,
    "PQ": 0.25,
    "programs_iou": 1.0,
}
TEMPERATURE = 1.0
MAX_PROMPT_LENGTH = 32

NUM_TRAIN_STEPS = 1000
LEARNING_RATE = 1e-5
BETA = 0.01
PROMPT_SOURCE = "piano" #"dataset" # "dataset" "no_prompt", "procedural", "piano"

BASE_MODEL_PATH = "/workspace/aestune/outputs/mt/treasured-cosmos-19/"
MAX_COMPLETION_LENGTH = 2048

# BASE_MODEL_PATH = "/workspace/aestune/outputs/mt/ruby-microwave-20/"
# MAX_COMPLETION_LENGTH = 1024

N_PROMPTS = (NUM_TRAIN_STEPS * BATCH_SIZE // NUM_GENERATIONS) * 10

# get latest checkpoint
BASE_MODEL_PATH = sorted(glob.glob(f"{BASE_MODEL_PATH}/checkpoint-*"))[-1]
print(f"Using checkpoint {BASE_MODEL_PATH}")
TOKENIZER_CONFIG_PATH = "data/tokenizer_config.json"

OUTPUT_DIR = "artefacts/loops-fluir3-2-iou-logstep-1e-4-beta=0.01-avg-aes-and-iou-32samples-piano-random-tempo"

#%%
# audio rendering settings
SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 32
AUDIO_SAVE_INTERVAL = 50
SOUNDFONT = "fluidr3" 

SF_PATH= {"musescore": BuiltInSF3.MuseScoreGeneral().path(download=True), 
            "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
            "monalisa":"./soundfonts/Monalisa_GM_v2_105.sf2",
            "ephesus":"./soundfonts/Ephesus_GM_Version_1_00.sf2",
            "touhou" : "./soundfonts/Touhou.sf2",
            "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
            "fluidr3": "./soundfonts/FluidR3 GM.sf2",
            "goldeneye":"./soundfonts/GoldenEye_007.sf2",
            "ronaldiho":"./soundfonts/InternationalSuperStarSoccer.sf2",
            "casio":"./soundfonts/Casio_CTK-230_GM.sf2",
            "n64":"soundfonts/General_MIDI_64_1.6.sf2",
            }[SOUNDFONT]
from render import MidiRenderer

synth = MidiRenderer(SF_PATH, SAMPLE_RATE)
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
program_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Program_")]

if PROMPT_SOURCE == "no prompt":

    print(f"Found {len(timesignature_tokens)} time signature tokens")
    print(f"Found {len(tempo_tokens)} tempo tokens")
    print(f"Found {len(pitch_tokens)} pitch tokens")
    print(f"Found {len(velocity_tokens)} velocity tokens")

    def gen():
        for i in range(N_PROMPTS):
            yield {"prompt": [tokenizer.vocab["BOS_None"] #, random.choice(program_tokens)
                            ]
                }
    ds = Dataset.from_generator(gen)

elif PROMPT_SOURCE == "procedural":
    def gen():
        for i in range(N_PROMPTS):
            yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_-1"]]}
    ds = Dataset.from_generator(gen)
elif PROMPT_SOURCE == "piano":
    def gen():
        for i in range(N_PROMPTS):
            yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_0"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
    ds = Dataset.from_generator(gen)


elif PROMPT_SOURCE == "dataset":
    print("Loading dataset")
    trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")
    print("Dataset loaded")
    # print length of dataset
    # take random subset of 1000
    print("Taking random subset")
    trn_ds = trn_ds.shuffle()
    # take random subset
    trn_ds = trn_ds.select(range(N_PROMPTS))

    # filter out where token ids is larger than completion length
    trn_ds = trn_ds.filter(lambda x: len(x["token_ids"]) <= MAX_COMPLETION_LENGTH)


    def extract_prompt(token_ids):
        tokens = tokenizer._ids_to_tokens(token_ids)
        # find index of first token with tempo in it
        first_tempo_idx = next((i for i, token in enumerate(tokens) if token.startswith("Tempo_")), None)
        # crop prompt up to tempo token
        prompt = token_ids[:first_tempo_idx+1]
        # pad left with PAD tokens until MAX_PROMPT_LENGTH
        prompt =[tokenizer.vocab["PAD_None"]] * (MAX_PROMPT_LENGTH - len(prompt)) + prompt
        return prompt
    
    # when using dataset as prompt, we need to prepare the input
    ds = trn_ds.map(lambda x: {"prompt": extract_prompt(prepare_input(x["token_ids"], tokenizer))})
    print("Dataset loaded")

    # print a 5 random prompts
    for i in range(5):
        print(tokenizer._ids_to_tokens(ds[i]["prompt"]))



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
    
    # Process sequences into structured music objects
    sms = [tokenizer(full_seqs[i].cpu().numpy()) for i in range(full_seqs.shape[0])]

    n_beats=16

    sms = [crop_sm(sm, n_beats) for sm in sms]
    
    # Create records for each sequence
    records = [
        {   
            "completion": completions[i].cpu(),
            "sm": sms[i],
            "prompt": prompts[i],
            "prompt_and_completion": full_seqs[i],
            "completion_tokens": tokenizer._ids_to_tokens(completions[i].cpu().numpy()),
            "prompt_tokens": tokenizer._ids_to_tokens(prompts[i].cpu().numpy()),
            "prompt_and_completion_tokens": tokenizer._ids_to_tokens(full_seqs[i].cpu().numpy()),
            "idx": i,
            "normalized_rewards": {},
            "reward_step": global_reward_step,
        } 
        for i in range(full_seqs.shape[0])
    ]

    # compute prompt programs
    for record in records:
        record["prompt_programs"] = set([token for token in record["prompt_tokens"] if token.startswith("Program_")])
        record["completion_programs"] = set([token for token in record["completion_tokens"] if token.startswith("Program_")])
        record["intersection_over_union_programs"] = len(record["prompt_programs"].intersection(record["completion_programs"])) / len(record["prompt_programs"].union(record["completion_programs"]))
        record["normalized_rewards"]["programs_iou"] = record["intersection_over_union_programs"]

    print(f"Processing {len(records)} records")
    
    # Render audio for each record
    for record in tqdm(records):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mid") as f:
                record["sm"].dump_midi(f.name)
                record["audio"] = synth.render(f.name, n_beats  * sm_beats_per_second(record["sm"]))
                # peak normalization
                record["audio"] = record["audio"] / np.max(np.abs(record["audio"])+1e-6)
            if record["audio"].shape[1] > MAX_AUDIO_DURATION * SAMPLE_RATE:
                record["audio"] = record["audio"][:, :MAX_AUDIO_DURATION * SAMPLE_RATE]
        except Exception as e:
            print(f"Error rendering audio: {e}")
            record["audio"] = None

    print("Audio rendered")
    print(f"Calculating aesthetic scores for {len(records)} records")
    # Get aesthetic scores for each record
    records = get_aes_scores(records)

    # add aes scores to raw_rewards
    for record in records:
        if record["aes_scores"] is not None:
            record["normalized_rewards"]["CE"] = record["aes_scores"]["CE"] / 10
            record["normalized_rewards"]["CU"] = record["aes_scores"]["CU"] / 10
            record["normalized_rewards"]["PC"] = record["aes_scores"]["PC"] / 10
            record["normalized_rewards"]["PQ"] = record["aes_scores"]["PQ"] / 10

    # get weighted reward normalized by total weight
    for record in records:
        record["reward"] = sum([record["normalized_rewards"].get(key, 0) * REWARD_WEIGHTS[key] for key in REWARD_WEIGHTS.keys()]) / sum(REWARD_WEIGHTS.values())
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
        input_ids = torch.tensor(x)
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
    temperature=TEMPERATURE,
    output_dir=OUTPUT_DIR,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    report_to="wandb",
    logging_steps=1,
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=NUM_TRAIN_STEPS,
    beta=BETA,
    bf16=USE_BF16,
    # set schedule to fixed
    lr_scheduler_type="constant",
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