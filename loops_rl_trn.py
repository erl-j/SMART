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
import random
from render import TinySoundfontRenderer, SymusicRenderer

#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

# set SEED 
SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS = 1
USE_BF16 = True
NUM_GENERATIONS=8
REWARD_WEIGHTS = {
    "CE": 1.0,
    "CU": 0.0,
    "PC": 0.0,
    "PQ": 0.0,
    "programs_iou": 0.0,
}
TEMPERATURE = 0.8
NUM_ITERATIONS = 1
SCALE_REWARDS = True

NUM_TRAIN_STEPS = 1000
LEARNING_RATE = 1e-4
BETA = 0.04

# MODEL = "piano" #"MIL"
# PROMPT_SOURCE = "procedural" #"dataset" # "dataset" "no_prompt", "procedural", "piano"
MODEL = "MIL"
PROMPT_SOURCE = "dataset"
AUDIO_SAVE_INTERVAL = NUM_ITERATIONS*10



N_PROMPTS = (NUM_TRAIN_STEPS * BATCH_SIZE // NUM_GENERATIONS) * 10


SAMPLE_RATE = 48_000
SOUNDFONT = "musescore" 

# get latest checkpoint
OUTPUT_DIR = "artefacts/loops-1e-4-beta=0.04-t=0.8"

SF_PATH= {
        "musescore": str(BuiltInSF3.MuseScoreGeneral().path(download=True)), 
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

match MODEL:
    case "piano":
        MAX_COMPLETION_LENGTH = 256
        MAX_BEATS = None
        MAX_AUDIO_DURATION = 16


        synth = SymusicRenderer(SF_PATH, SAMPLE_RATE)
        BASE_MODEL_PATH = "lucacasini/metamidipianophi3"
        model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
        tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
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

        match PROMPT_SOURCE:
            case "procedural":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [bar_token, 
                                        random.choice(timesignature_tokens), 
                                        position_zero_token, 
                                        random.choice(tempo_tokens), 
                                        random.choice(pitch_tokens), 
                                        random.choice(velocity_tokens)]
                            }
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case _:
                raise ValueError("Invalid prompt source for piano model")
    case "MIL":
        MAX_COMPLETION_LENGTH = 2048
        MAX_BEATS = 16
        MAX_AUDIO_DURATION = 32

        synth = TinySoundfontRenderer(SF_PATH, SAMPLE_RATE)
        BASE_MODEL_PATH = "/workspace/aestune/outputs/mt/treasured-cosmos-19/checkpoint-325000"
        TOKENIZER_CONFIG_PATH = "data/tokenizer_config.json"
        model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
        tokenizer_config = miditok.TokenizerConfig.load_from_json(TOKENIZER_CONFIG_PATH)
        tokenizer = miditok.REMI(tokenizer_config)
        bar_token = tokenizer.vocab["Bar_None"]
        position_zero_token = tokenizer.vocab["Position_0"]
        timesignature_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("TimeSig_")]
        tempo_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Tempo_")]
        pitch_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Pitch_")]
        velocity_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Velocity_")]

        match PROMPT_SOURCE:
            case "no prompt":
                print(f"Found {len(timesignature_tokens)} time signature tokens")
                print(f"Found {len(tempo_tokens)} tempo tokens")
                print(f"Found {len(pitch_tokens)} pitch tokens")
                print(f"Found {len(velocity_tokens)} velocity tokens")


                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"] #, random.choice(program_tokens)
                                        ]
                            }
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "procedural":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_-1"]]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "piano":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_0"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "dataset":
                print("Loading dataset")
                trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")
                print("Dataset loaded")
                # print length of dataset
                # take random subset of 1000
                print("Taking random subset")
                trn_ds = trn_ds.shuffle()
                # take random subset
                trn_ds = trn_ds.select(range(N_PROMPTS))
                trn_ds = trn_ds.filter(lambda x: len(x["token_ids"]) <= MAX_COMPLETION_LENGTH)

                max_prompt_length = 32

                def extract_prompt(token_ids):
                    tokens = tokenizer._ids_to_tokens(token_ids)
                    # find index of first token with tempo in it
                    first_tempo_idx = next((i for i, token in enumerate(tokens) if token.startswith("Tempo_")), None)
                    # crop prompt up to tempo token
                    prompt = token_ids[:first_tempo_idx+1]
                    # pad left with PAD tokens until max_prompt_length
                    prompt =[tokenizer.vocab["PAD_None"]] * (max_prompt_length - len(prompt)) + prompt
                    return prompt
                
                # when using dataset as prompt, we need to prepare the input
                trn_ds = trn_ds.map(lambda x: {"prompt": extract_prompt(prepare_input(x["token_ids"], tokenizer))})
                print("Dataset loaded")
            case _:
                raise ValueError("Invalid prompt source for MIL model")

        # print a 5 random prompts
        for i in range(5):
            print(tokenizer._ids_to_tokens(trn_ds[i]["prompt"]))
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
            "duration": record["audio"].shape[1] / SAMPLE_RATE,
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
    
    # if prompts.dim() < 3:
    #     prompts = prompts[:, None, :]
    #     print(f"full_seqs shape: {prompts.shape}")

    # if completions.dim() < 3:
    #     completions = completions[:, None, :]
    #     print(f"full_seqs shape: {completions.shape}")
    # Combine prompts and completions
    full_seqs = torch.cat([prompts, completions.cpu()], dim=1)

    # if has less than 2 dimensions, add a dimension in the middle


    if MODEL == "MIL":
        # Process sequences into structured music objects
        sms = [tokenizer(full_seqs[i].cpu().numpy()) for i in range(full_seqs.shape[0])]
        sms = [crop_sm(sm, MAX_BEATS) for sm in sms]

    if MODEL == "piano":
        sms = [tokenizer(full_seqs[i].cpu().numpy()[None, ...]) for i in range(full_seqs.shape[0])]

    # print sm scores
    for i, sm in enumerate(sms):
        print(f"SM {i} score: {sm}")
    
    # Create records for each sequence
    records = [
        {   
            "completion": completions[i].cpu(),
            "sm": sms[i],
            "prompt": prompts[i],
            "prompt_and_completion": full_seqs[i],
            "completion_tokens": tokenizer._ids_to_tokens(completions[i].tolist()),
            "prompt_tokens": tokenizer._ids_to_tokens(prompts[i].tolist()),
            "prompt_and_completion_tokens": tokenizer._ids_to_tokens(full_seqs[i].tolist()),
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
        try:
            record["intersection_over_union_programs"] = len(record["prompt_programs"].intersection(record["completion_programs"])) / len(record["prompt_programs"].union(record["completion_programs"]))
            record["normalized_rewards"]["programs_iou"] = record["intersection_over_union_programs"]
        except ZeroDivisionError:
            print(f"Couldnt compute program intersection over union with prompt programs {record['prompt_programs']} and completion programs {record['completion_programs']}")

    print(f"Processing {len(records)} records")
    
    # Render audio for each record
    for record in tqdm(records):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mid") as f:
                record["sm"].dump_midi(f.name)
                if MODEL == "MIL":
                    n_beats = MAX_BEATS
                    record["audio"] = synth.render(f.name, n_beats  * sm_beats_per_second(record["sm"]))
                if MODEL == "piano":
                    record["audio"] = synth.render(f.name, MAX_AUDIO_DURATION)
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
    
    print(f"Saving logs and audio for {len(records)} records")
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
    log_ds = Dataset.from_list(logs)
    log_ds.to_parquet(f"{OUTPUT_DIR}/rl_logs/{global_reward_step}/logs.parquet")
    
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
    print(f"Done saving logs and audio for {len(records)} records")
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
    def __call__(self,  **kwargs):
        # print(f"Calling __call__ with {kwargs}")
        x = kwargs["text"]
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
    save_steps=NUM_TRAIN_STEPS,
    beta=BETA,
    bf16=USE_BF16,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    # set schedule to fixed
    # lr_scheduler_type
)
trainer = GRPOTrainer(

    model=model,
    reward_funcs=aes_reward,
    args =  config,
    train_dataset=trn_ds,
    processing_class=dummy_tokenizer,

)
# save model
trainer.train()
trainer.save_model()
# %%