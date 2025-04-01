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
from symusic import dump_wav
import random
from processors import RewardManager, MidiTokToSymusicProcessor, SymusicSynthProcessor, TinySoundfontSynthProcessor, AudioBoxAesRewardProcessor, ProgramPromptAdherenceRewardProcessor, CLAPPromptRewardProcessor, CLAPZeroShotClassificationRewardProcessor
from loops_util import prepare_input
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
    # "CE": 1.0,
    # "CU": 1.0,
    # "PC": 0.0,
    # "PQ": 1.0,
    # "programs_iou": 3.0,
    "programs_iou": 1.0,
    "clap_clf":1.0,
    # "clap":20.0
}
TEMPERATURE = 1.0
NUM_ITERATIONS = 1
SCALE_REWARDS = True

NUM_TRAIN_STEPS = 100
LEARNING_RATE = 1e-4
SEARCH_SAMPLING_PARAMS = False

BETA = 0.04

# MODEL = "piano" #"MIL"
# PROMPT_SOURCE = "procedural" #"dataset" # "dataset" "no_prompt", "procedural", "piano"
MODEL = "MIL"
PROMPT_SOURCE = "dataset" #"dataset" # "dataset" "no_prompt", "procedural", "piano"
AUDIO_SAVE_INTERVAL = NUM_ITERATIONS*10

N_PROMPTS = (NUM_TRAIN_STEPS * BATCH_SIZE // NUM_GENERATIONS) * 10

SAMPLE_RATE = 48_000
SOUNDFONT = "matrix" if MODEL == "MIL" else "yamaha"

# get latest checkpoint
OUTPUT_DIR = "artefacts/mil-clap-iou-ds"

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
        "gba":"soundfonts/General_Game_Boy_Advance_Soundfont.sf2",
        "candy":"soundfonts/Candy_Set_Full_GM.sf2",
        "808": "soundfonts/General808.sf2",
        "grandeur": "soundfonts/[GD] The Grandeur D.sf2",
        "ydp": "soundfonts/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2",
        "yamaha": "soundfonts/Yamaha-C5-Salamander-JNv5_1.sf2",
        "matrix": "soundfonts/MatrixSF_v2.1.5.sf2"
            }[SOUNDFONT]

match MODEL:
    case "piano":
        MAX_COMPLETION_LENGTH = 512
        MAX_BEATS = 64
        MAX_AUDIO_DURATION = 10

        # BASE_MODEL_PATH = "lucacasini/metamidipianophi3"
        BASE_MODEL_PATH = "lucacasini/metamidipianophi3_6L"
        model = transformers.AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype="auto")
        tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
        bar_token = tokenizer.vocab["Bar_None"]
        position_zero_token = tokenizer.vocab["Position_0"]
        timesignature_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("TimeSig_")]
        tempo_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Tempo_")]
        pitch_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Pitch_")]
        velocity_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Velocity_")]
        duration_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Duration_")]

        print(f"Found {len(timesignature_tokens)} time signature tokens")
        print(f"Found {len(tempo_tokens)} tempo tokens")
        print(f"Found {len(pitch_tokens)} pitch tokens")
        print(f"Found {len(velocity_tokens)} velocity tokens")

        match PROMPT_SOURCE:
            case "dataset":
                print("Loading dataset")
                trn_ds = Dataset.load_from_disk("data/dataset_mmd_piano/train")
                print("Dataset loaded")
                # print length of dataset
                # take random subset of 1000
                print("Taking random subset")
                trn_ds = trn_ds.shuffle()
                # take random subset
                trn_ds = trn_ds.select(range(N_PROMPTS))
                # trn_ds = trn_ds.filter(lambda x: len(x["token_ids"]) <= MAX_COMPLETION_LENGTH)

                max_prompt_length = 8

                def extract_prompt(token_ids):
                    tokens = tokenizer._ids_to_tokens(token_ids)
                    print(tokens)
                    # find index of first token with tempo in it
                    first_tempo_idx = next((i for i, token in enumerate(tokens) if token.startswith("Tempo_")), None)
                    # crop prompt up to tempo token
                    prompt = token_ids[:first_tempo_idx+1]
                    # pad left with PAD tokens until max_prompt_length
                    prompt =[tokenizer.vocab["PAD_None"]] * (max_prompt_length - len(prompt)) + prompt
                    return prompt
                
                # when using dataset as prompt, we need to prepare the input
                trn_ds = trn_ds.map(lambda x: {"prompt": extract_prompt(prepare_input(x["tokens"], tokenizer))})
                print("Dataset loaded")
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
                max_prompt_length = len(trn_ds[0]["prompt"])
            case _:
                raise ValueError("Invalid prompt source for piano model")
            
        reward_manager = RewardManager(
            processors = [
                MidiTokToSymusicProcessor(tokenizer, is_multitrack=False, max_beats=100),
                # TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION),
                TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION),
                # SymusicSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION),
                AudioBoxAesRewardProcessor(),
                CLAPZeroShotClassificationRewardProcessor(sample_rate=SAMPLE_RATE, reference_prompts=["dissonant, low quality, caucophonous music, glitch, midi"], target_prompt="beautiful, high quality, amazing music, natural, calming", temperature=0.25),
                # CLAPPromptRewardProcessor(sample_rate=SAMPLE_RATE, target_prompt="jazzy jazz piano solo", k=NUM_GENERATIONS),
                # CLAPPromptRewardProcessor(sample_rate=SAMPLE_RATE, target_prompt="jazz", k=NUM_GENERATIONS),
                # CLAPPromptSoftmaxRewardProcessor(sample_rate=SAMPLE_RATE, target_prompt="jazz", temperature=1.0),
            ],
            reward_weights = REWARD_WEIGHTS,
            output_dir=OUTPUT_DIR
        )

    case "MIL":
        MAX_COMPLETION_LENGTH = 2048
        MAX_BEATS = 16
        MAX_AUDIO_DURATION = 24

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
        program_tokens = [value for key, value in tokenizer.vocab.items() if key.startswith("Program_")]

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
                        yield {"prompt": [tokenizer.vocab["BOS_None"], random.choice(program_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "piano":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_0"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "bass_drums_keys":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_5"],tokenizer.vocab["Program_36"], tokenizer.vocab["Program_-1"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "flbass_drums_keys":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_4"],tokenizer.vocab["Program_35"], tokenizer.vocab["Program_-1"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "bass_drums_keys_gtr":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_5"],tokenizer.vocab["Program_36"], tokenizer.vocab["Program_-1"],tokenizer.vocab["Program_28"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
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

        reward_manager = RewardManager(
            processors = [
                MidiTokToSymusicProcessor(tokenizer, is_multitrack=True, max_beats=MAX_BEATS),
                TinySoundfontSynthProcessor(SF_PATH, SAMPLE_RATE, MAX_AUDIO_DURATION),
                AudioBoxAesRewardProcessor(),
                CLAPZeroShotClassificationRewardProcessor(sample_rate=SAMPLE_RATE, reference_prompts=["dissonant, low quality, caucophonous music, glitch, midi"], target_prompt="groovy, amazing, natural, high quality, studio, live", temperature=0.25),
                # CLAPZeroShotClassificationRewardProcessor(sample_rate=SAMPLE_RATE, reference_prompts=["dissonant, low quality, caucophonous music, glitch, midi"], target_prompt="beautiful, high quality, amazing music, natural, calming", temperature=0.25),
                ProgramPromptAdherenceRewardProcessor(),
            ],
            reward_weights = REWARD_WEIGHTS,
            output_dir=OUTPUT_DIR
        )

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
        input_ids = torch.tensor( kwargs["text"])
        attention_mask = torch.where(
            input_ids != self.pad_token_id, 1, 0
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
dummy_tokenizer = DummyTokenizer(tokenizer)


if SEARCH_SAMPLING_PARAMS:
    # first do a grid search over temperature
    test_temperatures = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0, 1.1]

    # get a batch
    test_batch = trn_ds[:64]

    # rename prompt to text
    test_batch["text"] = test_batch["prompt"]

    # load model on gpu
    model.to("cuda")


    all_records = []
    for test_temperature in test_temperatures:
        outputs = model.generate(
                        input_ids=dummy_tokenizer(**test_batch)["input_ids"].to("cuda"),
                        do_sample=True,
                        temperature=test_temperature,
                        max_length=MAX_COMPLETION_LENGTH,
                        pad_token_id=dummy_tokenizer.pad_token_id,
                        eos_token_id=dummy_tokenizer.eos_token_id,
                        bos_token_id=dummy_tokenizer.bos_token_id,
        )

        records = reward_manager(completions = torch.tensor(outputs).cpu(), prompts = test_batch["prompt"],return_records=True)
        # get rewards
        # add temperature to records
        records = [{**r, "temperature": test_temperature} for r in records]
        # concatenate records into big list
        all_records.extend(records)


    # print average rewards per temperature
    import pandas as pd
    from IPython.display import display

    # print keys in first record
    logs = pd.DataFrame.from_records(all_records)

    # print columns
    for col in logs.columns:
        print(col)

    for col in ["normalized_rewards"]:
        if logs[col].apply(lambda x: isinstance(x, dict)).all():
            logs = pd.concat([logs, logs[col].apply(pd.Series).add_prefix(col + "_")], axis=1)
            logs.drop(col, axis=1, inplace=True)


    # print mean of normalized rewards per temperature and reward
    for temp in test_temperatures:
        print(f"Temperature: {temp}")
        # get normalized rewards for this temperature
        temp_logs = logs[logs["temperature"] == temp]
        # print mean of normalized rewards
        # for everything that starts with normalized_rewards_
        for col in temp_logs.columns:
            if col.startswith("normalized_rewards_"):
                print(f"{col}: {temp_logs[col].mean()}")
        print("\n\n")

    # save logs to parquet
    logs_ds = Dataset.from_pandas(logs)
    logs_ds.save_to_disk(f"artefacts/temperature_search_logs_{MODEL}_{PROMPT_SOURCE}")

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
    reward_funcs=reward_manager,
    args =  config,
    train_dataset=trn_ds,
    processing_class=dummy_tokenizer,

)
# save model
trainer.train()
trainer.save_model()
# %%