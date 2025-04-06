#%%
import miditok
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from symusic import BuiltInSF3
from datasets import Dataset, load_dataset
import os
from tqdm import tqdm
import numpy as np
import random
from processors import RewardManager, MidiTokToSymusicProcessor, TinySoundfontSynthProcessor, AudioBoxAesRewardProcessor, ProgramPromptAdherenceRewardProcessor, PamRewardProcessor
from loops_util import prepare_input
import os
import torch
from tqdm import tqdm
#%%
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

# set SEED 
SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

GRADIENT_ACCUMULATION_STEPS = 1
USE_BF16 = True
NUM_GENERATIONS=8

TEMPERATURE = 1
NUM_ITERATIONS = 1
SCALE_REWARDS = True

NUM_TRAIN_STEPS = 1000
LEARNING_RATE = 1e-4
SEARCH_SAMPLING_PARAMS = False

BETA = 0.16

# MODEL = "piano" #"MIL"
# PROMPT_SOURCE = "procedural" #"dataset" # "dataset" "no_prompt", "procedural", "piano"
MODEL = "mil"
PROMPT_SOURCE = "dataset" #"dataset" # "dataset" "no_prompt", "procedural", "piano"
AUDIO_SAVE_INTERVAL = NUM_ITERATIONS*10
SAVE_STEPS = 20
N_EVAL_PROMPTS=1000

BATCH_SIZE=32 if MODEL == "mil" else 64


N_PROMPTS = (NUM_TRAIN_STEPS * BATCH_SIZE // NUM_GENERATIONS) * 10

SAMPLE_RATE = 48_000
SOUNDFONT = "matrix" if MODEL == "mil" else "yamaha"

REWARD_WEIGHTS = {
    # "CE": 1.0,
    # "CU": 1.0,
    # "PC": 0.0,
    # "PQ": 1.0,
    # "programs_iou": 3.0,
    "programs_iou": 1.0,
    "pam_avg": 1.0,
}

# get latest checkpoint
OUTPUT_DIR = f"artefacts/all_runs_2/{MODEL}-{PROMPT_SOURCE}/aes-{BETA}-{TEMPERATURE}-{NUM_TRAIN_STEPS}"

prompt_pairs = [
    {
        "shorthand": "glitchless",
        "positive": "A smooth and continuous texture with seamless transitions and stable audio playback.",
        "negative": "A glitchy and unstable excerpt with digital artifacts, abrupt cuts, or dropouts."
    },
    {
        "shorthand": "expressive",
        "positive": "A dynamically rich excerpt with emotional phrasing, subtle articulations, and human-like nuance.",
        "negative": "A flat and mechanical sequence with rigid articulation, even dynamics, and static energy."
    },
    {
        "shorthand": "naturalfeel",
        "positive": "A fluid rhythmic feel with organic timing and slight expressive variations in phrasing.",
        "negative": "A stiff and quantized rhythm with robotic timing and uniform note placements."
    },
    {
        "shorthand": "clarity",
        "positive": "A clean mix where each instrument is clearly defined and occupies its own space in the spectrum.",
        "negative": "A muddy and crowded mix where sounds overlap and mask each other."
    },
    {
        "shorthand": "interest",
        "positive": "An engaging texture with subtle variations, rich timbres, and musical detail.",
        "negative": "A dull sequence with static timbres and repetitive gestures that feel uninspired."
    },
    {
        "shorthand": "prosound",
        "positive": "A polished sound with balanced levels, clean frequency distribution, and professional-grade production.",
        "negative": "A rough and uneven mix with harsh tones, poor balance, and lo-fi characteristics."
    },
    {
        "shorthand": "intent",
        "positive": "A focused excerpt with coherent phrasing and purposeful musical gestures.",
        "negative": "An unfocused sequence with scattered gestures and unclear musical direction."
    },
    {
        "shorthand": "groove",
        "positive": "A tight rhythmic feel with strong pulse, clear subdivisions, and natural momentum.",
        "negative": "An awkward rhythmic structure with imprecise timing and unstable pulse."
    },
    {
        "shorthand": "realism",
        "positive": "Instruments that sound lifelike and expressive, with detailed articulation and natural tone.",
        "negative": "Instruments that sound artificial and static, with generic tone and unrealistic behavior."
    },
    {
        "shorthand": "variation",
        "positive": "A rich texture with contrasting articulations, subtle shifts, and sonic variety.",
        "negative": "A uniform texture with repeated gestures and minimal sonic contrast."
    }
]

# warn if output dir exists and may be overwritten
if os.path.exists(OUTPUT_DIR):
    print(f"Warning: Output directory {OUTPUT_DIR} already exists and may be overwritten.")
    print("Type 'yes' to continue, or 'no' to abort.")
    response = input()
    if response != "yes":
        raise ValueError("Aborted by user.")
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        match PROMPT_SOURCE:
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
                # CLAPZeroShotClassificationRewardProcessor(sample_rate=SAMPLE_RATE, reference_prompts=["dissonant, low quality, caucophonous music, glitch, midi"], target_prompt="beautiful, high quality, amazing music, natural, calming", temperature=0.25),
                PamRewardProcessor(sample_rate=SAMPLE_RATE, prompt_configs=prompt_pairs,temperature=0.25)
            ],
            reward_weights = REWARD_WEIGHTS,
            output_dir=OUTPUT_DIR
        )

    case "mil":
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
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"]
                                        ]
                            }
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "procedural":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], random.choice(program_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "piano":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_0"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "bass_drums":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_36"], tokenizer.vocab["Program_-1"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "drums":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_-1"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "bass_drums_keys":
                def gen():
                    for i in range(N_PROMPTS):
                        yield {"prompt": [tokenizer.vocab["BOS_None"], tokenizer.vocab["Program_5"],tokenizer.vocab["Program_36"], tokenizer.vocab["Program_-1"], tokenizer.vocab["Bar_None"], tokenizer.vocab["TimeSig_4/4"], tokenizer.vocab["Position_0"], np.random.choice(tempo_tokens)]}
                trn_ds = Dataset.from_generator(gen)
                tst_ds = Dataset.from_generator(gen).select(range(N_EVAL_PROMPTS))
                max_prompt_length = len(trn_ds[0]["prompt"])
            case "dataset":
                trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")
                trn_ds = trn_ds.shuffle()
                trn_ds = trn_ds.filter(lambda x: len(x["token_ids"]) <= MAX_COMPLETION_LENGTH)
                trn_ds = trn_ds.select(range(N_PROMPTS))

                tst_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/tst")
                tst_ds = tst_ds.shuffle()
                tst_ds = tst_ds.filter(lambda x: len(x["token_ids"]) <= MAX_COMPLETION_LENGTH)
                tst_ds = tst_ds.select(range(N_EVAL_PROMPTS))

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
                tst_ds = tst_ds.map(lambda x: {"prompt": extract_prompt(prepare_input(x["token_ids"], tokenizer))})
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
                PamRewardProcessor(sample_rate=SAMPLE_RATE, prompt_configs=prompt_pairs,temperature=0.25),
                # CLAPZeroShotClassificationRewardProcessor(sample_rate=SAMPLE_RATE, reference_prompts=["dissonant, low quality, caucophonous music, glitch, midi"], target_prompt="groovy, amazing, natural, high quality, studio, live", temperature=0.25),
                ProgramPromptAdherenceRewardProcessor(),
            ],
            reward_weights = REWARD_WEIGHTS,
            output_dir=OUTPUT_DIR
        )

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


def evaluate_base_model(model, dataset, reward_manager, output_dir, batch_size=BATCH_SIZE):
    """Generate examples from base model and evaluate with reward function using batched inference"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model.to("cuda")
    all_records = []

    # rename prompt to text
    dataset = dataset.rename_column("prompt", "text")
    
    # Process examples in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]
        
        # use dummy tokenizer to encode the prompts
        batch = dummy_tokenizer(**batch)
        completions = []
        
        # Generate completions for the batch
        with torch.no_grad():
            completions = model.generate(
                input_ids = batch["input_ids"].to("cuda"),
                attention_mask = batch["attention_mask"].to("cuda"),
                max_length=MAX_COMPLETION_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=dummy_tokenizer.pad_token_id,
                eos_token_id=dummy_tokenizer.eos_token_id,

            ).cpu()
            completions = completions[:, batch["input_ids"].shape[1]:]  # remove prompt from completions
    
        # Evaluate the batch with reward function
        records = reward_manager(
            prompts=batch["input_ids"],
            completions=completions,
            return_records=True,
        )

        # replace idx in records with relative idx in dataset
        for record in records:
            record["idx"] = i + record["idx"]
        
        # Append records to all_records
        all_records.extend(records)
    
    reward_manager.export_records(all_records, save_audio=True, output_dir=output_dir + "/eval", step=0)

    # Save results
    print(f"Evaluation complete. Results saved to {output_dir}")

evaluate_base_model(model, tst_ds, reward_manager, output_dir=OUTPUT_DIR + "/pre_eval", batch_size=BATCH_SIZE)
reward_manager.reset()
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

reward_manager.reset()

evaluate_base_model(model, tst_ds, reward_manager, output_dir=OUTPUT_DIR + "/post_eval", batch_size=BATCH_SIZE)
# %%
# now we can generate completions with the trn_ds