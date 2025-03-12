#%%
import miditok
import symusic
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import GRPOConfig, GRPOTrainer

from datasets import Dataset, load_dataset
from symusic import Synthesizer, BuiltInSF3, dump_wav
from audiobox_aesthetics.infer import initialize_predictor


sample_rate = 44100
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)

synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)


#%%
import sys
sys.path.append("./multitrack-midi-music-generator")
from utils import get_midi_from_string, get_midi_bytes_from_string
import tempfile

    
# class MultiTrackMusicSystem():

#     def __init__(self):
        
#         # print sorted vocab by value
#         # sort_vocab = sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])


#         self.pad_token_id = self.tokenizer.vocab["[PAD]"]
#         self.eos_token_id = self.tokenizer.vocab["[PAD]"]
#         self.prompt = "PIECE_START GENRE=CLASSICAL TRACK_START"

#     def encode(self, x, **kwargs):
#         return self.tokenizer(x, return_dict=True, return_tensors="pt")
    
#     def decode(self, x, **kwargs):
#         output_str = self.tokenizer.decode(x)
        
    
#     def batch_decode(self, x, **kwargs):
#         return [self.decode(x[i]) for i in range(x.shape[0])]
    
    
#     def save_pretrained(self, path):
#         self.model.save_pretrained(path)

#     def __call__(self, x, **kwargs):
#         return self.tokenizer(x, return_tensors="pt")

# def test_reward(completions, **kwargs):
#     # each completions should be dict
#     return [1 for _ in range(completions.shape[0])]



# dummy_tokenizer = DummyTokenizer(tokenizer)

# for a new system we need.
# a tokenizer, a model
# a function that processes decoder output to get audio, midi etc..



tokenizer = AutoTokenizer.from_pretrained("juancopi81/lmd_8bars_tokenizer", max_length=4096)
model = AutoModelForCausalLM.from_pretrained(
            "juancopi81/lmd-8bars-2048-epochs40_v4"
        )
model.pad_token_id = tokenizer.vocab["[PAD]"]
model.eos_token_id = tokenizer.vocab["[PAD]"]

# set to tokenizer
tokenizer.pad_token_id = tokenizer.vocab["[PAD]"]
tokenizer.eos_token_id = tokenizer.vocab["[PAD]"]


# this one is passed to model to generate
INPUT_PROMPT = "PIECE_START GENRE=JAZZ TRACK_START"
# this one is just used for decoding
OUTPUT_PROMPT = "PIECE_START GENRE=JAZZ TRACK_START"

# takes whatever comes out of the tokenizer decode and returns a dict with midi, wav, token_ids, str
def render(x):
    output_str = OUTPUT_PROMPT + " " + x
    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        filepath = get_midi_from_string(output_str, f.name, qpm=24)
        midi = symusic.Score(filepath)
    # render the midi file
    audio = synth.render(midi)
    return {"token_ids": x, "midi": midi, "wav": audio, "str": output_str}

def gen():
    yield {"prompt": INPUT_PROMPT}

ds = Dataset.from_generator(gen)
    

#%%

OUTPUT_DIR = "artefacts/hello-world-6"

#%%
#%%
import sys

#%%


aes_predictor = initialize_predictor()

SAVE_INTERVAL = 100
reward_step = 0

def aes_reward(completions, **kwargs):
    completions = [render(completions[i]) for i in range(len(completions))]
    # sms = [tokenizer(completions[i].cpu().numpy()[None,...] ) for i in range(completions.shape[0])]
    sms = [completions[i]["midi"] for i in range(len(completions))]
    print(f"Rendering to audio")
    audio = [synth.render(sm) for sm in sms]
    predictor_inputs = [{"path": torch.tensor(audio[i]).float(), "sample_rate": 44100} for i in range(len(audio))]
    print(f"Predicting aesthetics")
    scores = aes_predictor.forward(predictor_inputs)
    rewards = [score["CE"] for score in scores]
    print(f"average reward: {sum(rewards)/len(rewards)}")
    print(f"Rewards: {rewards}")
    global reward_step
    print(f"reward step: {reward_step}")
    if reward_step % SAVE_INTERVAL == 0:
        os.makedirs(f"{OUTPUT_DIR}/fs_renders/{reward_step}", exist_ok=True)
        for i in range(len(sms)):
            sms[i].dump_midi(f"{OUTPUT_DIR}/fs_renders/{reward_step}/reward={rewards[i]}_{i}.mid")
            dump_wav( f"{OUTPUT_DIR}/fs_renders/{reward_step}/reward={rewards[i]}_{i}.wav", audio[i], sample_rate, use_int16=True)
    reward_step += 1
    return rewards
# %%

#%%
import os
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"


config = GRPOConfig(output_dir=OUTPUT_DIR, max_completion_length=50, num_train_epochs=100_000, learning_rate=1e-5, report_to="wandb", logging_steps=1, num_generations=8)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=aes_reward,
    args =  config,
    train_dataset=ds,
    processing_class=tokenizer,

)
# save model
trainer.save_model()
trainer.train()
#%%
#%%
