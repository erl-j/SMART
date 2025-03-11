#%%
import miditok
import symusic
import transformers
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
model = transformers.AutoModelForCausalLM.from_pretrained("Natooz/Maestro-TSD-bpe20k", trust_remote_code=True, torch_dtype="auto")
tokenizer = miditok.REMI.from_pretrained("Natooz/Maestro-TSD-bpe20k")


OUTPUT_DIR = "artefacts/hello-world-3"
#%%
def gen():
    yield {"prompt": ""}
ds = Dataset.from_generator(gen)
print(ds[0])


def reward_len(completions, **kwargs):
    sms = [tokenizer(completions[i].cpu().numpy()[None,...] ) for i in range(completions.shape[0])]
    rewards = [sm.note_num() for sm in sms]
    print(f"average reward: {sum(rewards)/len(rewards)}")
    print(f"Rewards: {rewards}")
    return rewards

aes_predictor = initialize_predictor()

SAVE_INTERVAL = 100
reward_step = 0

def aes_reward(completions, **kwargs):
    print(f"Decoding to MIDI")
    sms = [tokenizer(completions[i].cpu().numpy()[None,...] ) for i in range(completions.shape[0])]

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

class DummyTokenizer():
    def __init__(self,tokenizer):
        self.pad_token_id = tokenizer.vocab["PAD_None"]
        self.eos_token_id = tokenizer.vocab["EOS_None"]
        self.bos_token_id = tokenizer.vocab["BOS_None"]

    def encode(self, x, **kwargs):
        # print(f"Called encode with {x}")
        return {"input_ids": torch.tensor([[2]])}
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
        return {
                "input_ids": torch.tensor([[2] for sample in range(n_samples)]), 
                "attention_mask":torch.tensor([[1] for sample in range(n_samples)]),
            }
    
dummy_tokenizer = DummyTokenizer(tokenizer)
# %%

#%%
import os
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

config = GRPOConfig(output_dir=OUTPUT_DIR,max_prompt_length=1, num_train_epochs=100_000, learning_rate=1e-5, report_to="wandb", logging_steps=1, num_generations=8)
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
#%%
