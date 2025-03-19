#%%
import miditok
import symusic
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
from symusic import Synthesizer, BuiltInSF3, dump_wav
from audiobox_aesthetics.infer import initialize_predictor

#%%

from datasets import load_dataset
from transformers import ClapModel, ClapProcessor

sample_rate = 48_000

# clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(0)
# clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

# def get_clap_features(audio_samples):
#     audio_samples = [audio_samples[i].mean(0) for i in range(len(audio_samples))]
#     inputs = clap_processor(audios=audio_samples, return_tensors="pt", sampling_rate=sample_rate).to(0)
#     print(f"inputs: {inputs}")
#     audio_embed = clap_model.get_audio_features(**inputs)
#     return audio_embed

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


OUTPUT_DIR = "artefacts/pianophi-kl=0.1-frequent-logs-32-program=24"

PIANO_PROGRAM = 24

#%%

def set_piano_program(sm):
    for track in sm.tracks:
        track.program = PIANO_PROGRAM
    return sm

def set_to_drums(sm):
    for track in sm.tracks:
        track.is_drum = True
    return sm



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

SAVE_INTERVAL = 1
reward_step = 0

import torch.nn.functional as F






def get_nn_sim(embeddings, k=2, include_self=False):
    # Normalize embeddings to unit length for cosine similarity
    normalized_embeds = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(normalized_embeds, normalized_embeds.T)
    
    # Handle self-similarity
    if not include_self:
        # Set diagonal to -1 so it won't be selected in topk
        sim_matrix.fill_diagonal_(-1.0)
    
    # Get top-k similarities for each sample
    topk_sims = sim_matrix.topk(k, dim=1).values
    
    # Compute mean similarity for each sample
    mean_sims = topk_sims.mean(dim=1)
    
    # Lower value indicates more sparsity
    return mean_sims



def get_aes_scores(records):
    # prepare inputs. 
    predictor_inputs = [{"path": torch.tensor(record["audio"]).float(), "sample_rate": sample_rate, "idx":i} for i, record in enumerate(records) if record["audio"] is not None]
    print(f"Predicting aesthetics")
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

    print(f"Decoding to MIDI")
    sms = [tokenizer(completions[i].cpu().numpy()[None,...] ) for i in range(completions.shape[0])]

    # set piano program
    sms = [set_piano_program(sm) for
        sm
        in sms
    ]

    # sms = [set_to_drums(sm) for
    #     sm
    #     in sms
    # ]

    records = [{"completion": completions[i], "sm": sms[i]} for i in range(completions.shape[0])]


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
    # def get clap features
    # audio_embed = get_clap_features(audio)
    # print(f"audio_embed: {audio_embed.shape}")
    # # get matmul of audio_embed
    # nn_sim = get_nn_sim(audio_embed, k=1)
    # print(f"nn_sim: {nn_sim}")
    # nn_sim_argsort = nn_sim.argsort(descending=True)
    # # add to rewards
    # rewards = [rewards[i] + nn_sim_argsort[i] for i in range(len(rewards))]

    # give points according to highest sparsity
    # 

    # rewards = [float(sm.note_num()) for sm in sms]
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
#%%




class DummyTokenizer():
    def __init__(self,tokenizer):
        self.pad_token_id = tokenizer.vocab["PAD_None"]
        self.eos_token_id = tokenizer.vocab["EOS_None"]
        self.bos_token_id = tokenizer.vocab["BOS_None"]

    def encode(self, x, **kwargs):
        # print(f"Called encode with {x}")
        return {"input_ids": torch.tensor([[1]])}
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
                "input_ids": torch.tensor([[1] for sample in range(n_samples)]), 
                "attention_mask":torch.tensor([[1] for sample in range(n_samples)]),
            }
    
dummy_tokenizer = DummyTokenizer(tokenizer)
# %%

#%%
import os
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

BATCH_SIZE=64

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=250,
    max_prompt_length=1,
    num_train_epochs=200,
    learning_rate=1e-5,
    report_to="wandb",
    logging_steps=1,
    num_generations=BATCH_SIZE,
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
trainer.save_model()
trainer.train()
#%%
#%%
