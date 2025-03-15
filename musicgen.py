#%%
%env CUDA_VISIBLE_DEVICES=1

#%%
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import miditok
import symusic
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
from symusic import Synthesizer, BuiltInSF3, dump_wav
from audiobox_aesthetics.infer import initialize_predictor

#%%
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor.pad_token_id = 0

#%%
print(processor)
# %%
# unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
# audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=100)
#%%
import matplotlib.pyplot as plt
import IPython.display as ipd
def play_audio(audio):
    plt.figure(figsize=(10, 2))
    plt.plot(audio[0,0])
    plt.show()
    ipd.display(ipd.Audio(audio[0], rate=16000))

#%%

OUTPUT_DIR = "artefacts/musicgen"
#%%


#%%


aes_predictor = initialize_predictor()

SAVE_INTERVAL = 100
reward_step = 0


#%%

BATCH_SIZE=3
MAX_GENERATED_TOKENS = 100
# get examples from dataset

batch = model.get_unconditional_inputs(num_samples=BATCH_SIZE)


completions = model.generate(**batch, do_sample=True, max_new_tokens=MAX_GENERATED_TOKENS)
    

print(completions)

#%%
    
def aes_reward(completions, **kwargs):
    
    predictor_inputs = [{"path": torch.tensor(audio[i]).float(), "sample_rate": 44100} for i in range(len(audio))]
    print(f"Predicting aesthetics")
    scores = aes_predictor.forward(predictor_inputs)

    # take mean of CE, CU, PC, PQ
    # rewards = [sum([score["CE"], score["CU"], score["PC"], score["PQ"]]) for score in scores]
    # rewards = [score["CE"] for score in scores]

    rewards = [float(sm.note_num()) for sm in sms]
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
    rewards = [0.0 for i in range(completions.shape[0])]
    return rewards


#%%

#%% one pass through grpo



#%%
    
# dummy_tokenizer = DummyTokenizer()

import os
os.environ["WANDB_PROJECT"] = "music-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

BATCH_SIZE=8

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=100,
    max_prompt_length=1,
    num_train_epochs=100_000,
    learning_rate=1e-5,
    report_to="wandb",
    logging_steps=1,
    num_generations=BATCH_SIZE,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    beta=0.5
)
# trainer = GRPOTrainer(
#     model=model,
#     reward_funcs=aes_reward,
#     args =  config,
#     train_dataset=ds,
#     processing_class=dummy_tokenizer,
# )
# save model
trainer.save_model()
trainer.train()
#%%
#%%
