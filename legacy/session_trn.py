#%%
import symusic
import transformers
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
checkpoint = "session-gpt/checkpoints/nospace/checkpoint-8000"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(checkpoint)

OUTPUT_DIR = "artefacts/session-space-structure"
#%%
def gen():
    yield {"prompt": "This "}
ds = Dataset.from_generator(gen)


aes_predictor = initialize_predictor()

SAVE_INTERVAL = 100
reward_step = 0



def extract_abc(abc):
    if "@" not in abc:
        return "invalid"
    if "€" not in abc:
        return "invalid"
    abc = abc.split("@")[1]
    abc = abc.split("€")[0]
    return abc    

def aes_reward(completions, **kwargs):
    abc = [extract_abc(c) for c in completions]
    sms = []
    for a in abc:
        try:
            sms.append(symusic.Score.from_abc(a))
        except:
            print(f"Error parsing abc: {a}")
            sms.append(symusic.Score())
    audio = []
    for sm in sms:
        try:
            audio.append(synth.render(sm))
        except:
            print(f"Error rendering score: {sm}")
            audio.append(torch.zeros
            (1,44100*1))
    predictor_inputs = [{"path": torch.tensor(audio[i]).float(), "sample_rate": 44100} for i in range(len(audio))]
    print(f"Predicting aesthetics")
    scores = aes_predictor.forward(predictor_inputs)

    # take mean of CE, CU, PC, PQ
    rewards = [sum([score["CE"], score["CU"], score["PC"], score["PQ"]]) for score in scores]
    # rewards = [score["CE"] for score in scores]

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

BATCH_SIZE=16

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=200,
    # max_prompt_length=1,
    num_train_epochs=100_000,
    learning_rate=1e-5,
    report_to="wandb",
    logging_steps=1,
    num_generations=BATCH_SIZE,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=5000,
    beta=0.5,
)
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
import symusic
score = "artefacts/session/fs_renders/16900/reward=25.907721996307373_7.mid"

# check time signature
sm = symusic.Score(score)

print(sm.time_signatures)
# %%
