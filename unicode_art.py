#%%
import transformers
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
import torch.nn.functional as F
import os
import random

#%%
# set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# now count the number of available GPUs
print(torch.cuda.device_count())

#%%
#%%
#%%
model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True, torch_dtype="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
OUTPUT_DIR = "artefacts/unicode-art"

#%%

nouns = ["I", "no", "love"]
def gen():
    for i in range(1000):
        yield {"prompt": {"role": "user", "content": f"Write a short palindrome sentence with the word {random.choice(nouns)}."}
            }
ds = Dataset.from_generator(gen)

#%%

def is_palindrome(s):
    s = s.lower()
    s = s.strip()
    # remove spaces
    s = s.replace(" ", "")
    return s == s[::-1]

def aes_reward(completions, **kwargs):
    print(completions)
    # print input arguments
    prompts = torch.tensor(kwargs["prompts"])
    # check if palindrome
    rewards = [ 1.0 if is_palindrome(prompt["content"]) else 0.0 for prompt in prompts]
    return rewards

# %%
os.environ["WANDB_PROJECT"] = "palindrom"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

BATCH_SIZE=2

config = GRPOConfig(
    temperature=1.0,
    output_dir=OUTPUT_DIR,
    max_completion_length=250,
    max_prompt_length=6,
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
    processing_class=tokenizer,
)
# save model
trainer.save_model()
trainer.train()
#%%
