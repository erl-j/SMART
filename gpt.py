#%%
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

dataset = load_dataset("trl-lib/tldr", split="train")

# create peft config
from peft import LoraConfig, TaskType

# Define the reward function, which rewards completions that are close to 20 characters
def no_short_words(completions, prompts):
    return [len(c) for c in completions]

ckpt_path = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    ckpt_path,
    quantization_config=config,
    device_map="auto",
    trust_remote_code=True,
)

training_args = GRPOConfig(output_dir="Qwen2-7B-GRPO", logging_steps=10, per_device_train_batch_size=8, max_completion_length=20, bf16=True)
trainer = GRPOTrainer(
    model=ckpt_path,
    reward_funcs=no_short_words,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
# %%
