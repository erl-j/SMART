import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import os
import wandb
os.environ["WANDB_PROJECT"] = "gpt-grpo"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"

# rewards alliteration, i.e starting the words with the same letter in a row
def reward_(completion):
    # get maximum alliteration
    words = completion.split()
    max_alliteration = []
    # alliteration is counted if the first letter of the word is the same as the previous word but the word is not the same
    for i in range(1, len(words)):
        if words[i][0] == words[i-1][0] and words[i] != words[i-1]:
            max_alliteration.append(words[i])     
    
    return max_alliteration

def reward_fn(completions, **kwargs):
    rewards = [reward_(completion) for completion in completions]
    print(f"Rewards: {rewards}")
    return rewards

# Initialize model and tokenizer
model_name = "gpt2"  # You can replace with any GPT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Create a simple dataset with one prompt
def gen():
    yield {"prompt": "Once upon a time there was a king."}

ds = Dataset.from_generator(gen)

# Configure GRPO training
output_dir = "e_counter_grpo_output"
config = GRPOConfig(
    temperature=1.0,
    output_dir=output_dir,
    max_completion_length=50,  # Keep completions short
    num_train_epochs=10000,       # Small number of epochs for demonstration
    learning_rate=1e-4,
    logging_steps=1,
    # num_generations=4,         # Generate 4 samples per step
    save_steps=1000,              # Save model every 5 steps
    report_to="wandb",
    log_completions=True,
)

# Initialize and start trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=ds,
    processing_class=tokenizer,
)
trainer.train()
