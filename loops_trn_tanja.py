#%%
import torch
from tqdm import tqdm
import pandas as pd
from transformers import Trainer, TrainingArguments
import os
import wandb
from transformers.integrations import WandbCallback
from midi_player import MIDIPlayer
from midi_player.stylers import cifka_advanced
import miditok
from transformers import Phi3Config, Phi3ForCausalLM
import numpy as np
import torch
import wandb
import random
from loops_util import prepare_input
from tokenisation import TanjaTokenizer, TanjaTokenizerConfig

# %%


tokenizer = TanjaTokenizer(
    TanjaTokenizerConfig(
        ticks_per_beat=96,
        coarse_ticks_per_beat=12,
        tempo_range=(60, 200),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        n_events=300,
    )
)

from datasets import Dataset
trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")
val_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/val")

#%%
from util import preview_sm
import symusic

idx = 2000
# preview midi form first sample
preview_sm(symusic.Score.from_midi(trn_ds[idx]['midi_bytes']))

tokens = tokenizer.midi_to_tokens(symusic.Score.from_midi(trn_ds[idx]['midi_bytes']), shuffle_events=True)

# turn back into midi
midi = tokenizer.tokens_to_midi(tokens)
preview_sm(midi)

print(len(tokens))

# %%

model_config = Phi3Config(
    vocab_size=len(tokenizer.vocab),
    eos_token_id=tokenizer.token_to_idx["EOS_None"],
    bos_token_id=tokenizer.token_to_idx["BOS_None"],
    pad_token_id=tokenizer.token_to_idx["PAD_None"],
    num_hidden_layers=6,
    hidden_size=512,
    intermediate_size=2048,
    num_attention_heads=8,
    )
model = Phi3ForCausalLM(model_config)

# %%

# print model params in scientific notation
# print(f"Model has {model.num_parameters()} parameters")
print(f"Model has {model.num_parameters() / 1e6} million parameters")

#%% first add token ids to dataset, catch exceptions and set to none


def maybe_convert(x):
    try:
        return tokenizer.midi_to_token_ids(
            symusic.Score.from_midi(x["midi_bytes"]), shuffle_events=True
        )
    except Exception as e:
        # print(f"Error converting midi: {e}")
        return None

trn_ds = trn_ds.map(
    lambda x: {
        "token_ids": maybe_convert(x)
    },
    num_proc=16,
)

val_ds = val_ds.map(
    lambda x: {
        "token_ids": maybe_convert(x)
    },
    num_proc=16,
)

# filter out None
trn_ds = trn_ds.filter(lambda x: x["token_ids"] is not None, num_proc=16)
val_ds = val_ds.filter(lambda x: x["token_ids"] is not None, num_proc=16)

# print length of 10 samples
for i in range(10):
    print(f"Sample {i}: {len(trn_ds[i]['token_ids'])}")

#%%

print(tokenizer.ids_to_tokens(trn_ds[1]["token_ids"]))

 # %%
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # for each seq in the batch, select a random crop of N tokens
        # select a random crop of tokens
        input_ids_stack = []
        position_ids_stack = []
        for b in batch:
            input_ids = b["token_ids"]
            position_ids = [i+1 for i in range(len(input_ids))]

            n_masked = random.randint(0, len(input_ids))

            # pick n_masked random positions to mask
            masked_positions = random.sample(range(len(input_ids)), n_masked)
            unmasked_positions = list(set(range(len(input_ids))) - set(masked_positions))
            # sort unmasked positions
            unmasked_positions = sorted(unmasked_positions)

            new_input_ids = [input_ids[i] for i in unmasked_positions] + [input_ids[i] for i in masked_positions]
            new_position_ids = [position_ids[i] for i in unmasked_positions] + [position_ids[i] for i in masked_positions]

            # add BOS_token id
            new_input_ids = [self.tokenizer.token_to_idx["BOS_None"], *new_input_ids]
            new_position_ids = [0, *new_position_ids]

            input_ids = torch.LongTensor(new_input_ids)
            position_ids = torch.LongTensor(new_position_ids)
            # add to stack
            input_ids_stack.append(input_ids)
            position_ids_stack.append(position_ids)

        # stack
        # crop
        input_ids = torch.stack(input_ids_stack, dim=0).long()

        # now create position_ids
        position_ids = torch.stack(position_ids_stack, dim=0).long()

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": input_ids.clone(),
        }


collator = MyDataCollator(tokenizer)

# create fake test batch 
test_batch = [trn_ds[i] for i in range(10)]
# show length of input_ids
for i in range(10):
    print(f"Sample {i}: {len(test_batch[i]['token_ids'])}")

# run through collator
collated = collator(test_batch)
# show length of input_ids
print(collated["input_ids"].shape)
print(collated["position_ids"].shape)

print(collated["input_ids"][0])
print(collated["position_ids"][0])
#%%

# create custom trainer
class CMLMTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for the model, ensuring position_ids are correctly passed through.
        """
        # First apply the standard preparation
        inputs = super()._prepare_inputs(inputs)
        
        # Make sure position_ids are on the same device as other tensors
        if "position_ids" in inputs:
            inputs["position_ids"] = inputs["position_ids"].to(inputs["input_ids"].device)
            
        return inputs
    


#%%

with wandb.init(
    project="aestune_mt",
    job_type="training",
    anonymous="allow",
    save_code=True,
) as run:

    training_args = TrainingArguments(
        output_dir=f'./outputs/mt/{run.name}',
        # max_steps=,
        num_train_epochs=20,
        eval_strategy="steps",
        eval_steps=50000,
        # eval_on_start=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500, 
        weight_decay=0.01,
        save_total_limit=8,
        bf16=True,
        # torch_compile=True,
        learning_rate=5e-4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 5e-6},
        remove_unused_columns=False,
        save_steps=25000,
        logging_steps=10,
    )

    trainer = CMLMTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=trn_ds,         # training dataset
        eval_dataset=val_ds,             # evaluation dataset
        data_collator=collator,  
    )

    

    trainer.train()


# %%
