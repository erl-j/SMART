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
trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn")
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


# %%

from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel

model_config = GPT2Config(
    vocab_size=len(tokenizer.vocab),
    eos_token_id=tokenizer.token_to_idx["EOS_None"],
    bos_token_id=tokenizer.token_to_idx["BOS_None"],
    pad_token_id=tokenizer.token_to_idx["PAD_None"],
    num_hidden_layers=6,
    hidden_size=512,
    intermediate_size=512*4,
    num_attention_heads=8,
    max_position_embeddings=10_000,
    tied_word_embeddings=False,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    resid_pdrop=0.0,
)

model = GPT2LMHeadModel(model_config)

# model_config = Phi3Config(
#     vocab_size=len(tokenizer.vocab),
#     eos_token_id=tokenizer.token_to_idx["EOS_None"],
#     bos_token_id=tokenizer.token_to_idx["BOS_None"],
#     pad_token_id=tokenizer.token_to_idx["PAD_None"],
#     num_hidden_layers=6,
#     hidden_size=512,
#     intermediate_size=2048,
#     num_attention_heads=8,
#     )
# model = Phi3ForCausalLM(model_config)



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

    def __call__(self, batch, retokenize=True):
        # for each seq in the batch, select a random crop of N tokens
        # select a random crop of tokens
        input_ids_stack = []
        position_ids_stack = []
        for b in batch:
            if retokenize:
                # retokenize
                input_ids = tokenizer.midi_to_token_ids(symusic.Score.from_midi(b["midi_bytes"]), shuffle_events=True)
            else:
                input_ids = b["token_ids"].copy()

            position_ids = [i for i in range(len(input_ids))]

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
            new_position_ids = [*new_position_ids, len(new_position_ids)]

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
collated = collator(test_batch, retokenize=False)

# now check that we can recover original input_ids by sorting by position_ids
for i in range(10):
    shuffled_input_ids = collated["input_ids"][i][1:]
    shuffled_position_ids = collated["position_ids"][i][:-1]
    argsorted = torch.argsort(shuffled_position_ids)
    # sort input_ids by position_ids
    sorted_input_ids = shuffled_input_ids[argsorted]
    # assert that sorted_input_ids is equal to original input_ids
    assert torch.equal(sorted_input_ids, torch.LongTensor(test_batch[i]["token_ids"])), f"Sample {i} failed"
    print(f"Sample {i} passed")

    # assert that the first sorted input id is the BOS token
    # assert that the second is a Program_ token
    sorted_tokens = tokenizer.ids_to_tokens(sorted_input_ids)
    print(sorted_tokens[:10])
    assert sorted_tokens[0].startswith("Tempo_"), f"Sample {i} failed"
    # assert that second starts with "Onset_"
    assert sorted_tokens[1].startswith("Program_"), f"Sample {i} failed"
# now sort input_ids by length


def verify_position_embeddings(model):
    # Get the device that model is on
    device = next(model.parameters()).device
    
    # Create test inputs with different position IDs but same token IDs
    test_inputs = {
        "input_ids": torch.LongTensor([[1, 2, 3]]).to(device),
        "position_ids": torch.LongTensor([[5, 2, 9]]).to(device)
    }
    
    with torch.no_grad():
        # Get outputs with first set of position IDs
        out1 = model(**test_inputs)
        
        # Change only the position IDs
        test_inputs["position_ids"] = torch.LongTensor([[1, 2, 3]]).to(device)
        
        # Get outputs with second set of position IDs
        out2 = model(**test_inputs)

        test_inputs["position_ids"] = torch.LongTensor([[1, 2, 3]]).to(device)

        out3 = model(**test_inputs)

    # check if out 1 and out 3 are the same
    if torch.allclose(out2.logits, out3.logits):
        print("Great news! Same position IDs are giving same outputs.")
    else:
        print("WARNING: Position IDs are not working correctly!")
        print("This may cause issues with your masked language modeling approach.")
        raise ValueError("Position IDs are not working correctly!")
        
    # Check if outputs are different
    if torch.allclose(out1.logits, out2.logits):
        print("WARNING: Position IDs don't seem to affect model outputs!")
        print("This may cause issues with your masked language modeling approach.")
        raise ValueError("Position IDs are not affecting model outputs.")
    else:
        diff = (out1.logits - out2.logits).abs().max().item()
        print(f"Position embeddings are working correctly. Max difference: {diff:.6f}")

verify_position_embeddings(model)
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
        eval_steps=100_000,
        # eval_on_start=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=10_000, 
        weight_decay=0.0,
        save_total_limit=8,
        bf16=True,
        # torch_compile=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 5e-6},
        remove_unused_columns=False,
        save_steps=10_000,
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
