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
from tokenisation import IrmaTokenizer, IrmaTokenizerConfig

# %%
IrmaTokenizerConfig(
        ticks_per_beat=96,
        positions_per_beat=12,
        tempo_range=(60, 250),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        duration_ranges=((2,12), (16,6))
)

tokenizer = IrmaTokenizer(
    IrmaTokenizerConfig(
        ticks_per_beat=96,
        positions_per_beat=12,
        tempo_range=(60, 250),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        duration_ranges=((2,12), (16,6))
    )
)

from datasets import Dataset
trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")
val_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/val")

#%%
from util import preview_sm
import symusic

idx = 100
# preview midi form first sample
preview_sm(symusic.Score.from_midi(trn_ds[idx]['midi_bytes']))

tokens = tokenizer.midi_to_tokens(symusic.Score.from_midi(trn_ds[idx]['midi_bytes']), shuffle_tracks=True)

# turn back into midi
midi = tokenizer.tokens_to_midi(tokens)
preview_sm(midi)

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

#%% first add token ids to dataset
trn_ds = trn_ds.map(
    lambda x: {
        "token_ids": tokenizer.midi_to_token_ids(
            symusic.Score.from_midi(x["midi_bytes"]), shuffle_tracks=True
        )
    },
    num_proc=16,
)

val_ds = val_ds.map(
    lambda x: {
        "token_ids": tokenizer.midi_to_token_ids(
            symusic.Score.from_midi(x["midi_bytes"]), shuffle_tracks=True
        )
    },
    num_proc=16,
)

#%%
MAX_SEQ_LENGTH = 2048

# filter out sequences longer than MAX_SEQ_LENGTH
trn_ds = trn_ds.filter(lambda x: len(x["token_ids"]) < MAX_SEQ_LENGTH, num_proc=16)
val_ds = val_ds.filter(lambda x: len(x["token_ids"]) < MAX_SEQ_LENGTH, num_proc=16)
# %%
class MyDataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _crop(self, seq):
        return seq[: self.max_seq_len]

    def __call__(self, batch):
        # for each seq in the batch, select a random crop of N tokens
        # select a random crop of tokens
        input_ids_stack = []
        for b in batch:
            input_ids = b["token_ids"]
            # add BOS_token and EOS_token
            input_ids = [self.tokenizer.token_to_idx["BOS_None"], *input_ids, self.tokenizer.token_to_idx["EOS_None"]]
            # convert to long tensor
            input_ids = torch.LongTensor(input_ids)
            # append to stack
            input_ids_stack.append(input_ids)
    
        # crop
        input_ids = [self._crop(seq) for seq in input_ids_stack]
            
        # pad tokens to max length
        input_ids = self._pad_batch(input_ids)
        
        attention_mask = torch.where(
            input_ids != self.tokenizer.token_to_idx["PAD_None"], 1, 0
        )
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        
    def _pad_batch(
        self,
        batch,
        pad_on_left=False,
    ):
        # Check if padding is necessary.
        length_of_first = batch[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in batch)

        if are_tensors_same_length:
            return torch.stack(batch, dim=0).long()

        return torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=self.tokenizer.token_to_idx["PAD_None"]
        ).long()
    

collator = MyDataCollator(tokenizer, MAX_SEQ_LENGTH)


with wandb.init(
    project="aestune_mt",
    job_type="training",
    anonymous="allow",
    save_code=True,
) as run:

    training_args = TrainingArguments(
        output_dir=f'./outputs/mt/{run.name}',
        # max_steps=,
        num_train_epochs=10,
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

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=trn_ds,         # training dataset
        eval_dataset=val_ds,             # evaluation dataset
        data_collator=collator,  
    )

    # # move it up
    # class WandbPredictionProgressCallback(WandbCallback):
    #     def __init__(self, trainer, tokenizer):
    #         super().__init__()
    #         self.trainer = trainer
    #         self.tokenizer = tokenizer
    #         self.inf_tokens = [tokenizer[t] for t in tokenizer.vocab if t.startswith("INF")]

    #     """
    #     take the outputs, try to reoder the sequence and return midi player for wandb
    #     puts parts in separate tracks if possible
    #     """

    #     def display_midi(self, out, title="generation sample"):
    #         out.dump_midi("./artefacts/tmp.mid")
    #         return MIDIPlayer(
    #             "./artefacts/tmp.mid",
    #             height=400,
    #             styler=cifka_advanced,
    #             title=title,
    #         )

    #     # render unconditional generation example
    #     def render_prediction(self):
    #         out = self.trainer.model.generate(
    #                 torch.LongTensor([[self.tokenizer.vocab["BOS_None"]]]).to(self.trainer.model.device),
    #                 max_new_tokens=1024 + 16,
    #                 do_sample=True,
    #                 use_cache=True,
    #                 # top_p=0.95,
    #             )
    #         print("gen_sample", out)
    #         out = self.tokenizer(out.cpu().tolist())
    #         mp = self.display_midi(
    #             out, title=f"Generation sample epoch {round(self.trainer.state.epoch)}"
    #         )
    #         wandb.log({"gen_sample": wandb.Html(mp.html)})

    #     def on_evaluate(self, args, state, control, **kwargs):
    #         super().on_evaluate(args, state, control, **kwargs)
    #         # print("after eval callback")
    #         assert not self.trainer.model.training
    #         self.render_prediction()


    # progress_callback = WandbPredictionProgressCallback(
    #         trainer=trainer,
    #         tokenizer=tokenizer,
    #     )
    # trainer.add_callback(progress_callback)

    trainer.train()


# %%
