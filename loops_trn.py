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

# %%
    

MAX_PROMPT_LENGTH = 16    
MAX_TOKENS = 2048

tokenizer_config = miditok.TokenizerConfig.load_from_json("data/gmd_loops_2_tokenized_2/tokenizer_config.json")
tokenizer = miditok.REMI(tokenizer_config)


from datasets import Dataset
trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn")
trn_ds.filter(lambda x: len(x["token_ids"]) <= MAX_TOKENS, num_proc=16)

val_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/val")
val_ds.filter(lambda x: len(x["token_ids"]) <= MAX_TOKENS, num_proc=16)

#%%
# print example
sample = trn_ds[25]

tokens = tokenizer._ids_to_tokens(sample["token_ids"])
# convert to tokens


# We'll call this representation IRMA.

# now we will rearange the tokens
# first, put the time_signature and tempo.
# then put the program tokens that will be included.
# then for each program.
# program then Bar_None and then the whole sequence.
# 
# so BOS_None, TimeSig_0, Tempo_0
# Program_0, Program_1, Program_2 
# SEP. Program_0 Bar_None Position_ Pitch_ Duration_ Velocity_ Bar_None ...
# SEP. Program_1 Bar_None Position_ Pitch_ Duration_ Velocity_ Bar_None ...
# SEP. Program_2 Bar_None Position_ Pitch_ Duration_ Velocity_ Bar_None ...
# BAR_None ... 
# BAR_None ...

# At the end
# use
# sep

# make a coloured print of the tokens with colours indicating type ("Prefix")

colormap = {"Tempo" : "red", "TimeSig": "blue", "Program": "green", "INF": "yellow", "Bar": "purple", "Position": "orange", "Pitch": "pink", "Duration": "brown", "Velocity": "grey", "SEP": "teal"}

from IPython.display import HTML, display
def display_tokens(tokens):
    html = "<div style='font-size: 10px;'>"
    for token in tokens:
        # get the type of the token
        token_type = token.split("_")[0]
        color = colormap.get(token_type, "black")
        html += f"<span style='color: {color};'>{token}</span> "
    html += "</div>"
    return display(HTML(html))

def crop_tokens_by_bars(tokens, bars):
    # crop the tokens to the first N bars
    # find the first Bar_None token
    bar_count = 0
    for i, token in enumerate(tokens):
        if token.startswith("Bar_None"):
            bar_count += 1
        if bar_count == bars:
            return tokens[:i+1]
    return tokens

def remove_timesig_tokens(tokens):
    # remove all timesig tokens after first
    # find the first TimeSig token
    tokens = [t for t in tokens if not t.startswith("TimeSig_")]
    return tokens

def remove_tempo_tokens(tokens):
    # remove all tempo tokens after first
    # find the first Tempo token
    tokens = [t for t in tokens if not t.startswith("Tempo_")]
    return tokens

# first step is to remove all timesig token after first
# second step it to remove all tempo tokens after first

def isolate_program(tokens, program):
    # remove all tokens referring to other programs.
    cursor_idx = 0
    new_tokens = []
    while cursor_idx < len(tokens):
        token = tokens[cursor_idx]
        attr, value = token.split("_")
        if attr == "Program" and value != program:
            # skip this token and next 3
            cursor_idx += 4
        else:
            new_tokens.append(token)
            cursor_idx += 1
    # remove redundant position tokens, 
    # they can be inferred by them being followed by a position token
    tokens = new_tokens
    cursor_idx = 0
    new_tokens = []
    while cursor_idx < len(tokens):
        token = tokens[cursor_idx]
        attr, value = token.split("_")
        if cursor_idx<len(tokens)-1 and attr == "Position" and not tokens[cursor_idx+1].startswith(f"Program_{program}"):
            cursor_idx += 1
        else:
            new_tokens.append(token)
            cursor_idx += 1
    return new_tokens

def remove_durations_from_drums(tokens):
    # remove all durations from drums
    # find the first Program_0 token
    cursor_idx = 0
    new_tokens = []
    while cursor_idx < len(tokens):
        token = tokens[cursor_idx]
        attr, value = token.split("_")
        if attr == "Program" and value == "-1":
            # add curr, next, next after that but skip the duration
            new_tokens.append(tokens[cursor_idx])
            new_tokens.append(tokens[cursor_idx+1])
            new_tokens.append(tokens[cursor_idx+2])
            cursor_idx += 4
        else:
            new_tokens.append(token)
            cursor_idx += 1
    return new_tokens


def split_out_programs(tokens):
    # get all programs
    program_tokens = [t.split("_")[-1] for t in tokens if t.startswith("Program_")]
    # get unique programs
    programs = np.unique(np.random.permutation(program_tokens)).tolist()

    program_tokens = []
    for program in programs:
        # make copy of tokens
        new_tokens = isolate_program(tokens, program)
        # remove all durations from drums
        if program == "-1":
            new_tokens = remove_durations_from_drums(new_tokens)
        # remove all program tokens
        new_tokens = [t for t in new_tokens if not t.startswith("Program_")]
        # add program prefix before the program tokens
        new_tokens = ["SEP_None", f"Program_{program}"] + new_tokens
        # append to program_tokens
        program_tokens.append(new_tokens)
    # shuffe the programs
    program_tokens = random.sample(program_tokens, len(program_tokens))
    # flatten the list
    program_tokens = [item for sublist in program_tokens for item in sublist]
    return program_tokens

def std_to_irma(tokens, bars=4):
    # crop the tokens to the first 4 bars
    tokens = crop_tokens_by_bars(tokens, bars+ 1)
    # get first timesig token
    for i, token in enumerate(tokens):
        if token.startswith("TimeSig_"):
            timesig = token
            break
    # get first tempo token
    for i, token in enumerate(tokens):
        if token.startswith("Tempo_"):
            tempo = token
            break
    tokens = remove_timesig_tokens(tokens)
    tokens = remove_tempo_tokens(tokens)
    # split out programs
    tokens = split_out_programs(tokens)

    # get all unique programs in order of appearance
    header_programs = [t for t in tokens if t.startswith("Program_")]
    # remove duplicates while preserving order
    header_programs = list(dict.fromkeys(header_programs))
    # now create header.
    header = [timesig, tempo]
    header.extend(header_programs)
    # now add back tokens
    header.extend(tokens)
    # now add EOS token
    # now add PAD token
    return header

def irma_to_std(tokens, default_drum_duration="0.1.12"):

    # first remove the header
    # remove all tokens until the first SEP token
    header, body = tokens[:tokens.index("SEP_None")], tokens[tokens.index("SEP_None")+1:]
    timesig = header[0]
    tempo = header[1]
    # get program tokens
    program_tokens = [t for t in header if t.startswith("Program_")]

    # now split tokens in body by "SEP_None"
    SEP_indices = [i for i, token in enumerate(body) if token == "SEP_None"]
    program_bodies = []
    for i in range(len(SEP_indices)-1):
        program_bodies.append(body[SEP_indices[i]+1:SEP_indices[i+1]])
    # add the last program body
    program_bodies.append(body[SEP_indices[-1]+1:])

    # now add back the programs 
    # first add the header
    body = []
    body.append(timesig)
    body.append(tempo)

    # now add the program bodies
    for i, program_body in enumerate(program_bodies):
        # add the program token
        body.append(f"Program_{i}")
        # add the program body
        body.extend(program_body)
        # add the SEP token
        body.append("SEP_None")

    # now add the last program token
    body.append(f"Program_{i}")

    # now add the SEP token
    body.append("SEP_None")

    # to the body add the Bar_None token



#%%
# write tokens to file
with open("artefacts/tokens.txt", "w") as f:
    for token in tokens:
        f.write(token + "\n")

# now we will rearange the tokens
irma_tokens = std_to_irma(tokens)

# write tokens to file
with open("artefacts/tokens_irma.txt", "w") as f:
    for token in irma_tokens:
        f.write(token + "\n")

tokens_hat = irma_to_std(irma_tokens, default_drum_duration="0.1.12")

# write tokens to file
with open("artefacts/tokens_hat.txt", "w") as f:
    for token in tokens_hat:
        f.write(token + "\n")


# print len before and after
print(f"Tokens before: {len(tokens)}")
print(f"Tokens after: {len(irma_tokens)}")

# convert to ids
from util import preview_sm

tokens_sm = tokenizer.decode(tokens)
preview_sm(tokens_sm)

# irma_sm = tokenizer.decode(irma_tokens)

def irma_decode(tokens):
    # remove head
    # remove all tokens until the first SEP token
    header, body = tokens[:tokens.index("SEP_None")], tokens[tokens.index("SEP_None")+1:]
    timesig = header[0]
    tempo = header[1]
    # get program tokens
    program_tokens = [t for t in header if t.startswith("Program_")]

    # now get 
    
    return tokens



preview_sm(irma_sm)

display_tokens(irma_tokens)

tokens_hat_sm = tokenizer.decode(tokens_hat)
preview_sm(tokens_hat_sm)

# %%

model_config = Phi3Config(
    vocab_size=tokenizer.vocab_size,
    eos_token_id=tokenizer.vocab["EOS_None"],
    bos_token_id=tokenizer.vocab["BOS_None"],
    pad_token_id=tokenizer.vocab["PAD_None"],
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
# %%





# %%

class MyDataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.infilling_tokens = [
            tokenizer[t] for t in tokenizer.vocab if t.startswith("INF")
        ]
        self.max_seq_len = max_seq_len

    def _random_crop(self, seq):
        return seq[: self.max_seq_len]

    def __call__(self, batch):
        # for each seq in the batch, select a random crop of N tokens
        input_ids = []

        # select a random crop of tokens
        for b in batch:
            # tokens = self.tokenizer._ids_to_tokens(b["token_ids"])
            # program_tokens = [t for t in tokens if t.startswith("Program_")]
            # program_tokens = np.unique(np.random.permutation(program_tokens)).tolist()
            # seq = program_tokens + tokens
            # seq = ["BOS_None"] + seq + ["EOS_None"]
            # ids = self.tokenizer._tokens_to_ids(seq)
            # input_ids.append(torch.LongTensor(ids))
            input_ids.append(torch.LongTensor(prepare_input(b["token_ids"], self.tokenizer)))
        # crop
        input_ids = [self._random_crop(seq) for seq in input_ids]
            
        # pad tokens to max length
        input_ids = self._pad_batch(input_ids)
        
        attention_mask = torch.where(
            input_ids != self.tokenizer.vocab["PAD_None"], 1, 0
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
            batch, batch_first=True, padding_value=self.tokenizer.vocab["PAD_None"]
        ).long()
    

collator = MyDataCollator(tokenizer, MAX_TOKENS+MAX_PROMPT_LENGTH)


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
        save_total_limit=3,
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
