#%%
import transformers
import datasets
import torch
import miditok
from pathlib import Path
# %%
dataset = datasets.load_from_disk(f"../artefacts/dataset_mmd_v6")
tokenizer = miditok.REMI(params=Path("../artefacts/tokenizer_remi_v6.json"))
dataset

# %%
print(tokenizer.vocab_size)
print(tokenizer)
# %%

from transformers import Mamba2Config, Mamba2ForCausalLM
from transformers import Phi3Config, Phi3ForCausalLM


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


# print model params in scientific notation
# print(f"Model has {model.num_parameters()} parameters")
print(f"Model has {model.num_parameters() / 1e6} million parameters")


class MyDataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.infilling_tokens = [
            tokenizer[t] for t in tokenizer.vocab if t.startswith("INF")
        ]
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # for each seq in the batch, select a random crop of N tokens
        input_ids = []        
        # select a random crop of tokens
        for b in batch:
            seq = [self.tokenizer.vocab['BOS_None']] + b['input_ids'] + [self.tokenizer.vocab['EOS_None']]
            start = torch.randint(0, len(seq), (1,)).item()
            end = min(start + self.max_seq_len, len(seq))
            input_ids.append(torch.LongTensor(seq[start:end]))
            
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




# %% train model for 1 epoch
from transformers import Trainer, TrainingArguments
import os
import wandb
from transformers.integrations import WandbCallback
from midi_player import MIDIPlayer
from midi_player.stylers import cifka_advanced

with wandb.init(
    project="aestune",
    job_type="training",
    anonymous="allow",
) as run:

    training_args = TrainingArguments(
        output_dir=f'./outputs/{run.name}',
        max_steps=50_000*3,
        eval_strategy="steps",
        eval_steps=5000,
        # eval_on_start=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500, 
        weight_decay=0.01,
        save_total_limit=1,
        bf16=True,
        # torch_compile=True,
        learning_rate=5e-4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 5e-6},
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset['train'],         # training dataset
        eval_dataset=dataset['test'],             # evaluation dataset
        data_collator=MyDataCollator(tokenizer, 500),   
    )

    # move it up
    class WandbPredictionProgressCallback(WandbCallback):
        def __init__(self, trainer, tokenizer):
            super().__init__()
            self.trainer = trainer
            self.tokenizer = tokenizer
            self.inf_tokens = [tokenizer[t] for t in tokenizer.vocab if t.startswith("INF")]

        """
        take the outputs, try to reoder the sequence and return midi player for wandb
        puts parts in separate tracks if possible
        """

        def display_midi(self, out, title="generation sample"):
            out.dump_midi("./artefacts/tmp.mid")
            return MIDIPlayer(
                "./artefacts/tmp.mid",
                height=400,
                styler=cifka_advanced,
                title=title,
            )

        # render unconditional generation example
        def render_prediction(self):
            out = self.trainer.model.generate(
                    torch.LongTensor([[self.tokenizer.vocab["BOS_None"]]]).to(self.trainer.model.device),
                    max_new_tokens=250,
                    do_sample=True,
                    use_cache=True,
                    # top_p=0.95,
                )
            print("gen_sample", out)
            out = self.tokenizer(out.cpu().tolist())
            mp = self.display_midi(
                out, title=f"Generation sample epoch {round(self.trainer.state.epoch)}"
            )
            wandb.log({"gen_sample": wandb.Html(mp.html)})

        def on_evaluate(self, args, state, control, **kwargs):
            super().on_evaluate(args, state, control, **kwargs)
            # print("after eval callback")
            assert not self.trainer.model.training
            self.render_prediction()


    progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            tokenizer=tokenizer,
        )
    trainer.add_callback(progress_callback)

    trainer.train()

#%%
# from transformers import Phi3ForCausalLM
# import miditok
# import torch
# from midi_player import MIDIPlayer
# from midi_player.stylers import cifka_advanced

# model = Phi3ForCausalLM.from_pretrained("./aestune/results/checkpoint-50000")

# tokenizer = miditok.REMI(params="../artefacts/tokenizer_remi_v6.json")
# seq = model.generate(
#     torch.LongTensor([[tokenizer.vocab["BOS_None"]]]),
#     max_new_tokens=1000,
#     do_sample=True,
#     use_cache=True,
#     top_p=0.975,
#     # repetition_penalty=1.05,
# )
# itos = {v: k for k, v in tokenizer.vocab.items()}
# print( [itos[i] for i in seq[0].tolist()])
      
# score = tokenizer(seq)
# score.dump_midi("./tmp.mid")
# MIDIPlayer(
#         "./tmp.mid",
#         height=400,
#         styler=cifka_advanced,
#     )

# %%
