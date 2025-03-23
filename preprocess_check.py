#%%
import torch
from tqdm import tqdm
import pandas as pd

# %%

print(f"Saving datasets to Hugging Face format")
ds_path = "data/gmd_loops_2_tokenized/trn.pt"
records = torch.load(ds_path, weights_only=False)
print(f"Loaded {len(records)} records")
df = pd.DataFrame(records)
# write to hf dataset
import datasets
dataset = datasets.Dataset.from_pandas(df)
dataset.save_to_disk("data/gmd_loops_2_tokenized/trn_hf")

#%%
print(f"Saving datasets to Hugging Face format")
ds_path = "data/gmd_loops_2_tokenized/tst.pt"
records = torch.load(ds_path, weights_only=False)
df = pd.DataFrame(records)
# write to hf dataset
import datasets
dataset = datasets.Dataset.from_pandas(df)
dataset.save_to_disk("data/gmd_loops_2_tokenized/tst_hf")

#%%



#%%


#%%


# %%
df["n_tokens"] = df["token_ids"].apply(len)
# %%
df["n_tokens"].describe()
# %%
# show cumulative distribution
df["n_tokens"].hist(bins=100, range=(0, 2000), cumulative=True, density=True)

#%%

#%%
# decode tokens
from miditok import REMI
from miditok import TokenizerConfig
import miditok

config = miditok.TokenizerConfig.load_from_json("data/gmd_loops_2_tokenized/tokenizer_config.json")
tokenizer = REMI(config)

df["tokens"] = df["token_ids"].apply(lambda x: tokenizer._ids_to_tokens(x))

#%%

df["n_programs"] = df["tokens"].apply(lambda x: len(set([t for t in x if t.startswith("Program_")])))

#%%
df["n_programs"].describe()

#%%
# plot scatter plot of number of tokens vs number of programs
import matplotlib.pyplot as plt
# set x range from 0 to 2000
plt.scatter(df["n_tokens"], df["n_programs"], alpha=0.01, s=0.1)
plt.xlim(0, 2000)
plt.ylim(0, 16)
plt.xlabel("Number of tokens")
plt.ylabel("Number of programs")

plt.show()

#%%

#%%
from util import preview_sm
# shuffle records
import random
random.shuffle(records)
# decode 5 examples and preview
for i in range(5):
    print(f"Example {i}")
    # print number of tokens
    print(f"Number of tokens: {len(records[i]['token_ids'])}")
    preview_sm(tokenizer.decode(records[i]["token_ids"]))
    print("\n")
    tokens = tokenizer._ids_to_tokens(records[i]["token_ids"])
    print(f"Tokens: {tokens}")


# %%
# count how mnay have less than 512 tokens
from transformers import Phi3Config, Phi3ModelForCausalLM

# create data collator
class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        # pad to the maximum length
        max_length = max(len(ex["token_ids"]) for ex in examples)
        padded = [ex["token_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(ex["token_ids"])) for ex in examples]
        return {"input_ids": torch.tensor(padded)}

config = Phi3Config(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=4096,
    bos_token_id=tokenizer.vocab["BOS_None"],
    eos_token_id=tokenizer.vocab["EOS_None"],
    pad_token_id=tokenizer.vocab["PAD_None"],
)

model = Phi3ModelForCausalLM(config)

# create data collator
data_collator = DataCollator(tokenizer)

#%%
# create data loader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset




# 
