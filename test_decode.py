#%%
import miditok

# load tokenizer
miditok_tokenizer = miditok.REMI.from_pretrained("lucacasini/metamidipianophi3_6L_long")

a = miditok_tokenizer.encode("data/toktest2.mid")

dur_tokens = [t for t in a[0].tokens if t.startswith("Duration_")]
print(dur_tokens)

[
'Duration_4.0.12', # 4 bars
'Duration_4.0.12', # 2 bars
'Duration_4.0.12', # 1 bar
'Duration_2.0.12', # 2 beats
'Duration_1.0.12', # 1 beat
'Duration_0.6.12', # 1/2 beat
'Duration_0.3.12', # 1/4 beat
'Duration_0.2.12',
'Duration_0.1.12'
]



# %%
# read input string
input_string = input("Enter a string (or 'exit' to quit): ")
# check for exit condition
# split on whitespace
tokens = input_string.split()
# convert tokens to ids
token_ids = miditok_tokenizer._tokens_to_ids(tokens)


import numpy as np

sm = miditok_tokenizer.decode(np.array(token_ids)[None,...])

from util import preview_sm

preview_sm(sm)

# %%
