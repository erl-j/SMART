#%%
from datasets import Dataset
import miditok

tokenizer_config = miditok.TokenizerConfig.load_from_json("data/tokenizer_config.json")
tokenizer = miditok.REMI(tokenizer_config)

trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")

trn_ds = trn_ds.map(lambda x: {"tokens": tokenizer._ids_to_tokens(x["token_ids"])}, num_proc=8)


#%%
trn_ds = trn_ds.map(lambda x: {"program_set" : sorted(list(set([token for token in x["tokens"] if token.startswith("Program_")]))), "tokens": x["tokens"]}, num_proc=8)


#%%
from tqdm import tqdm
# get how many times each  program set appears in the dataset
program_set_to_count = {}

for example in tqdm(trn_ds):
    if str(example["program_set"]) in program_set_to_count:
        program_set_to_count[str(example["program_set"])] += 1
    else:
        program_set_to_count[str(example["program_set"])] = 1


#%%
# print the program sets in order of frequency
sorted_program_sets = sorted(program_set_to_count.items(), key=lambda x: x[1], reverse=True)
for program_set, count in sorted_program_sets:
    print(program_set, count)



# %%
