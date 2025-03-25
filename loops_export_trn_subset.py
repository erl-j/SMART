from datasets import Dataset
trn_ds = Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn")

SUBSET_SIZE = 200_000

# export a random subset of the training data
trn_ds = trn_ds.shuffle()
trn_ds = trn_ds.select(range(SUBSET_SIZE))

# print subset size
print(len(trn_ds))

# save the subset
trn_ds.save_to_disk("data/gmd_loops_2_tokenized_2/trn_subset")