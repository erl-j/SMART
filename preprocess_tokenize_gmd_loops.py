#%%

from pathlib import Path
from datasets import load_dataset
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.utils import get_bars_ticks
from symusic import Score
import miditok
from tqdm import tqdm
import torch

TS_MAPS = {
    # 16: [15, 11, 7, 4, 3, 12, 6],
    8: [3, 5, 6, 7, 9, 10, 11, 12],
    4: [2, 3, 4, 5, 6],
    2: [2, 3],
}
BEAT_RES = {(0, 2): 12, (2, 4): 12}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "beat_res_rest": BEAT_RES,
    "special_tokens": [
        "PAD",
        "BOS",
        "EOS",
        "SEP",
    ],
    "use_chords": False,
    # "chord_maps": CHORD_MAPS,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "time_signature_range": TS_MAPS,
    "use_programs": True,
    "use_sustain_pedals": False,
    "one_token_stream_for_programs": True,
    "use_pitchdrum_tokens": True,
    "remove_duplicated_notes": True,
    "num_tempos": 64,  # nb of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = miditok.TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = miditok.REMI(config)

# save config
config.save_to_json("data/tokenizer_config.json")

#%% 
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import datasets
from datasets import Dataset

#%%

    
def music2tokens(example, idx):
    """Process a single MIDI example"""
    processed = {}
    processed['token_ids'] = tokenizer(example['midi']).ids
    processed["loop_idx"] = idx
    processed["midi_bytes"] = example["midi"].dumps_midi()
    # Remove memory-intensive fields
    processed["midi"] = None
    processed["music"] = None
    return processed
    
   
# Main execution
# Define paths
output_dir = "data/gmd_loops_2_tokenized_2"
os.makedirs(output_dir, exist_ok=True)

# Import your tokenizer here
# from your_tokenizer_module import tokenizer
import pandas as pd

# Process each split
for split in ["tst", "val", "trn"]:
    src_path = f"../slm/data/gmd_loops_2/{split}_midi_records_loops.pt"
    dataset = torch.load(src_path, weights_only=False)

    # Process each example
    dataset = Parallel(n_jobs=16)(
        delayed(music2tokens)(subrecord, idx)
        for record in tqdm(dataset, desc=f"Processing {split}")
        for idx, subrecord in enumerate(record)
    )
    dataset = pd.DataFrame(dataset)
    # 
    # save the dataset
    dataset = Dataset.from_pandas(dataset)
    dataset.save_to_disk(f"{output_dir}/{split}")


print("All splits processed and saved successfully!")
#%%


# %%
