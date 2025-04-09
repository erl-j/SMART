
import pytest
import symusic
import numpy as np
import random
from pathlib import Path
import sys
from datasets import Dataset
from tqdm import tqdm

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent.parent))

# Import directly from the tokenizers module in the root directory
from tokenisation import IrmaTokenizer, IrmaTokenizerConfig

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

@pytest.fixture
def tokenizer_config():
    """Create a TanjaTokenizerConfig for testing."""
    return IrmaTokenizerConfig(
        ticks_per_beat=96,
        positions_per_beat=12,
        tempo_range=(60, 250),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        duration_ranges=((2,12), (16,6))
    )
        
@pytest.fixture
def tokenizer(tokenizer_config):
    """Create a TanjaTokenizer instance for testing."""
    return IrmaTokenizer(tokenizer_config)

@pytest.fixture
def dataset():
    return Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")

def test_tokenizer(tokenizer, dataset):
    # for 10 first examples
    for i in tqdm(range(10)):
        sample = dataset[i]
        midi = symusic.Score.from_midi(sample["midi_bytes"])
        # count notes 
        tokens = tokenizer.midi_to_tokens(midi, shuffle_tracks=False)
        token_ids = tokenizer.midi_to_token_ids(midi, shuffle_tracks=False)
        # print number of event attributes
        midi_hat = tokenizer.tokens_to_midi(tokens)
        tokens_hat = tokenizer.midi_to_tokens(midi_hat, shuffle_tracks=False)
        midi_hat_hat = tokenizer.tokens_to_midi(tokens_hat)
        # check that first token starts with Tempo_
        # make sure the tokens are the same at every position
        for j in range(len(tokens)):
            print(f"Token {j}: {tokens[j]} -> {tokens_hat[j]}")
            assert tokens[j] == tokens_hat[j], f"Expected {tokens[j]} but got {tokens_hat[j]}"