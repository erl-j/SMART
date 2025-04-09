
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
from tokenisation import TanjaTokenizer, TanjaTokenizerConfig

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

@pytest.fixture
def tokenizer_config():
    """Create a TanjaTokenizerConfig for testing."""
    return TanjaTokenizerConfig(
        ticks_per_beat=96,
        coarse_ticks_per_beat=12,
        tempo_range=(60, 200),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        n_events=300,
    )

@pytest.fixture
def tokenizer(tokenizer_config):
    """Create a TanjaTokenizer instance for testing."""
    return TanjaTokenizer(tokenizer_config)

@pytest.fixture
def dataset():
    return Dataset.load_from_disk("data/gmd_loops_2_tokenized_2/trn_subset")

def test_tokenizer(tokenizer, dataset):
    # for 10 first examples
    for i in tqdm(range(10)):
        sample = dataset[i]
        midi = symusic.Score.from_midi(sample["midi_bytes"])
        tokens = tokenizer.midi_to_tokens(midi, shuffle_events=False)
        token_ids = tokenizer.midi_to_token_ids(midi, shuffle_events=False)
        tokens_test = tokenizer.ids_to_tokens(token_ids)
        token_ids_test = tokenizer.tokens_to_ids(tokens)
        # check that the tokens are the same
        for j in range(len(tokens)):
            assert tokens[j] == tokens_test[j], f"Expected {tokens[j]} but got {tokens_test[j]}"
            assert token_ids[j] == token_ids_test[j], f"Expected {token_ids[j]} but got {token_ids_test[j]}"

        # check that first token starts with Tempo_
        assert tokens[0].startswith("Tempo_"), f"Expected Tempo_ but got {tokens[0]}"
        # print number of event attributes
        print(f"Number of event attributes: {len(tokenizer.event_attribute_order)}")
        assert len(tokens) == tokenizer.config.n_events * len(tokenizer.event_attribute_order) + 1
        midi_hat = tokenizer.tokens_to_midi(tokens)
        tokens_hat = tokenizer.midi_to_tokens(midi_hat, shuffle_events=False)
        midi_hat_hat = tokenizer.tokens_to_midi(tokens_hat)
        # check that first token starts with Tempo_
        assert tokens_hat[0].startswith("Tempo_"), f"Expected Tempo_ but got {tokens_hat[0]}"
        assert len(tokens_hat) == tokenizer.config.n_events * len(tokenizer.event_attribute_order) + 1

        tokens_hat_hat = tokenizer.midi_to_tokens(midi_hat_hat, shuffle_events=False)
        # make sure the tokens are the same at every position
        for j in range(len(tokens)):
            print(f"Token {j}: {tokens_hat_hat[j]} -> {tokens_hat[j]}")
            assert tokens_hat_hat[j] == tokens_hat[j], f"Expected {tokens_hat_hat[j]} but got {tokens_hat[j]}"


