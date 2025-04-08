
import pytest
import symusic
import numpy as np
import random
from pathlib import Path
import sys
from datasets import Dataset

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
        coarse_ticks_per_beat=24,
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
    for i in range(10):
        sample = dataset[i]
        midi = symusic.Score.from_midi(sample["midi_bytes"])
        tokens = tokenizer.midi_to_tokens(midi, shuffle_events=False)
        # print number of event attributes
        print(f"Number of event attributes: {len(tokenizer.event_attribute_order)}")
        assert len(tokens) == tokenizer.config.n_events * len(tokenizer.event_attribute_order) + 1
        midi_hat = tokenizer.tokens_to_midi(tokens)
        tokens_hat = tokenizer.midi_to_tokens(midi_hat, shuffle_events=False)
        # make sure the tokens are the same
        # find and print diffs
        for a, b in zip(tokens, tokens_hat):
            assert a == b, f"Diff at index {i}: {a} != {b}"


