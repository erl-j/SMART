#%%
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM
import miditok

# checkpoint = "outputs/mt/silvery-forest-28/copies/checkpoint-850000"
checkpoint = "artefacts/all_runs_5/irma-dataset/ce-sc-iou-0.04-1.0-1000-linear-scale-rewards=True/checkpoint-1000"
from tokenisation import IrmaTokenizer, IrmaTokenizerConfig
# %%

tokenizer = IrmaTokenizer(
    IrmaTokenizerConfig(
        ticks_per_beat=96,
        positions_per_beat=12,
        tempo_range=(60, 250),
        n_tempo_bins=32,
        n_velocity_bins=32,
        n_bars=4,
        duration_ranges=((2,12), (16,6))
    )
)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

#%%

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

# Create tokenizer from WordLevel model
wordlevel_tokenizer = Tokenizer(WordLevel(tokenizer.token_to_idx, unk_token=None))
wordlevel_tokenizer.pre_tokenizer = WhitespaceSplit()

# Wrap it with Hugging Face
hf_tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=wordlevel_tokenizer,
    pad_token="PAD_None",
    bos_token="BOS_None",
    eos_token="EOS_None",
    sep_token="SEP_None",
)

print("Tokenizer vocab size: ", hf_tokenizer.vocab_size)

test_input = "BOS_None Program_0"

outpath = "demo/irma_model_SMART"
model.save_pretrained(outpath)
hf_tokenizer.save_pretrained(outpath)
# save tokenizer

#%%

# %%
