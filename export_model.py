#%%
import transformers
import miditok

domain = "mil"

match domain:
    case "mil":
        # load model 
        ckpt_path = "artefacts/all_runs_2/mil-dataset/aes-iou-0.04-1-100/checkpoint-100"
        TOKENIZER_CONFIG_PATH = "data/tokenizer_config.json"

        # load model
        model = transformers.AutoModelForCausalLM.from_pretrained(ckpt_path)

        # load tokenizer
        miditok_tokenizer_config = miditok.TokenizerConfig.load_from_json(TOKENIZER_CONFIG_PATH)
        miditok_tokenizer = miditok.REMI(miditok_tokenizer_config)

    case "piano":
        # load model 
        ckpt_path = "artefacts/all_runs_2/piano-procedural/aes-0.04-1-100/checkpoint-100"
        model = transformers.AutoModelForCausalLM.from_pretrained(ckpt_path)

        miditok_tokenizer = miditok.REMI.from_pretrained("lucacasini/metamidipianophi3_6L")

    case "piano-long":
        # load model 
        ckpt_path = "artefacts/all_runs_2/piano-long-procedural/aes-0.04-1-100/checkpoint-100"
        model = transformers.AutoModelForCausalLM.from_pretrained(ckpt_path)

        # load tokenizer
        miditok_tokenizer = miditok.REMI.from_pretrained("lucacasini/metamidipianophi3_6L_long")


#%%

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

# Create tokenizer from WordLevel model
wordlevel_tokenizer = Tokenizer(WordLevel(miditok_tokenizer.vocab, unk_token=None))
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

outpath = "demo/model"
model.save_pretrained(outpath)
hf_tokenizer.save_pretrained(outpath)
# save tokenizer

#%%

# %%
