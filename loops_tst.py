#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import miditok

checkpoint = "./outputs/mt/treasured-cosmos-19/checkpoint-75000"

tokenizer_config = miditok.TokenizerConfig.load_from_json("./data/tokenizer_config.json")
tokenizer = miditok.REMI(tokenizer_config)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# %%
# generate a sequence
out = model.generate(
    max_length=2048,
    do_sample=True,
    pad_token_id=tokenizer.vocab["PAD_None"],
    bos_token_id=tokenizer.vocab["BOS_None"],
    eos_token_id=tokenizer.vocab["EOS_None"],
    num_return_sequences=1,
    temperature=0.9,
    use_cache=True
    # top_k=1,
    # top_p=0.95,
)

#%%
# decode the sequence
tokens = tokenizer._ids_to_tokens(out[0].tolist())
print(tokens)
sm = tokenizer.decode(out[0])

from util import preview_sm

preview_sm(sm)
# %%
from symusic import BuiltInSF3, Synthesizer
import IPython.display as ipd

SAMPLE_RATE = 41_000
MAX_AUDIO_DURATION = 32
AUDIO_SAVE_INTERVAL = 10

#%%
print(sm.tracks)
def set_drum_to_program(sm):
    # remove expression
    sm = sm.copy()
    for track in sm.tracks:
        if track.is_drum:
            track.program=0
            track.controls=[]
            track.pitch_bends=[]
            track.pedals = []
            for note in track.notes:
                note.duration = sm.tpq
    return sm

sm = set_drum_to_program(sm)

SF_PATH= {"musescore": BuiltInSF3.MuseScoreGeneral().path(download=True), 
            "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
            "monalisa":"./soundfonts/Monalisa_GM_v2_105.sf2",
            "ephesus":"./soundfonts/Ephesus_GM_Version_1_00.sf2",
            "touhou" : "./soundfonts/Touhou.sf2",
            "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
            "fluidr3": "./soundfonts/FluidR3 GM.sf2",
            }


# render with every soundfont
for sf in SF_PATH:
    synth = Synthesizer(
        sf_path = SF_PATH[sf], # the path to the soundfont
        sample_rate = SAMPLE_RATE, # the sample rate of the output wave, sample_rate is the default value
        quality=2
    )
    # print synth
    print(synth)

    audio = synth.render(sm, stereo=True)
    print(f"Rendered audio with {sf} soundfont")
    ipd.display(ipd.Audio(audio, rate=SAMPLE_RATE))



# %%
