#%%
import jax
from synthax.config import SynthConfig
from synthax.synth import ParametricSynth

# Instantiate config
config = SynthConfig(
    batch_size=16,
    sample_rate=44100,
    buffer_size_seconds=4.0
)

# Instantiate synthesizer
synth = ParametricSynth(
    config=config,
    sine=1,
    square_saw=1,
    fm_sine=1,
    fm_square_saw=0
)

# Initialize and run
key = jax.random.PRNGKey(42)
params = synth.init(key)
audio = jax.jit(synth.apply)(params)

# %%
from IPython.display import Audio, display

print(f"Audio shape: {audio.shape}")
display(Audio(audio[0], rate=config.sample_rate))
# %%
