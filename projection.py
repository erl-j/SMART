#%%
import umap
from pathlib import Path
from datasets import load_dataset
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.utils import get_bars_ticks
from symusic import Score
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm
from symusic import Synthesizer, BuiltInSF3
import symusic
from tqdm import tqdm

#%%


N_EXAMPLES = 5000
MAX_AUDIO_DURATION = 20
CUDA_DEVICE = 3

clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(CUDA_DEVICE)
clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
sample_rate = 48_000

def get_clap_features(audio_samples):
    inputs = clap_processor(audios=audio_samples.mean(0)[None,...], return_tensors="pt", sampling_rate=sample_rate).to(CUDA_DEVICE)
    audio_embed = clap_model.get_audio_features(**inputs)
    return audio_embed

#%%


def is_score_valid(
    score, min_num_bars: int, min_num_notes: int
) -> bool:
    """
    Check if a ``symusic.Score`` is valid, contains the minimum required number of bars.

    :param score: ``symusic.Score`` to inspect or path to a MIDI file.
    :param min_num_bars: minimum number of bars the score should contain.
    :param min_num_notes: minimum number of notes that score should contain.
    :return: boolean indicating if ``score`` is valid.
    """
    if isinstance(score, Path):
        try:
            score = Score(score)
        except SCORE_LOADING_EXCEPTION:
            return False
    elif isinstance(score, bytes):
        try:
            score = Score.from_midi(score)
        except SCORE_LOADING_EXCEPTION:
            return False

    return (
        len(get_bars_ticks(score)) >= min_num_bars and score.note_num() > min_num_notes
    )

dataset = load_dataset("Metacreation/GigaMIDI", split="validation")
dataset = dataset.filter(
    lambda ex: is_score_valid(ex["music"], min_num_bars=8, min_num_notes=50)
)

#%%
# keep only those with genre labels
dataset = dataset.filter(lambda ex: len(ex["genres_lastfm"]["genre"])>0)
print(f"Size of dataset: {len(dataset)}")

#%% 
# turn dataset into records
records = [{**ex} for ex in tqdm(dataset)]

#%% add lastfm first genre
records = [{**record, "genre_lastfm": record["genres_lastfm"]["genre"][0]} for record in records]

# limit records to 100 examples
if N_EXAMPLES < len(records):
    records = records[:N_EXAMPLES]


#%%

sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)
synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)

for record in tqdm(records):
    try:
        score = Score.from_midi(record["music"])
        score = score.to(symusic.TimeUnit.second)
        # crop score
        score = score.clip(0,30, clip_end=True)
        audio = synth.render(score)
        if audio.shape[1] > MAX_AUDIO_DURATION*sample_rate:
            audio = audio[:,-MAX_AUDIO_DURATION*sample_rate:]
        record["audio"] = audio
    except Exception as e:
        print(f"Error rendering audio: {e}")
        record["audio"] = None


#%%

# add clap embeddings
for record in tqdm(records):
    try:
        record["clap_features"] = get_clap_features(record["audio"]).detach().cpu().numpy()
    except Exception as e:
        print(f"Error getting clap features: {e}")
        record["clap_features"] = None
#%%

# %%

# plot first 3 waveforms
import matplotlib.pyplot as plt
import IPython.display as ipd

for i in range(3):
    plt.figure(figsize=(10, 2))
    plt.plot(records[i]["audio"][0])
    plt.show()
    ipd.display(ipd.Audio(records[i]["audio"][0], rate=sample_rate))
    # plot clap features
    if records[i]["clap_features"] is not None:
        plt.figure(figsize=(10, 2))
        plt.plot(records[i]["clap_features"][0])
        plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt

# fit a umap model on clap features
clap_features = [record["clap_features"][0] for record in records if record["clap_features"] is not None]
clap_features = np.array(clap_features)

# get genre labels
genre_labels = [record["genre_lastfm"] for record in records if record["clap_features"] is not None]



#%% 
# do pca down to 2 dimensions
from sklearn.decomposition import PCA

# create a pipeline which first mean centers the data (dont change the variance) and then does PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


pca_model = Pipeline([
    ("normalizer", Normalizer()),
    ("pca", PCA(n_components=2))
])

pca_features = pca_model.fit_transform(clap_features)

# show only genres rock pop and classical
plt.figure(figsize=(10,10))
for genre in set(genre_labels):
    indices = [i for i, g in enumerate(genre_labels) if g == genre]
    plt.scatter(pca_features[indices,0], pca_features[indices,1], label=genre)
plt.legend()
plt.show()


from collections import Counter
# show top 5 genres
plt.figure(figsize=(10,10))
# get top 5 genres
top_genres = [genre for genre, count in Counter(genre_labels).most_common(10)]
for genre in top_genres:
    indices = [i for i, g in enumerate(genre_labels) if g == genre]
    plt.scatter(pca_features[indices,0], pca_features[indices,1], label=genre)
plt.legend()
plt.show()

# %%

# same with umap.
# scale to unit length first

umap_model = Pipeline([
    # first scale to unit length
    ("scaler", Normalizer()),
    ("umap", umap.UMAP(n_components=2, metric="cosine", n_neighbors=5))
])

umap_features = umap_model.fit_transform(clap_features)

# plot scatter plot, colour by genre
plt.figure(figsize=(10,10))
for genre in set(genre_labels):
    indices = [i for i, g in enumerate(genre_labels) if g == genre]
    plt.scatter(umap_features[indices,0], umap_features[indices,1], label=genre)
plt.legend()
plt.show()

#%%
# show only genres rock pop and classical
plt.figure(figsize=(10,10))
for genre in ["metal","classical", "hiphop","experimental", "electronic"]:
    indices = [i for i, g in enumerate(genre_labels) if g == genre]
    plt.scatter(umap_features[indices,0], umap_features[indices,1], label=genre)
plt.legend()
plt.show()

#%% 
from collections import Counter
# show top 5 genres
plt.figure(figsize=(10,10))
# get top 5 genres
top_genres = [genre for genre, count in Counter(genre_labels).most_common(5)]
for genre in top_genres:
    indices = [i for i, g in enumerate(genre_labels) if g == genre]
    plt.scatter(umap_features[indices,0], umap_features[indices,1], label=genre)
plt.legend()
plt.show()


# %%
