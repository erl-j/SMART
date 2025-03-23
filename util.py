import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import os
import IPython.display as ipd
import symusic
from midi_player import MIDIPlayer
from midi_player.stylers import basic, cifka_advanced
import IPython.display as ipd
import math
import glob
from tqdm import tqdm

instrument_class_to_selected_program_nr = {
    "Piano": 1,
    "Chromatic Percussion": 12,
    "Organ": 21,
    "Guitar": 25,
    "Bass": 36,
    "Strings": 43,
    "Ensemble": 54,
    "Brass": 61,
    "Reed": 72,
    "Pipe": 80,
    "Synth Lead": 83,
    "Synth Pad": 90,
    "Synth Effects": 103,
    "Ethnic": 108,
    "Percussive": 115,
    "Sound Effects": 112,
}


def detail_plot(sm):
    pr_tpq = 12
    tempo = int(sm.tempos[-1].qpm)
    time_sig = f"{sm.time_signatures[-1].numerator}/{sm.time_signatures[-1].denominator}"
    
    # Get instrument names
    instrument_names = [pretty_midi.program_to_instrument_name(track.program) if not track.is_drum else "Drums" for track in sm.tracks]
    
    # Get pianoroll data
    pr = sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
    
    # Get unique instruments
    unique_instruments = np.unique(instrument_names)

    # Create a mapping from instrument names to list of track indices
    instrument_to_track_indices = {instrument: [] for instrument in unique_instruments}
    for i, instrument in enumerate(instrument_names):
        instrument_to_track_indices[instrument].append(i)
    
    # We know that it's 4 bars so let's crop it
    loop_ticks = pr_tpq * 4 * 4
    
    # Use a colormap with distinguishable colors
    colors = plt.cm.tab10.colors

    # Get drum track indices
    drum_indices = np.where(np.array(instrument_names) == "Drums")[0]
    has_drums = len(drum_indices) > 0
    
    # Get melodic track indices (all non-drum tracks)
    melodic_indices = [i for i, name in enumerate(instrument_names) if name != "Drums"]
    
    # Get all non-zero pitches from melodic instruments to determine the y-axis range
    all_melodic_pitches = []
    for idx in melodic_indices:
        instrument_pr = pr[idx][:, :loop_ticks]
        pitches = np.where(np.any(instrument_pr > 0, axis=1))[0]
        all_melodic_pitches.extend(pitches)
    
    if all_melodic_pitches:
        # Find the min and max pitches used with padding of ±3
        min_pitch = max(0, min(all_melodic_pitches) - 3)
        max_pitch = min(127, max(all_melodic_pitches) + 3)
    else:
        # Default range if no melodic notes found
        min_pitch = 60 - 24
        max_pitch = 60 + 24
    
    # Set up the figure with two subplots if there are drums
    if has_drums:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot melodic instruments on the top subplot
    legend_handles = []
    for i, instrument_name in enumerate(unique_instruments):
        if instrument_name == "Drums":
            continue
        
        # Get indices of tracks with this instrument_name
        track_indices = instrument_to_track_indices[instrument_name]
        
        # Sum all channels with the same program number
        instrument_pr = pr[track_indices].sum(axis=0)
        instrument_pr = instrument_pr[:, :loop_ticks]
        
        # Only show non-zero pitches
        non_zero_indices = np.where(instrument_pr > 0)
        
        # Plot with distinct color
        color = colors[i % len(colors)]
        scatter = ax1.scatter(non_zero_indices[1], non_zero_indices[0], 
                             color=color, marker='s', s=10, 
                             label=instrument_name)
        legend_handles.append(scatter)
    
    # Calculate beat positions
    lines_tpq = 1
    
    # Create tick positions for all data points
    time_ticks = np.arange(0, loop_ticks, lines_tpq * pr_tpq)
    
    # Create matching labels
    time_labels = np.arange(0, len(time_ticks))
    
    # Remove default grid
    ax1.grid(False)

    # Add only vertical lines for each beat
    for tick in time_ticks:
        ax1.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
    
    # Add horizontal lines for each octave
    for octave in range(min_pitch // 12, (max_pitch // 12) + 1):
        pitch = octave * 12
        if min_pitch <= pitch <= max_pitch:
            ax1.axhline(y=pitch, color='gray', linestyle='-', alpha=0.2)
    
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels)
    
    # Set the x and y limits to focus on the actual used pitch range
    ax1.set_xlim(0, loop_ticks)
    ax1.set_ylim(min_pitch, max_pitch)
    
    # Add pitch labels with note names every 12 semitones (C notes)
    pitch_ticks = [pitch for pitch in range(min_pitch - (min_pitch % 12), max_pitch + 12, 12) if min_pitch <= pitch <= max_pitch]
    pitch_labels = [f"C{(pitch // 12) - 1}" for pitch in pitch_ticks]  # C4 is MIDI 60
    ax1.set_yticks(pitch_ticks)
    ax1.set_yticklabels(pitch_labels)
    
    ax1.set_xlabel("Beat")
    ax1.set_ylabel("Pitch")
    ax1.set_title(f"Melodic Instruments - Tempo: {tempo}, Time Signature: {time_sig}")
    
    # Add legend
    ax1.legend(handles=legend_handles, title="Instruments", 
               loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot drums if they exist
    if has_drums:
        # Create mapping for drum pitch to name
        pitch_to_drum_name = {
            pitch: pretty_midi.note_number_to_drum_name(pitch) 
            for pitch in range(0, 128)
            if pretty_midi.note_number_to_drum_name(pitch) is not None
        }
        
        # Get drum pianoroll
        drum_pr = np.zeros_like(pr[0])
        for idx in drum_indices:
            drum_pr += pr[idx]
            
        drum_pr = drum_pr[:, :loop_ticks]
        
        # Get non-zero indices in the drum track
        drum_non_zero = np.where(drum_pr > 0)
        
        # Get unique drum pitches used
        unique_drum_pitches = np.unique(drum_non_zero[0])
        
        # Create a mapping of actual pitch to display position
        pitch_to_position = {pitch: i for i, pitch in enumerate(unique_drum_pitches)}
        
        # Map the pitches to display positions
        display_positions = [pitch_to_position[pitch] for pitch in drum_non_zero[0]]
        
        # Plot drum hits
        ax2.scatter(drum_non_zero[1], display_positions, 
                   color='black', marker='s', s=20)
        
        # Add only vertical lines for each beat (same as above)
        for tick in time_ticks:
            ax2.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
        
        # Set x-axis ticks and limits to match the main plot
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.set_xlim(0, loop_ticks)
        
        # Set y-axis labels with drum names
        ax2.set_yticks(range(len(unique_drum_pitches)))
        drum_labels = [f"{pitch} - {pitch_to_drum_name.get(pitch, 'Unknown')}" 
                      for pitch in unique_drum_pitches]
        ax2.set_yticklabels(drum_labels)
        
        # Set y limits with a bit of padding
        ax2.set_ylim(-0.5, len(unique_drum_pitches) - 0.5)
        
        ax2.set_xlabel("Beat")
        ax2.set_ylabel("Drum Type")
        ax2.set_title("Drum Pattern")
    
    plt.tight_layout()
    plt.show()


def sm_set_track_order(sm):
    track_order = ["Drums"] + list(instrument_class_to_selected_program_nr.keys())
    sm = sm.copy()
    # check out current instrument classes
    existing_instrument_classes = set([pretty_midi.program_to_instrument_class(track.program) for track in sm.tracks])
    if any([t.is_drum for t in sm.tracks]):
        existing_instrument_classes.add("Drums")
    
    # add missing instrument classes
    for instrument_class in track_order:
        if instrument_class not in existing_instrument_classes:
            if instrument_class == "Drums":
                sm.tracks.append(symusic.Track(is_drum=True, program=0))
            else:
                sm.tracks.append(symusic.Track(program=instrument_class_to_selected_program_nr[instrument_class], is_drum=False))
    # sort tracks by instrument class
    # sort tracks by is drum
    sm.tracks = sorted(sm.tracks, key=lambda x: x.is_drum)
    sm.tracks = sorted(sm.tracks, key=lambda x: track_order.index(pretty_midi.program_to_instrument_class(x.program)))
    return sm

def sm_fix_overlap_notes(sm):
    sm = sm.copy()
    for track_idx in range(len(sm.tracks)):
        track = sm.tracks[track_idx]
        notes = track.notes
        # sort notes by start
        notes = sorted(notes,key=lambda x: x.start)
        new_notes = []
        for pitch in range(128):
            new_pitch_notes = []
            pitch_notes = [note for note in notes if note.pitch == pitch]
            if len(pitch_notes) > 0:
                new_pitch_notes.append(pitch_notes[0])
                for i in range(len(pitch_notes)-1):
                    if pitch_notes[i].start == pitch_notes[i+1].start:
                        continue
                    elif pitch_notes[i].end > pitch_notes[i+1].start:
                        pitch_notes[i].duration = pitch_notes[i+1].start - pitch_notes[i].start
                    new_pitch_notes.append(pitch_notes[i+1])
            new_notes.extend(new_pitch_notes)
        sm.tracks[track_idx].notes = new_notes
    return sm

def sm_reduce_dynamics(sm, factor):
    sm = sm.copy()
    for track in sm.tracks:
        for note in track.notes:
            note.velocity = int(note.velocity * factor)
    return sm

def load_merged_models(pattern, model_class):
    print("Instantiating merged model")

    ckpt_paths = glob.glob(pattern,recursive=True)

    models = [model_class.load_from_checkpoint(ckpt_path, map_location="cpu") for ckpt_path in ckpt_paths]

    # make a copy of the last model
    merged_model = model_class.load_from_checkpoint(ckpt_paths[-1], map_location="cpu")

    # set merged model to all zero
    for p in merged_model.parameters():
        p.data = torch.zeros_like(p.data)

    total_params = sum(1 for _ in merged_model.parameters())
    progress_bar = tqdm(total=total_params, desc="Merging parameters")

    # print mean of the model prior to merging
    mean_params = [p.data.mean() for p in merged_model.parameters()]
    print(f"Mean of the model prior to merging: {mean_params}")

    for model in models:
        for p, p_merged in zip(model.parameters(), merged_model.parameters()):
            p_merged.data += p.data/len(models)
            progress_bar.update(1)

    # print mean of the model after merging
    mean_params = [p.data.mean() for p in merged_model.parameters()]
    print(f"Mean of the model after merging: {mean_params}")
    
    progress_bar.close()
    return merged_model

def top_p_probs(probs, top_p=0.0):
    # inputs are probs with shape [N, V]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_indices = sorted_indices.argsort(-1)  # Corrected line
    probs = torch.gather(sorted_probs, 1, sorted_indices)
    # renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def preview_sm(x_sm):
    # rand int
    rand_int = np.random.randint(0, 1000000)
    tmp_file_path = "sm_preview_" + str(rand_int) + ".mid"
    x_sm.dump_midi(tmp_file_path)
    ipd.display(MIDIPlayer(tmp_file_path, 400, styler=cifka_advanced, title='My Player'))
    # delete file
    os.remove(tmp_file_path)


def has_drum(sm):
    for track in sm.tracks:
        if track.is_drum and len(track.notes) > 0:
            return True
    return False

def has_harmonic(sm):
    for track in sm.tracks:
        if not track.is_drum and len(track.notes) > 0:
            return True
    return False
            

def get_sm_pitch_range(sm):
    pitches = []
    for track in sm.tracks:
        if not track.is_drum:
            for note in track.notes:
                pitches.append(note.pitch)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    return min_pitch, max_pitch

def render_directory_with_fluidsynth(midi_dir, audio_dir, overwrite=False):
    os.makedirs(audio_dir, exist_ok=True)
    for midi_file in os.listdir(midi_dir):
        midi_path = midi_dir + "/" + midi_file
        audio_path = audio_dir + "/" + midi_file.replace(".mid", ".wav")
        # open with symusic
        sm = symusic.Score(midi_path)


        # 16 beats in 4/4
        end_4_bars_tick = 16 * sm.ticks_per_quarter
        # assert sm.end() <= end_4_bars_tick
        # crop midi file to 4 bars
        for track in sm.tracks:
            for note in track.notes:
                    note.duration = min(end_4_bars_tick - note.start, note.duration)
        print(f"end of midi file: {sm.end()}")

        tmp_path = midi_path
        tmp_path = tmp_path.replace(".mid", "_cropped.mid")
        tmp_path = tmp_path.replace(midi_dir, audio_dir)
        sm.dump_midi(tmp_path)
        
        # assert sm.end()<40, "midi file is too long"
        # crop to 4 bars

        # midi_path = tmp_path


        if not os.path.exists(audio_path) or overwrite:
            os.system(f"fluidsynth {tmp_path} -F {audio_path}")


def preview(sm, tmp_dir, audio=True):
    # SAMPLE_RATE = 44_100
    os.makedirs(tmp_dir, exist_ok=True)
    midi_path = tmp_dir + "/tmp.mid"
    audio_path = tmp_dir + "/output.wav"
    sm.dump_midi(midi_path)
    pr = piano_roll(sm, tpq=16)
    plt.figure(figsize=(10, 10))
    sns.heatmap(pr, cmap="magma")
    plt.show()

    if audio:
        os.system(f"fluidsynth {midi_path} -F {audio_path}")
        ipd.display(ipd.Audio(audio_path))


def get_scale(scale, range):
    root = scale.split(" ")[0]
    mode = scale.split(" ")[1]

    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "pentatonic": [0, 2, 4, 7, 9],
        "blues": [0, 3, 5, 6, 7, 10],
        "kaweco":[0,1,5,7]
    }

    if root not in [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]:
        raise ValueError("Root not found")

    if mode not in scales:
        raise ValueError(f"Mode not found, options are {scales.keys()}")

    root_midi = pretty_midi.note_name_to_number(root + "0")

    while root_midi < range[0]:
        root_midi += 12

    midi_notes = []

    octave = 0
    while True:
        for interval in scales[mode]:
            new_note = octave * 12 + root_midi + interval
            if new_note >= range[1]:
                return midi_notes
            else:
                midi_notes.append(new_note)
        octave += 1


def crop_sm(sm, n_bars, beats_per_bar):
    sm = sm.copy()
    end_tick = n_bars * sm.ticks_per_quarter * beats_per_bar
    for track_idx in range(len(sm.tracks)):
        track = sm.tracks[track_idx]
        new_notes = []
        for note in track.notes:
            if note.start < end_tick:
                # crop duration
                if note.start + note.duration > end_tick:
                    note.duration = end_tick - note.start
                new_notes.append(note)
        sm.tracks[track_idx].notes = new_notes
    return sm

def loop_sm(sm, loop_bars, n_loops):
    '''
    4/4 only
    '''
    sm = sm.copy()
    # get resolution
    tpq = sm.ticks_per_quarter

    # get duration of a bar in ticks
    loop_duration = tpq * 4 * loop_bars
    
    for track in sm.tracks:
        new_notes = []
        for loop_idx in range(1,n_loops):
            for note in track.notes:
                note = note.copy()
                note.start = note.start + loop_duration*(loop_idx)
                new_notes.append(note)
        track.notes.extend(new_notes)
    return sm

def piano_roll(sm, tpq, include_drums):
    sm = sm.copy()
    sm = sm.resample(tpq=tpq, min_dur=0)

    # if include drums, set all tracks to not be drums
    if include_drums:
        for track in sm.tracks:
            track.is_drum = False
    else:
        # remove drum tracks
        sm.tracks = [track for track in sm.tracks if not track.is_drum]

    pr = sm.pianoroll(modes=["frame"]).sum(axis=0).sum(axis=0)

    # flip y axis
    pr = pr[::-1]

    return pr

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    # from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    return logits