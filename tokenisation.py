import symusic
import pretty_midi
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, TypeVar, Generic, Type
import json
import random
import logging

logger = logging.getLogger(__name__)

# Constants
MIDI_DRUM_PITCHES = range(22, 82)

class Quantizer:
    def __init__(
        self, value_range: Tuple[float, float], n_bins: int, round_values: bool = False
    ):
        self.range = value_range
        self.n_bins = n_bins
        self.bins = np.linspace(value_range[0], value_range[1], n_bins)
        if round_values:
            self.bins = np.round(self.bins).astype(int)

    # returns float or int
    def quantize(self, value: float):
        """Returns the closest bin value for a given input."""
        return self.bins[np.argmin(np.abs(self.bins - value))]

@dataclass
class TokenizerConfig:
    pass

# Create a type variable bound to TokenizerConfig
T = TypeVar("T", bound=TokenizerConfig)


class BaseTokenizer(Generic[T]):
    """Abstract base class for MIDI tokenizers."""

    config_cls: Type[T]  # Type annotation for the class variable

    def __init__(self, config: T) -> None:
        self.config = config
        self.vocab: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.pad_token_id = -1

    def to_json(self, path: str) -> None:
        """Save the tokenizer configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        """Load the tokenizer configuration from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        config = cls.config_cls(**config_dict)
        return cls(config)

    def midi_to_tokens(self, midi: symusic.Score) -> List[str]:
        raise NotImplementedError

    def tokens_to_midi(self, tokens: List[str]) -> symusic.Score:
        raise NotImplementedError

    def ids_to_midi(self, ids: List[int]) -> symusic.Score:
        return self.tokens_to_midi(self.ids_to_tokens(ids))

    def midi_to_ids(self, midi: symusic.Score) -> List[int]:
        return self.tokens_to_ids(self.midi_to_tokens(midi))

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to their corresponding indices."""
        return [self.token_to_idx[token] for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert indices back to tokens."""
        return [self.vocab[idx] for idx in ids]
    

class IRMATokenizerConfig(TokenizerConfig):
    ticks_per_beat: int
    positions_per_beat : int
    tempo_range: Tuple[int, int]
    n_tempo_bins: int
    n_velocity_bins: int
    n_bars : int
    duration_ranges: List[Tuple[int, int]]

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
    

@dataclass
class TanjaTokenizerConfig(TokenizerConfig):
    ticks_per_beat: int
    coarse_ticks_per_beat: int
    tempo_range: Tuple[int, int]
    n_tempo_bins: int
    n_velocity_bins: int
    n_bars : int
    n_events : int
    
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
    
class TanjaTokenizer(BaseTokenizer):

    '''
    CMLM Tokenizer.
    This tokenizer outputs a list of tokens in the following format:
    # First tempo is provided.
    Tempo
    # Then for each note we have 7 attributes.
    Program Pitch OnsetCoarse OnsetFine Offset Duration Velocity
    # There is also a mask token
    '''

    def __init__(self, config: TanjaTokenizerConfig):

        self.config = config
        self.vocab = []


        self.n_beats = config.n_bars * 4

        # add MASK_None token
        self.vocab.append("MASK_None")

        # add SEP_None token
        self.vocab.append("SEP_None")

        # first create tempo quantizer
        self.tempo_quantizer = Quantizer(
            config.tempo_range, config.n_tempo_bins, round_values=True
        )
        # add tempo tokens
        self.vocab.extend(f"Tempo_{tempo}" for tempo in self.tempo_quantizer.bins)

        # now add program tokens
        for i in range(128):
            self.vocab.append(f"Program_{i}")

        # add program for drums
        self.vocab.append(f"Program_Drums")

        # add inactive state for program
        self.vocab.append(f"Program_inactive")

        # now add pitch tokens
        self.vocab.extend(f"Pitch_{pitch}" for pitch in range(128))

        # add pitch tokens for drums
        self.vocab.extend(f"Pitch_Drum{pitch}" for pitch in range(128))

        # add inactive state for pitch
        self.vocab.append(f"Pitch_inactive")

        # now add coarse onset tokens
        self.vocab.extend(f"Onset_{i}" for i in range(self.n_beats * self.config.ticks_per_beat // config.coarse_ticks_per_beat))
        # add inactive state for onset
        self.vocab.append(f"Onset_inactive")

        # add onset micro
        self.vocab.extend(f"OnsetMicro_{i}" for i in range(self.config.coarse_ticks_per_beat))
        # add inactive state for onset micro
        self.vocab.append(f"OnsetMicro_inactive")

        # now add offset tokens
        self.vocab.extend(f"Offset_{i}" for i in range(self.n_beats * self.config.ticks_per_beat // config.coarse_ticks_per_beat))
        # add inactive state for offset
        self.vocab.append(f"Offset_inactive")

        # now add duration tokens
        # we use fractions from 1/32 to 4/1, in powers of 2
        # 32 ticks
        thirtysecond_ticks = (self.config.ticks_per_beat * 4) // 32
        fourbar_ticks = (self.config.ticks_per_beat * self.n_beats)

        ticks = thirtysecond_ticks
        while ticks <= fourbar_ticks:
            # add duration token
            self.vocab.append(f"Duration_{ticks}")
            # multiply by 2
            ticks *= 2

        self.durations = [int(t.split("_")[-1]) for t in self.vocab if t.startswith("Duration_")]
        
        # add inactive state for duration
        self.vocab.append(f"Duration_inactive")
        
        # then create velocity quantizer
        self.velocity_quantizer = Quantizer(
            (1, 127), config.n_velocity_bins, round_values=True
        )
        # add velocity tokens
        self.vocab.extend(f"Velocity_{v}" for v in self.velocity_quantizer.bins)
        # add inactive state for velocity
        self.vocab.append(f"Velocity_inactive")

        self.event_attribute_order = [
            "Program",
            "Pitch",
            "OnsetMicro",
            "OnsetFine",
            "Offset",
            "Duration",
            "Velocity",
        ]

    def get_inactive_note_tokens(self):
        # get inactive note attributes
        program_token = f"Program_inactive"
        pitch_token = f"Pitch_inactive"
        onset_coarse_token = f"Onset_inactive"
        onset_fine_token = f"OnsetMicro_inactive"
        offset_token = f"Offset_inactive"
        duration_token = f"Duration_inactive"
        velocity_token = f"Velocity_inactive"
        # create note dict
        return [program_token, pitch_token, onset_coarse_token, onset_fine_token, offset_token, duration_token, velocity_token]


    def get_note_tokens(self, note, program, is_drums):
        # get note attributes
        program_token = f"Program_{program}" if not is_drums else f"Program_Drums"
        pitch_token = f"Pitch_{note.pitch}" if not is_drums else f"Pitch_Drum{note.pitch}"
        onset_coarse_token = f"Onset_{int(note.start // self.config.coarse_ticks_per_beat)}"
        onset_fine_token = f"OnsetMicro_{int(note.start % self.config.coarse_ticks_per_beat)}"
        offset_token = f"Offset_{int(note.end // self.config.coarse_ticks_per_beat)}"
        duration_token = f"Duration_{min(self.durations, key=lambda x: abs(x - note.end + note.start))}"
        velocity_token = f"Velocity_{self.velocity_quantizer.quantize(note.velocity)}"
        # create note dict
        return [program_token, pitch_token, onset_coarse_token, onset_fine_token, offset_token, duration_token, velocity_token]

    def midi_to_tokens(self, midi, shuffle_events=True):
        # first resample the midi to the ticks per beat
        midi = midi.copy().resample(self.config.ticks_per_beat)
        # assert that the time signature is 4/4
        time_signature = midi.time_signatures[-1]
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            raise ValueError(
                "Only 4/4 time signature is supported for Tanja tokenizer."
            )
        # get tempo
        tempo = midi.tempos[-1].qpm if len(midi.tempos) > 0 else 120
        # quantize tempo
        tempo_token = f"Tempo_{self.tempo_quantizer.quantize(tempo)}"
        note_tokens = []
        for track in midi.tracks:
            is_drums = track.is_drum
            program_nr = track.program
            for note in track.notes:
                # get note attributes
                note_tokens.append(self.get_note_tokens(note, program_nr, is_drums))
        # shuffle note tokens
        # we'll 
        n_inactive_notes = self.config.n_events - len(note_tokens)
        # add inactive notes
        for i in range(n_inactive_notes):
            note_tokens.append(self.get_inactive_note_tokens())
        if shuffle_events:
            note_tokens = random.sample(note_tokens, len(note_tokens))

        def flatten(lst):
            return [item for sublist in lst for item in sublist]
        # now we have the note tokens, we can create the final token list
        tokens = [tempo_token, *flatten(note_tokens)]
        return tokens
    
    def tokens_to_midi(self, tokens):
        # create score
        midi = symusic.Score()

        # set tempo
        tempo_token = tokens.pop(0)
        tempo = int(tempo_token.split("_")[-1])
        midi.tempos = [symusic.Tempo(qpm=tempo, time=0)]

        # set time signature
        midi.time_signatures.append(symusic.TimeSignature(numerator=4, denominator=4, time=0))

        program_notes = {}

        while len(tokens) > 0:
            # pop len(self.event_attribute_order) tokens
            note_tokens = tokens[:len(self.event_attribute_order)]
            tokens = tokens[len(self.event_attribute_order):]

            # get note attributes
            program_token = note_tokens[0]
            # assert that this is a program token
            assert program_token.startswith("Program_"), "First token must be a program token"
            program_str = program_token.split("_")[-1]
            if program_str == "inactive":
                continue
            program = int(program_str) if program_str != "Drums" else -1
            is_drum = program_str == "Drums"
            pitch_token = note_tokens[1]
            # assert that this is a pitch token
            assert pitch_token.startswith("Pitch_"), "Second token must be a pitch token"
            pitch_str = pitch_token.split("_")[-1]
            pitch = int(pitch_str) if "Drum" not in pitch_str else int(pitch_str.split("Drum")[-1])
            # get onset coarse token
            onset_coarse_token = note_tokens[2]
            # assert that this is a onset token
            assert onset_coarse_token.startswith("Onset_"), "Third token must be an onset token"
            onset_coarse_str = onset_coarse_token.split("_")[-1]
            onset_coarse = int(onset_coarse_str)
            # get onset fine token
            onset_fine_token = note_tokens[3]
            # assert that this is a onset token
            assert onset_fine_token.startswith("OnsetMicro_"), "Fourth token must be an onset token"
            onset_fine_str = onset_fine_token.split("_")[-1]
            onset_fine = int(onset_fine_str)
            # get offset token
            offset_token = note_tokens[4]
            # assert that this is a offset token
            assert offset_token.startswith("Offset_"), "Fifth token must be an offset token"
            offset_str = offset_token.split("_")[-1]
            offset = int(offset_str)
            # get duration token
            duration_token = note_tokens[5]
            # assert that this is a duration token
            assert duration_token.startswith("Duration_"), "Sixth token must be a duration token"
            duration_str = duration_token.split("_")[-1]
            duration = int(duration_str)
            # get velocity token
            velocity_token = note_tokens[6]
            # assert that this is a velocity token
            assert velocity_token.startswith("Velocity_"), "Seventh token must be a velocity token"
            velocity_str = velocity_token.split("_")[-1]
            velocity = int(velocity_str)
            # create note
            if program not in program_notes:
                program_notes[program] = []
            
            program_notes[program].append(
                symusic.Note(
                    time=onset_coarse * self.config.coarse_ticks_per_beat + onset_fine,
                    pitch=pitch,
                    velocity=velocity,
                    duration = max((offset * self.config.coarse_ticks_per_beat + onset_fine) - (onset_coarse * self.config.coarse_ticks_per_beat + onset_fine),1),
                )
            )
        # now sort programs by program number
        program_notes = sorted(program_notes.items(), key=lambda x: x[0])
        # sort program notes by start time, end time, pitch, velocity
        for program, notes in program_notes:
            notes.sort(key=lambda note: (note.start, note.end, note.pitch, note.velocity))
        # now create tracks for each program
        for program, notes in program_notes:
            # create a new track
            track = symusic.Track(is_drum=program == -1, program=program if program != -1 else 0)
            # add notes to track
            for note in notes:
                track.notes.append(note)
            # add track to midi
            midi.tracks.append(track)
        return midi    

# header
# Tempo_120 Program_Drums Program_1 Program_34 Program_1 Track_None Bar_None Position_0 Offset_2 Pitch_Drum:60 Velocity ... Track_None Bar_None Postion_0 Offset_2 Pitch_60 Velocity_100 Duration_46 ... ... Track_None   

class IRMATokenizer(BaseTokenizer):
    '''
    IRMA Tokenizer.
    Starts with a header that contains the time signature and tempo.
    Then, it contains the programs that will be involved (in arbitrary order).
    Then, the body starts.
    The body has one part per program, separated by the separator token.
    A body part is structured as follows:
    Track_None Program_0 BAR_None Position_12 Offset_2 Pitch_60 Velocity_100 Duration_1.1.12 ...
    Track_None ... 
    We can have multiple tracks per program.
    Offset is only present if needed.
    Only supports 4/4 time signature.
    '''

    def __init__(self, config: IRMATokenizerConfig):
        super().__init__(config)


        self.ticks_per_position = self.config.ticks_per_beat / self.config.positions_per_beat

        self.vocab = []
        # Special tokens
        self.vocab.append("BOS_None")
        self.vocab.append("EOS_None")
        self.vocab.append("SEP_None")
        self.vocab.append("PAD_None")  
        self.vocab.append("Bar_None")
        self.vocab.append("Track_None")

        # now add tempo tokens
        self.tempo_quantizer = Quantizer(
            self.config.tempo_range, self.config.n_tempo_bins, round_values=True
        )
        self.vocab.extend(f"Tempo_{tempo}" for tempo in self.tempo_quantizer.bins)
    
        # Now add program tokens
        for i in range(128):
            self.vocab.append(f"Program_{i}")
        # add program for drums
        self.vocab.append(f"Program_Drums")

        # Now add position tokens
        positions_per_bar = 4 * config.positions_per_beat
        for i in range(positions_per_bar):
            self.vocab.append(f"Position_{i}")

        # Now add offset tokens
        n_offsets = config.ticks_per_beat / config.positions_per_beat
        for i in range(1, int(n_offsets)):
            self.vocab.append(f"Offset_{i}")

        # Now add pitch tokens
        self.vocab.extend(f"Pitch_{pitch}" for pitch in range(128))

        # now add drum pitch tokens
        self.vocab.extend(f"Pitch_Drum{pitch}" for pitch in range(128))

        # Now add duration tokens
        # durations operate as follows.
        # if between 0 and 1, it is a note
        # 
        example_dur_ranges = [(1, 24), (2, 12), (4, 8), (16, 4)]
        # assert that all durations divisions are divisors of 96
        for dur_range in self.config.duration_ranges:
            assert dur_range[1] % self.config.ticks_per_beat == 0, "Duration division must be a divisor of ticks_per_beat"

        range_start = 0

        self.durations = []
        for dur_range in self.config.duration_ranges:
            range_end = dur_range[0]
            # add all durations between range_start and range_end
            range_start_ticks = range_start * self.config.ticks_per_beat
            range_end_ticks = range_end * self.config.ticks_per_beat
            dur_skip_ticks = self.config.ticks_per_beat / dur_range[1]
            for i in range(range_start_ticks, range_end_ticks, int(dur_skip_ticks)):
                self.vocab.append(f"Duration_{i}d{self.config.ticks_per_beat*4}")
                self.durations.append(i)
            range_start = range_end


         # Now add velocity tokens
        self.velocity_quantizer = Quantizer(
            (1, 127), self.config.n_velocity_bins, round_values=True
        )
        self.vocab.extend(f"Velocity_{v}" for v in self.velocity_quantizer.bins)

        # Create token to index mapping
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}

    def midi_to_tokens(self, midi: symusic.Score) -> List[str]:
        """Convert a MIDI score to tokens."""
        midi = midi.copy().resample(self.config.ticks_per_beat)

        tempo = midi.tempos[-1].qpm if len(midi.tempos) > 0 else 120
        time_signature = midi.time_signatures[-1]
        if time_signature.numerator != 4 or time_signature.denominator != 4:
            raise ValueError(
                "Only 4/4 time signature is supported for IRMA tokenizer."
            )
        
        tempo_token = f"Tempo_{self.tempo_quantizer.quantize(tempo)}"

        # shuffle tracks
        tracks = midi.tracks.copy()
        tracks = random.sample(tracks, len(tracks))

        program_tokens = []
        track_tokens = []
        for track in tracks:

            if track.is_drum:
                # add 
                program_tokens.append(f"Program_Drums")
            else:
                program_tokens.append(f"Program_{track.program}")

            new_track_tokens = ["Track_None","Bar_None"]
            # add bar
            bar_count = 0
            curr_position = 0
            curr_offset = 0
            notes = track.notes.copy()
            notes.sort(key=lambda note: (note.start, note.pitch, note.velocity))
            for note in notes:
                # add bar tokens
                bar_idx = note.start // (self.config.ticks_per_beat * 4)
                while bar_count < bar_idx:
                    new_track_tokens.append("Bar_None")
                    bar_count += 1
                    position = 0
                    offset = 0

                # get onset
                onset = note.start

                # get position
                position = int(onset % (self.config.ticks_per_beat * 4) // self.ticks_per_position)
                if position != curr_position:
                    new_track_tokens.append(f"Position_{position}")
                    curr_position = position
                    offset = 0
                
                offset = int(onset % self.ticks_per_position)
                if offset != curr_offset:
                    new_track_tokens.append(f"Offset_{offset}")
                    curr_offset = offset

                # get pitch
                if track.is_drum:
                    new_track_tokens.append(f"Pitch_Drum{note.pitch}")
                else:
                    new_track_tokens.append(f"Pitch_{note.pitch}")

                # get velocity
                new_track_tokens.append(f"Velocity_{self.velocity_quantizer.quantize(note.velocity)}")
                # get duration

                # get duration
                duration = note.end - note.start

                # get closest duration in self.durations
                closest_duration = min(self.durations, key=lambda x: abs(x - duration))

                # get duration token
                new_track_tokens.append(f"Duration_{closest_duration}d{self.config.ticks_per_beat*4}")

            track_tokens.append(new_track_tokens)

        tokens = [tempo_token, *program_tokens]

        for track in track_tokens:
            tokens.extend(track)

        return tokens
    

    def tokens_to_midi(self, tokens):

        # assert that the first token is a tempo token
        assert tokens[0].startswith("Tempo_"), "First token must be a tempo token"

        tempo_token = tokens.pop(0)

        # then pop program tokens until we reach the first Track_None
        program_tokens = []
        while tokens and not tokens[0].startswith("Track_None"):
            pr_token = tokens.pop(0)
            assert pr_token.startswith("Program_"), "Program token must start with Program_"
            program_tokens.append(pr_token)
        
        # now we have the program tokens, we can start processing the tracks
        # first create symusic.Score object
        midi = symusic.Score()
        # set tick rate
        midi.resample(self.config.ticks_per_beat)

        # set tempo
        tempo = int(tempo_token.split("_")[-1])
        midi.tempos = [symusic.Tempo(qpm=tempo, time=0)]

        # set time signature
        midi.time_signatures.append(symusic.TimeSignature(numerator=4, denominator=4, time=0))
        

        def split_list_by_value(lst, value):
            result = []
            current_sublist = []
            
            for item in lst:
                if item == value:
                    if current_sublist:  # Save the current sublist if it's not empty
                        result.append(current_sublist)
                        current_sublist = []
                    # Optionally add the split value to a separate list or discard it
                else:
                    current_sublist.append(item)
            
            if current_sublist:  # Add the last sublist if it's not empty
                result.append(current_sublist)
            
            return result
            # split tokens by Track_None
            tokens_split_by_track = []
            token_idx = 0
            while token_idx < len(tokens):
                if tokens[token_idx].startswith("Track_None"):
                    # start a new track
                    track_tokens = []
                    token_idx += 1
                    while token_idx < len(tokens) and not tokens[token_idx].startswith("Track_None"):
                        track_tokens.append(tokens[token_idx])
                        token_idx += 1
                    tokens_split_by_track.append(track_tokens)
                else:
                    token_idx += 1

        # split tokens by Track_None
        tokens_split_by_track = split_list_by_value(tokens, "Track_None")

        # assert that we have the same number of tracks as programs
        assert len(tokens_split_by_track) == len(program_tokens), "Number of tracks must be equal to number of programs"

        # now create a track for each program
        for track_tokens, track_program in zip(tokens_split_by_track, program_tokens):
            # create a new track
            track = symusic.Track(is_drum=track_program == "Program_Drums", program=int(track_program.split("_")[-1]))
            # set bar count
            bar_count = 0
            curr_position = 0
            curr_offset = 0
            for token in track_tokens:
                if token.startswith("Bar_None"):
                    bar_count += 1
                    curr_position = 0
                elif token.startswith("Position_"):
                    curr_position = int(token.split("_")[-1])
                    curr_offset = 0
                elif token.startswith("Offset_"):
                    curr_offset = int(token.split("_")[-1])
                elif token.startswith("Pitch_"):
                    pitch_str = token.split("_")[-1]
                    if pitch_str.startswith("Drum"):
                        pitch = int(pitch_str.split("Drum")[-1])
                    else:
                        pitch = int(pitch_str)
                elif token.startswith("Velocity_"):
                    velocity = int(token.split("_")[-1])
                elif token.startswith("Duration_"):
                    duration = int(token.split("_")[-1].split("d")[0])
                    # create note
                    note = symusic.Note(time=bar_count * self.config.ticks_per_beat * 4 + curr_position * self.ticks_per_position + curr_offset, pitch=pitch, velocity=velocity, duration=duration)
                    track.notes.append(note)
            
            # add track to midi
            midi.tracks.append(track)

        return midi