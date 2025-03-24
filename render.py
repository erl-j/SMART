import tinysoundfont
import numpy as np

class MidiRenderer:
    def __init__(self, soundfont_path, sample_rate):
        """
        Initialize the MIDI renderer with a specified soundfont.
        
        Parameters:
        -----------
        soundfont_path : str
            Path to the soundfont file (.sf2)
        samplerate : int
            Sample rate for audio rendering
        """
        # Initialize the synthesizer
        self.sample_rate = sample_rate
        self.synth = tinysoundfont.Synth(samplerate=self.sample_rate)
        
        # Load the soundfont
        self.sfid = self.synth.sfload(soundfont_path)
        
        # Create a sequencer
    
    def render(self, midi_path, duration_seconds):
        """
        Render a MIDI file to audio.
        
        Parameters:
        -----------
        midi_path : str
            Path to the MIDI file to render
        duration_seconds : float or None
            Duration in seconds to render. If None, will try to determine from MIDI.
            
        Returns:
        --------
        numpy.ndarray
            Audio as a numpy array with shape (2, samples) for stereo output
        """
        # Load the MIDI file
        self.synth.notes_off()

        self.seq = tinysoundfont.Sequencer(self.synth)

        self.seq.midi_load(midi_path)
        
        buffer_size = int(self.sample_rate * duration_seconds)
        
        # Generate audio buffer
        buffer = self.synth.generate(buffer_size)
        
        # Convert to numpy array
        block = np.frombuffer(bytes(buffer), dtype=np.float32)
        
        # Reshape to stereo (channels, samples)
        # The buffer is interleaved stereo where left channel is even samples, 
        # right channel is odd samples
        stereo_audio = np.stack([block[::2], block[1::2]])
        
        return stereo_audio