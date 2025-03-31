
import os
import tempfile
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset
from symusic import dump_wav
from audiobox_aesthetics.infer import initialize_predictor
import tinysoundfont
import numpy as np
from symusic import Synthesizer
import symusic
from util import crop_sm, sm_seconds
import datetime
import numpy as np
import tempfile
import tinysoundfont
import os
from joblib import Parallel, delayed, parallel_backend
import threading

class RewardManager:
    def __init__(self, processors, reward_weights, output_dir):
        """
        Initialize reward manager with rendering and reward functions.
        """
        self.processors = processors
        self.reward_weights = reward_weights
        self.global_reward_step = 0
        self.output_dir = output_dir
        self.audio_save_interval = 10
        self.__name__ = "RewardManager"

    def __call__(self, completions, prompts, **kwargs):
        # Render completions
        prompt_and_completions = torch.cat([torch.Tensor(prompts), completions.cpu()], dim=1)

        records = [{
            "completion": completion, 
            "prompt": prompt, 
            "prompt_and_completion": prompt_and_completion,
            "normalized_rewards":{}, 
            "reward_step": self.global_reward_step
            } for completion, prompt, prompt_and_completion in zip(completions, prompts,prompt_and_completions)]

        # add index
        records = [{"idx": i, **record} for i, record in enumerate(records)]
        # add full seqs
        # take time of each process
        time_taken = []
        for processor in self.processors:
            start = datetime.datetime.now()
            records = processor(records)
            end = datetime.datetime.now()
            time_taken.append((processor.__class__.__name__, end-start))
        print(f"Time taken for each processor: {time_taken}")

        # compute total reward
        for record in records:
            record["reward"] = sum([record["normalized_rewards"].get(key, 0) * self.reward_weights[key] for key in self.reward_weights.keys()]) / sum(self.reward_weights.values())
            record["reward_weights"] = self.reward_weights
        self.export_records_callback(records)
        self.global_reward_step += 1
        return [record["reward"] for record in records]

    def export_records_callback(self, records):

        # Prepare logs (exclude audio and sm fields)
        logs = []
        dont_log = ["audio", "sm"]
        for record in records:
            log = {**record}
            for key in dont_log:
                log.pop(key)
            logs.append(log)
        
        # Save logs as parquet
        os.makedirs(f"{self.output_dir}/rl_logs/{self.global_reward_step}", exist_ok=True)
        log_ds = Dataset.from_list(logs)
        log_ds.to_parquet(f"{self.output_dir}/rl_logs/{self.global_reward_step}/logs.parquet")
        
        # Save MIDI files
        os.makedirs(f"{self.output_dir}/midi/{self.global_reward_step}", exist_ok=True)
        for i in range(len(records)):
            records[i]["sm"].dump_midi(f"{self.output_dir}/midi/{self.global_reward_step}/reward={records[i]["reward"]}_{i}.mid")
        
        # Save audio files periodically
        if self.global_reward_step % self.audio_save_interval == 0:
            os.makedirs(f"{self.output_dir}/audio/{self.global_reward_step}", exist_ok=True)
            for i in range(len(records)):
                try:
                    dump_wav(
                        f"{self.output_dir}/audio/{self.global_reward_step}/reward={records[i]["reward"]}_{i}.wav", 
                        records[i]["audio"], 
                        records[i]["sample_rate"], 
                        use_int16=True
                    )
                except Exception as e:
                    print(f"Error dumping wav: {e}")
        print(f"Done saving logs and audio for {len(records)} records")
        return records

class Processor:
    def __call__(self, records):
        '''
        Takes a list of records and returns a list of records of the same length with additional fields
        '''
        raise NotImplementedError
            

class AudioBoxAesRewardProcessor(Processor):
    def __init__(self,):
        self.aes_predictor = initialize_predictor()
        
    def get_aes_scores(self,records):
        """
        Calculate aesthetic scores for records that have audio data.
        
        Args:
            records: List of record dictionaries containing audio data
            
        Returns:
            Updated records with aesthetic scores added
        """
        # Prepare inputs for predictor (only for records with valid audio)
        predictor_inputs = [
            {
                "path": torch.tensor(record["audio"]).float(), 
                "duration": record["audio_duration"],
                "sample_rate": record["sample_rate"],
                "idx": i
            } 
            for i, record in enumerate(records) 
            if record["audio"] is not None
        ]
        
        # Get scores from aesthetic predictor
        scores = self.aes_predictor.forward(predictor_inputs)
        
        # Map scores back to original records
        record_with_audio_index = 0
        for record in records:
            if record["audio"] is not None:
                record["aes_scores"] = scores[record_with_audio_index]
                record_with_audio_index += 1
            else:
                record["aes_scores"] = None

        # add normalized rewards
        for record in records:
            if record["aes_scores"] is not None:
                record["normalized_rewards"]["CE"] = record["aes_scores"]["CE"] / 10
                record["normalized_rewards"]["CU"] = record["aes_scores"]["CU"] / 10
                record["normalized_rewards"]["PC"] = record["aes_scores"]["PC"] / 10
                record["normalized_rewards"]["PQ"] = record["aes_scores"]["PQ"] / 10
                
        return records
    
    def __call__(self, records):
        return self.get_aes_scores(records)
    
class MidiTokToSymusicProcessor(Processor):
    def __init__(self, tokenizer, is_multitrack, max_beats):
        self.tokenizer = tokenizer
        self.is_multitrack = is_multitrack
        self.max_beats = max_beats

    def __call__(self, records):
        for record in records:
            record["prompt_and_completion_tokens"] = self.tokenizer._ids_to_tokens(record["prompt_and_completion"].tolist())
            if self.is_multitrack:
                record["sm"] = self.tokenizer(record["prompt_and_completion"])
            else:
                record["sm"] = self.tokenizer(record["prompt_and_completion"][None, ...])
            if self.max_beats is not None:
                record["sm"] = crop_sm(record["sm"], self.max_beats)
                record["sm_duration"] = sm_seconds(record["sm"])
        return records

class SymusicSynthProcessor(Processor):
    def __init__(self, soundfont_path, sample_rate, max_duration_seconds):
        self.synth = Synthesizer(
            sf_path = soundfont_path, # the path to the soundfont
            sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
        )
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds

    def render(self, midi_path_or_symusic, duration_seconds):
        if isinstance(midi_path_or_symusic, str):
            midi = symusic.Score(midi_path_or_symusic)
        else:
            midi = midi_path_or_symusic
        audio = self.synth.render(midi, stereo=True)
        if audio.shape[1] > self.max_duration_seconds * self.sample_rate:
            audio = audio[:, :int(self.max_duration_seconds * self.sample_rate)]
        if audio.shape[1] > duration_seconds * self.sample_rate:
            audio = audio[:, :int(duration_seconds * self.sample_rate)]
        audio = audio / np.abs(audio).max() + 1e-6
        return audio
    
    def __call__(self, records):
        for record in records:
            record["audio"] = self.render(record["sm"], record["sm_duration"])
            # add sample rate
            record["sample_rate"] = self.sample_rate
            # add audio duration
            record["audio_duration"] = record["audio"].shape[1] / self.sample_rate
        return records

class TinySoundfontSynthProcessor(Processor):
    def __init__(self, soundfont_path, sample_rate, max_duration_seconds,):
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
        sfid = self.synth.sfload(soundfont_path)
        # Create a sequencer
        self.max_duration_seconds = max_duration_seconds
    
    def render(self, midi_path, duration_seconds):
        """
        Render a MIDI file to audio.
        
        Parameters:
        -----------
        midi_path : str or symusic.Score
            Path to the MIDI file to render or a symusic object
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
        if stereo_audio.shape[1] > self.max_duration_seconds * self.sample_rate:
            stereo_audio = stereo_audio[:, :int(self.max_duration_seconds * self.sample_rate)]
        if stereo_audio.shape[1] > duration_seconds * self.sample_rate:
            stereo_audio = stereo_audio[:, :int(duration_seconds * self.sample_rate)]
        # normalize
        # if audio has zero durattion, set to zeros for 1 second
        if stereo_audio.shape[1] == 0:
            stereo_audio = np.zeros((2, self.sample_rate))
        stereo_audio = stereo_audio / (np.abs(stereo_audio).max() + 1e-6)
        return stereo_audio
    
    def __call__(self, records):
        for record in records:
            with tempfile.NamedTemporaryFile(suffix=".mid") as f:
                record["sm"].dump_midi(f.name)
                record["audio"] = self.render(f.name, record["sm_duration"])
                # add sample rate
                record["sample_rate"] = self.sample_rate
                # add audio duration
                record["audio_duration"] = record["audio"].shape[1] / self.sample_rate
        return records


class ProgramPromptAdherenceRewardProcessor(Processor):

    def __call__(self, records):
        for record in records:
            try:
                # split tokens into head body
                # head is everything before the first bar token
                head = record["prompt_and_completion_tokens"][:record["prompt_and_completion_tokens"].index("Bar_None")]
                # body is everything after the first bar token
                body = record["prompt_and_completion_tokens"][record["prompt_and_completion_tokens"].index("Bar_None")+1:]
                # head programs are everything that starts with Program_
                record["head_programs"] = set([x for x in head if x.startswith("Program_")])
                # body programs are everything that starts with Program_
                record["body_programs"] = set([x for x in body if x.startswith("Program_")])

                record["intersection_over_union_programs"] = len(record["head_programs"].intersection(record["body_programs"])) / len(record["head_programs"].union(record["body_programs"]))
                record["normalized_rewards"]["programs_iou"] = record["intersection_over_union_programs"]
            except:
                print(f"Couldnt compute program intersection over union with prompt programs for record {record['idx']}")
        return records

from transformers import ClapModel, ClapProcessor

class CLAPPromptRewardProcessor(Processor):

    def __init__(self, sample_rate, target_prompt, k):
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(0)
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

        # get text prompt features
        inputs = self.clap_processor(text=target_prompt, return_tensors="pt").to(0)
        self.text_embed = self.clap_model.get_text_features(**inputs).detach()
        self.sample_rate = sample_rate
        self.k = k

    def get_clap_features(self,audio_samples):
        audio_samples = [audio_samples[i].mean(0) for i in range(len(audio_samples))]
        inputs = self.clap_processor(audios=audio_samples, return_tensors="pt", sampling_rate=self.sample_rate).to(0)
        audio_embed = self.clap_model.get_audio_features(**inputs)
        return audio_embed

    def get_clap_text_features(self,prompt):
        inputs = self.clap_processor(text=prompt, return_tensors="pt").to(0)
        text_embed = self.clap_model.get_text_features(**inputs)
        return text_embed

    def score_clap(self, audio):
        audio_embed = self.get_clap_features(audio)
        # get cosine similarity to text prompt
        scores = torch.nn.functional.cosine_similarity(audio_embed, self.text_embed)
        return scores
    
    def __call__(self, records):
        # first get audio
        audio = [record["audio"] for record in records]
        scores = self.score_clap(audio)
        

        # rescale from -1 to 1 to 0-1
        norm_scores = (scores + 1) / 2 
        #
        
        # All other samples get 0 by default
        
        # Apply rewards to records
        for i, record in enumerate(records):
            record["normalized_rewards"]["clap"] = norm_scores[i].item()
            record["clap_score_raw"] = scores[i].item()
        
        return records


class CLAPZeroShotClassificationRewardProcessor(Processor):

    def __init__(self, sample_rate, target_prompt, reference_prompts, temperature):
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_general").to(0)
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

        prompts = [target_prompt] + reference_prompts
        # get text prompt features
        self.text_embeds = []
        for prompt in prompts:
            inputs = self.clap_processor(text=prompt, return_tensors="pt").to(0)
            self.text_embeds.append(self.clap_model.get_text_features(**inputs).detach())
        self.sample_rate = sample_rate
        self.temperature = temperature

    def get_clap_features(self,audio_samples):
        audio_samples = [audio_samples[i].mean(0) for i in range(len(audio_samples))]
        inputs = self.clap_processor(audios=audio_samples, return_tensors="pt", sampling_rate=self.sample_rate).to(0)
        audio_embed = self.clap_model.get_audio_features(**inputs)
        return audio_embed

    def get_clap_text_features(self,prompt):
        inputs = self.clap_processor(text=prompt, return_tensors="pt").to(0)
        text_embed = self.clap_model.get_text_features(**inputs)
        return text_embed

    def score_clap(self, audio):
        audio_embed = self.get_clap_features(audio).detach()
        # get softmax over all text prompts
        # get cosine similarity to text prompt
        scores = torch.nn.functional.cosine_similarity(audio_embed, torch.stack(self.text_embeds), dim=-1)
        # get softmax
        scores = torch.nn.functional.softmax(scores / self.temperature, dim=0).T
        print(f"Scores: {scores.shape}")
        return scores
    
    def __call__(self, records):
        # first get audio
        audio = [record["audio"] for record in records]
        raw_scores = self.score_clap(audio)

        print(f"Raw scores: {raw_scores.shape}")

        # Apply rewards to records
        for i, record in enumerate(records):
            # save raw scores
            record["clap_score_raw"] = raw_scores[i].cpu().numpy()
            record["normalized_rewards"]["clap_clf"] = raw_scores[i][0].item()
        
        return records
