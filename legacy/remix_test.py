#%%
import pedalboard
import soundfile as sf
from audiobox_aesthetics.infer import initialize_predictor
import torch
import numpy as np
import pygad


#%%
# set cuda visbble devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# show devices that pytorch can see
print(torch.cuda.device_count())

#%%

sample_rate = 48_000 
aes_predictor = initialize_predictor()


#%%
def get_aes_scores(records):
    # prepare inputs. 
    predictor_inputs = [{"path": torch.tensor(record["audio"]).float(), "sample_rate": sample_rate, "idx":i} for i, record in enumerate(records) if record["audio"] is not None]
    print(f"Predicting aesthetics")
    scores = aes_predictor.forward(predictor_inputs)
    # put back scores to records that have audio
    record_with_audio_index = 0
    for i, record in enumerate(records):
        if record["audio"] is not None:
            record["aes_scores"] = scores[record_with_audio_index]
            record_with_audio_index += 1
        else:
            record["aes_scores"] = None
    return records

audio, sample_rate = sf.read("data/test.wav")

audio = audio.T
audio_full = audio

SECONDS = 10

# get crop from middle
start = int((audio.shape[1] - SECONDS * sample_rate) / 2)
end = start + SECONDS * sample_rate
audio = audio[:, start:end]

# play audio
import IPython.display as ipd
ipd.Audio(audio, rate=sample_rate)

#%%

from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, PeakFilter, HighShelfFilter, Limiter

class MasteringChain:
    def __init__(self, sample_rate):
        # Create the plugins for the mastering chain
        self.low_shelf = LowShelfFilter()
        self.mid_peak = PeakFilter() 
        self.high_shelf = HighShelfFilter()
        self.compressor = Compressor()
        self.gain = Gain()
        self.limiter = Limiter()
        
        # Create the processing chain
        self.chain = Pedalboard([
            self.low_shelf,
            self.mid_peak,
            self.high_shelf,
            self.compressor,
            self.gain,
            self.limiter
        ])
        self.sample_rate = sample_rate
        
    def process(self, audio):
        """
        Process audio through the mastering chain.
        
        Args:
            audio: NumPy array containing the audio samples
            sample_rate: Sample rate of the audio in Hz
            
        Returns:
            Processed audio as a NumPy array
        """
        return self.chain(audio, self.sample_rate)
    
    def set_parameters(self, params):
        """
        Set parameters for the mastering chain.
        
        Args:
            params: List of parameters in the following order:
                [low_shelf_gain, mid_peak_gain, high_shelf_gain, 
                 compressor_threshold, compressor_ratio, 
                 makeup_gain, limiter_threshold]
        """
        if len(params) != 7:
            raise ValueError("Expected 7 parameters")
            
        self.low_shelf.gain_db = params[0]
        self.mid_peak.gain_db = params[1]
        self.high_shelf.gain_db = params[2]
        self.compressor.threshold_db = params[3]
        self.compressor.ratio = params[4]
        self.gain.gain_db = params[5]
        self.limiter.threshold_db = params[6]
        
    def get_parameter_ranges(self):
        """
        Get the minimum and maximum values for each parameter.
        
        Returns:
            List of tuples, each containing (min, max) for each parameter
        """
        return [
            (-12.0, 12.0),    # low_shelf_gain (dB)
            (-12.0, 12.0),    # mid_peak_gain (dB)
            (-12.0, 12.0),    # high_shelf_gain (dB)
            (-60.0, 0.0),     # compressor_threshold (dB)
            (1.01, 20.0),      # compressor_ratio
            (-12.0, 12.0),    # makeup_gain (dB)
            (-10.0, 0.0)      # limiter_threshold (dB)
        ]

chain = MasteringChain(sample_rate)

def fitness_function(ga_instance, solution, solution_idx):
    chain.set_parameters(solution)
    processed_audio = chain.process(audio)
    record = [{"audio": processed_audio}]
    record = get_aes_scores(record)
    print(f"Solution {solution_idx} - scores: {record[0]['aes_scores']}")
    return sum(value for value in record[0]["aes_scores"].values())

gene_space = chain.get_parameter_ranges()
num_genes = len(chain.get_parameter_ranges())

num_generations = 10
num_parents_mating = 8  # Increased from 4 for better diversity
sol_per_pop = 100  # Larger population to compensate for fewer generations
parent_selection_type = "tournament"  # More effective selection method
K_tournament = 3  # Small tournament size for quick convergence

initial_population = np.random.uniform(
    low=[x[0] for x in gene_space],
    high=[x[1] for x in gene_space],
    size=(sol_per_pop, num_genes)
)

# Initialize the PyGAD instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    initial_population=initial_population,
    fitness_func=fitness_function,
    gene_space=gene_space,
    gene_type=float,
    parent_selection_type=parent_selection_type,
    K_tournament=K_tournament,
    mutation_type="random",  # Simpler mutation type
    mutation_percent_genes=15,  # Mutate 15% of genes
    crossover_type="uniform",  # Better mixing of genes
    crossover_probability=0.9,
    keep_parents=4,  # Keep more elite solutions
    save_best_solutions=True,
)

# Run the genetic algorithm optimization
print("Starting optimization...")
ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution parameters: {solution}")
print(f"Best solution fitness (PQ score): {solution_fitness}")

#%%
# Optional: Plot the fitness history
ga_instance.plot_fitness()

#%%
# Apply the best parameters to the master chain and process the audio
chain.set_parameters(solution)
optimized_audio = chain.process(audio_full)


# %%
# play optimized audio
import IPython.display as ipd
ipd.Audio(optimized_audio, rate=sample_rate)


# %%
