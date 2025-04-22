import os
import torch
import numpy as np
import random
from tqdm import tqdm
import miditok
from datasets import Dataset
from processors import RewardManager, MidiTokToSymusicProcessor, TinySoundfontSynthProcessor, AudioBoxAesRewardProcessor, PamRewardProcessor
from pam_prompt_pairs import prompt_pairs
from symusic import BuiltInSF3

# Constants from the original script
SAMPLE_RATE = 48_000
MAX_AUDIO_DURATION = 10
MAX_BEATS = 100
NUM_EXAMPLES = 10

# Set seed for reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# List of soundfonts
SOUNDFONTS = [
    "musescore", 
    "sgm",
     # "monalisa", "ephesus", "touhou", "arachno", 
     "fluidr3",
    #"goldeneye", "ronaldiho", "casio", "n64", "gba", "candy", "808", 
    "grandeur", "ydp",
      "yamaha"
      , "matrix"
]

# Mapping of soundfont names to file paths
SF_PATHS = {
    "musescore": str(BuiltInSF3.MuseScoreGeneral().path(download=True)), 
    "sgm": "./soundfonts/SGM-V2.01-XG-2.04.sf2",
    "monalisa": "./soundfonts/Monalisa_GM_v2_105.sf2",
    "ephesus": "./soundfonts/Ephesus_GM_Version_1_00.sf2",
    "touhou": "./soundfonts/Touhou.sf2",
    "arachno": "./soundfonts/Arachno SoundFont - Version 1.0.sf2",
    "fluidr3": "./soundfonts/FluidR3 GM.sf2",
    "goldeneye": "./soundfonts/GoldenEye_007.sf2",
    "ronaldiho": "./soundfonts/InternationalSuperStarSoccer.sf2",
    "casio": "./soundfonts/Casio_CTK-230_GM.sf2",
    "n64": "soundfonts/General_MIDI_64_1.6.sf2",
    "gba": "soundfonts/General_Game_Boy_Advance_Soundfont.sf2",
    "candy": "soundfonts/Candy_Set_Full_GM.sf2",
    "808": "soundfonts/General808.sf2",
    "grandeur": "soundfonts/[GD] The Grandeur D.sf2",
    "ydp": "soundfonts/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2",
    "yamaha": "soundfonts/Yamaha-C5-Salamander-JNv5_1.sf2",
    "matrix": "soundfonts/MatrixSF_v2.1.5.sf2",
    "touhou": "soundfonts/Touhou.sf2",
}

# Reward weights (from the original script)
REWARD_WEIGHTS = {
    "CE": 1.0,
    # Uncomment others if needed
    # "CU": 1.0,
    # "PC": 0.0,
    # "PQ": 1.0,
    # "programs_iou": 3.0,
    # "programs_iou": 1.0,
    # "pam_avg": 1.0,
}

def main():
    print("Loading the piano dataset...")
    
    # Load the piano dataset (using the path from the original script)
    dataset = Dataset.load_from_disk("data/dataset_mmd_piano/train")
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_EXAMPLES))
    
    # Load tokenizer from pretrained model (for REMI tokenizer)
    BASE_MODEL_PATH = "lucacasini/metamidipianophi3_6L_long"  # or whichever model was used
    tokenizer = miditok.REMI.from_pretrained(BASE_MODEL_PATH)
    
    # Create output directory
    output_dir = "soundfont_renderings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each soundfont
    for soundfont in tqdm(SOUNDFONTS, desc="Processing soundfonts"):
        try:
            sf_path = SF_PATHS[soundfont]
            sf_output_dir = os.path.join(output_dir, soundfont)
            os.makedirs(sf_output_dir, exist_ok=True)
            
            print(f"\nProcessing soundfont: {soundfont}")
            
            # Set up reward manager with the appropriate processors
            reward_manager = RewardManager(
                processors=[
                    MidiTokToSymusicProcessor(tokenizer, is_multitrack=False, max_beats=MAX_BEATS),
                    TinySoundfontSynthProcessor(sf_path, SAMPLE_RATE, MAX_AUDIO_DURATION),
                    AudioBoxAesRewardProcessor(),
                    PamRewardProcessor(sample_rate=SAMPLE_RATE, prompt_configs=prompt_pairs, temperature=0.25)
                ],
                reward_weights=REWARD_WEIGHTS,
                output_dir=sf_output_dir
            )
            
            # Process the dataset samples
            all_records = []
            
            for idx, example in enumerate(tqdm(dataset, desc=f"Processing examples with {soundfont}")):
                # Get token_ids from the example
                token = example["tokens"]

                token_ids = tokenizer._tokens_to_ids(token)
                
                # Create a tensor for the token_ids (simulating a single sample)
                token_tensor = torch.tensor([token_ids])

                # use first token as prompt
                prompt_token = token_tensor[:, 0].unsqueeze(1)

                # use the rest as completion
                completion_tokens = token_tensor[:, 1:]
                
                # Evaluate with reward manager
                records = reward_manager(
                    prompts=prompt_token,
                    completions=completion_tokens,
                    return_records=True,
                )
                
                # Set the original dataset index
                for record in records:
                    record["idx"] = idx
                
                all_records.extend(records)
            
            # Export the records with audio
            reward_manager.export_records(
                all_records, 
                save_audio=True, 
                output_dir=sf_output_dir, 
                step=0
            )
            
            # Reset reward manager for next soundfont
            reward_manager.reset()
            
            print(f"Completed processing for {soundfont}. Results saved to {sf_output_dir}")
            
        except Exception as e:
            print(f"Error processing soundfont {soundfont}: {str(e)}")
            continue

if __name__ == "__main__":
    from symusic import BuiltInSF3  # Import here to use MuseScore path
    main()