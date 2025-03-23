import torch
from tqdm import tqdm
import pandas as pd
import datasets
import os
import gc

def convert_to_hf_dataset_in_batches(
    input_path, 
    output_path, 
    batch_size=1000, 
    remove_temp_files=True
):
    """
    Convert PyTorch dataset to Hugging Face dataset in batches to avoid memory issues.
    
    Args:
        input_path: Path to the PyTorch dataset file
        output_path: Path to save the Hugging Face dataset
        batch_size: Number of records to process at once
        remove_temp_files: Whether to remove temporary files after processing
    """
    print(f"Loading dataset from {input_path}")
    
    # Create temp directory for batch files
    temp_dir = f"{output_path}_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load all records but only keep IDs or some minimal info to create batches
    full_data = torch.load(input_path, weights_only=False)
    total_records = len(full_data)
    print(f"Total records: {total_records}")
    
    # Process in batches
    batch_paths = []
    for i in tqdm(range(0, total_records, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, total_records)
        batch_data = full_data[i:end_idx]
        
        # Convert batch to DataFrame and then to HF dataset
        df_batch = pd.DataFrame(batch_data)
        dataset_batch = datasets.Dataset.from_pandas(df_batch)
        
        # Save batch
        batch_path = f"{temp_dir}/batch_{i}_{end_idx}"
        dataset_batch.save_to_disk(batch_path)
        batch_paths.append(batch_path)
        
        # Clear memory
        del batch_data, df_batch, dataset_batch
        gc.collect()
    
    # Release memory from full dataset
    del full_data
    gc.collect()
    
    # Combine all batches
    print("Combining batches...")
    datasets_to_combine = [datasets.load_from_disk(path) for path in batch_paths]
    combined_dataset = datasets.concatenate_datasets(datasets_to_combine)
    
    # Save final dataset
    print(f"Saving combined dataset to {output_path}")
    combined_dataset.save_to_disk(output_path)
    
    # Clean up temporary files if requested
    if remove_temp_files:
        print("Cleaning up temporary files...")
        for path in batch_paths:
            try:
                import shutil
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing {path}: {e}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error removing {temp_dir}: {e}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    print("Saving datasets to Hugging Face format")
    ds_path = "data/gmd_loops_2_tokenized/trn.pt"
    output_path = "data/gmd_loops_2_tokenized/trn_hf"
    
    # You can adjust the batch size based on your available memory
    convert_to_hf_dataset_in_batches(ds_path, output_path, batch_size=500)