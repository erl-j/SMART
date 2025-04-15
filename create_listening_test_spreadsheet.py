import glob

wav_dir = "listening_test"

# Get all audio files in the directory
audio_files = glob.glob(f"{wav_dir}/*.wav")

# Sort the files by name
audio_files.sort()

# now create a speadsheet with columns display (always "Trial"), and Stimulus (file name)
import pandas as pd

# Create a DataFrame with the desired columns
df = pd.DataFrame(columns=["display", "Stimulus"])

# Add the audio files to the DataFrame
for audio_file in audio_files:
    # Extract the file name without the directory
    file_name = audio_file.split("/")[-1]
    # Append the file name to the DataFrame
    df = df._append({"display": "Trial", "Stimulus": file_name}, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("listening_test.csv", index=False)

