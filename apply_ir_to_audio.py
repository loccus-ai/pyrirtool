import argparse
import librosa
import numpy as np
import os
import soundfile as sf
from deconvolve import replace_extension

def process(audio_file, ir_data, sample_rate, suffix: str, output_directory: str):
    """
    Applies the reverb effect to the given audio file using the provided IR parameter.
    
    :param audio_file: The name of the audio file to process.
    :param ir: The impulse response parameter for the reverb effect.
    """

    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=sample_rate)
    print(f"Loaded audio file: {audio_file}, Sample rate: {sample_rate}")

    output_file = replace_extension(audio_file, "_"+suffix)    
    if output_directory:
        output_file = os.path.join(output_directory, output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    output = np.convolve(audio_data, ir_data, mode='full')    
    
    sf.write(output_file, output, sample_rate)
    print(f"Processed audio file saved as: {output_file}")
    

def main():
    parser = argparse.ArgumentParser(description="Apply reverb to audio files.")
    parser.add_argument("audio_files", nargs="+", help="List of audio files to process.")
    parser.add_argument("--ir", required=True, help="Impulse response parameter for reverb.")
    parser.add_argument("--output_directory", default="", help="Directory to save processed audio files.")
    
    args = parser.parse_args()
    ir_data, sample_rate = librosa.load(args.ir)
    
    for audio_file in args.audio_files:
        process(audio_file, ir_data, sample_rate, suffix=os.path.basename(args.ir), output_directory=args.output_directory)

if __name__ == "__main__":
    main()