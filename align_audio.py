import soundfile as sf
import numpy as np
from scipy.signal import correlate
import argparse
import librosa
import json
import matplotlib.pyplot as plt
from pathlib import Path

from deconvolve import replace_extension


def read_audio(file_path, sample_rate, num_channel=1):
    audio_data, sample_rate = librosa.load(file_path, sr=sample_rate, mono=False)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[num_channel]
    return audio_data, sample_rate

def find_delay(audio1, audio2):
    correlated = correlate(audio1, audio2, mode='full', method='auto')
    delay = np.argmax(correlated) - len(audio2) + 1
    return delay

def align_audio(audio1, audio2):
    delay = find_delay(audio1, audio2)

    if delay > 0:
        audio1 = audio1[delay:]  # Trim the start of audio1
    else:
        audio2 = audio2[-delay:]  # Trim the start of audio2

    # min_length = min(len(audio1), len(audio2))
    # audio1 = audio1[:min_length]
    # audio2 = audio2[:min_length]

    return audio1, audio2
    
def parse_args():
    parser = argparse.ArgumentParser(description='Audio alignment tool')
    parser.add_argument('--original', required=True, help='Path to original audio file. Example: /data/audio/EXP28-tel/concatenated_sweep/recordings/20250424_112112_plus0000_RE5292d71d0c42dfdc79b594a1d9445dac_original.wav')
    parser.add_argument('--recorded', required=True, help='Path to recorded audio file, Example: /data/audio/EXP28-tel/concatenated_sweep/recordings/20250424_112112_plus0000_RE5292d71d0c42dfdc79b594a1d9445dac_nokia3310.wav')
    parser.add_argument('--jsonl', type=str, required=True, help='Path to JSONL file. Example: /data/audio/EXP28-tel/concatenated_sweep/recordings/20250424_112112_plus0000_RE5292d71d0c42dfdc79b594a1d9445dac_info.jsonl')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Sampling rate')
    parser.add_argument('--output_dir', default='.', help='Output directory. Example: /data/audio/EXP28-tel/recordings/')
    parser.add_argument('--output_suffix', default='aligned', help='Suffix for output files. Example: .20250424_112112_plus0000_RE5292d71d0c42dfdc79b594a1d9445dac.nokia3310.wav')
    parser.add_argument('--channel', type=int, default=0, help='Audio channel to process')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--basedir', type=str, default='.', help='Base input directory. Example: /data/audio/EXP28-tel/')
    parser.add_argument('--max_lag', type=float, default=0.25, help='Maximum lag in seconds')
    return parser.parse_args()
    
def main():
    
    args = parse_args()
    
    SR = args.sample_rate
    
    original_audio, _ = read_audio(args.original, args.sample_rate, args.channel)
    recorded_audio, _ = read_audio(args.recorded, args.sample_rate, args.channel)
    
    original_audio_aligned, recorded_audio_aligned = align_audio(original_audio, recorded_audio)
    
    MAX_LAG = int(args.max_lag * SR) # fine-tune lag range 
    # Iterate through each line in the JSONL file
    with open(args.jsonl, 'r') as file:
        n = 0
        for line in file:
            # Parse the JSON object
            data = json.loads(line)
            start = int(data["start"]*SR)
            end = int(data["end"]*SR)
            # Load the long audio file
            original_filename = data['filename']
            if original_filename == "sweep.wav":
                original_filename = "sweeps/" + replace_extension(original_filename, f"_{start}_{end}.wav")
                
            try:      
                original_audio_segment = original_audio[start:end]

                recorded_audio_start = np.max([0, start - MAX_LAG])
                recorded_audio_end = np.min([len(recorded_audio), end + MAX_LAG])
                recorded_audio_chunk = recorded_audio_aligned[recorded_audio_start:recorded_audio_end]
                
                print(f"Original audio segment: {start}:{end} (len: {len(original_audio_segment)})")
                print(f"Recorded audio chunk: {recorded_audio_start}:{recorded_audio_end} (len: {len(recorded_audio_chunk)})")
                
                # Compute the cross-correlation with the long array
                cross_corr = correlate(original_audio_segment, recorded_audio_chunk, mode='full', method='auto')
                
                # Find the index of the maximum correlation
                max_corr_index = np.argmax(cross_corr)
                
                # Calculate the start index in the long array
                delay = max_corr_index - len(recorded_audio_chunk) + 1
                
                if delay > 0:
                    original_audio_segment = original_audio_segment[delay:]  # Trim the start of audio1
                else:
                    recorded_audio_chunk = recorded_audio_chunk[-delay:]
                
                
                output_filename = original_filename.replace(args.basedir, "")
                output_filename = Path(args.output_dir) / output_filename
                output_filename.parent.mkdir(parents=True, exist_ok=True)
                output_filename = replace_extension(output_filename, args.output_suffix)
                print(f"Output filename: {output_filename}")

                sf.write(output_filename, recorded_audio_chunk[:len(original_audio_segment)], SR)
                
                print(f"Filename: {data['filename']}, delay {max_corr_index}, max corr {cross_corr[max_corr_index]}, Start Index in Long Array: {delay}")
                
                if args.plot:
                    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

                    # Plot audio1
                    axs[0].plot(original_audio_segment)
                    axs[0].set_title("Original audio segment")
                    axs[0].set_ylabel("Amplitude")
                    axs[0].grid()

                    # Plot audio2
                    axs[1].plot(recorded_audio_aligned[recorded_audio_start:recorded_audio_end])
                    axs[1].set_title("Recorded audio chunk")
                    axs[1].set_xlabel("Sample Index")
                    axs[1].set_ylabel("Amplitude")
                    axs[1].grid()

                    # Plot recovered recorded segment
                    axs[2].plot(recorded_audio_chunk[:len(original_audio_segment)])
                    axs[2].set_title(f"Recovered recorded segment (Start Index: {delay})")
                    axs[2].set_xlabel("Sample Index")
                    axs[2].set_ylabel("Amplitude")
                    axs[2].grid()

                    plt.tight_layout()
                    # plt.show()
                    plot_filename = replace_extension(output_filename, ".png")                
                    plt.savefig(plot_filename)
                    print(f"Plot saved as: {plot_filename}")
            
            except Exception as e:
                print(f"Error processing line {n}: {e}")
                continue
            
            n += 1


if __name__ == "__main__":
    main()