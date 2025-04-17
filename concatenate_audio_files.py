import os
import random
import argparse
from pydub import AudioSegment
import jsonlines


def maybe_insert_sweep(sweep_audio, probability):
    """
    Determines whether to insert the sweep audio based on the specified probability.

    Args:
        sweep_audio (AudioSegment): The sweep audio segment.
        probability (float): The probability of inserting the sweep audio (0 to 1).

    Returns:
        tuple: A tuple containing the sweep audio segment (or silence) and a boolean indicating whether the sweep was inserted.
    """
    if random.random() < probability:
        return sweep_audio, True
    else:
        return AudioSegment.silent(duration=0), False

def generate_long_audios(input_files, sweep_audio, output_dir, sweep_probability, root_dir, output_length_seconds):
    """
    Generates audio files of specified length by concatenating multiple input audio files and inserting sweep audio segments.

    Args:
        input_files (list): List of input audio file paths relative to root_dir.
        sweep_audio (AudioSegment): The sweep audio segment.
        output_dir (str): Directory to save the output audio files.
        sweep_probability (float): Probability of inserting the sweep audio segment (0 to 1).
        root_dir (str): Root directory containing the audio files.
        output_length_seconds (int): Length of the output audio files in seconds.
    """
    output_length_ms = output_length_seconds * 1000  # Convert seconds to milliseconds
    file_count = 1

    while input_files:
        current_audio = sweep_audio  # Start with the sweep audio
        file_info = [{'filename': 'sweep.wav', 'start': 0, 'end': sweep_audio.duration_seconds, 'is_sweep': True}]  # Init file info with the first sweep

        random.shuffle(input_files)  # Shuffle input files for randomness

        while input_files and current_audio.duration_seconds < output_length_seconds:
            file = os.path.join(root_dir, input_files.pop(0))
            audio_segment = AudioSegment.from_file(file)
            sweep_segment, was_sweep_inserted = maybe_insert_sweep(sweep_audio, sweep_probability)
            
            start = current_audio.duration_seconds
            end = start + audio_segment.duration_seconds
            file_info.append({'filename': file, 'start': start, 'end': end, 'is_sweep': False})

            current_audio += audio_segment + sweep_segment

            if was_sweep_inserted:
                sweep_start = end
                sweep_end = sweep_start + sweep_segment.duration_seconds
                file_info.append({'filename': 'sweep.wav', 'start': sweep_start, 'end': sweep_end, 'is_sweep': True})

        # Add the sweep audio at the end
        end_sweep_start = current_audio.duration_seconds
        end_sweep_end = end_sweep_start + sweep_audio.duration_seconds
        current_audio += sweep_audio
        file_info.append({'filename': 'sweep.wav', 'start': end_sweep_start, 'end': end_sweep_end, 'is_sweep': True})

        output_file = os.path.join(output_dir, f'long_audio_{file_count}.wav')
        current_audio.export(output_file, format="wav")

        with jsonlines.open(os.path.join(output_dir, f'long_audio_{file_count}_info.jsonl'), 'w') as info_file:
            for entry in file_info:
                info_file.write(entry)

        print(f"Generated {output_file} with duration {current_audio.duration_seconds / 60.0} minutes")

        file_count += 1

def main():
    """
    Main function to parse command line arguments and call the generate_long_audios function.
    """
    parser = argparse.ArgumentParser(description="Concatenate audio files into specified length audio files, inserting sweep file with a random probability and always adding sweep at the beginning and end")
    parser.add_argument('--input_list', type=str, required=True, help="Path to the file containing the list of audio files to concatenate")
    parser.add_argument('--sweep_file', type=str, required=True, help="Path to the sweep audio file to insert")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output audio files")
    parser.add_argument('--sweep_probability', type=float, required=True, help="Probability of inserting the sweep audio file (0 to 1)")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing the audio files")
    parser.add_argument('--output_length', type=int, required=True, help="Length of the output audio files in seconds")

    args = parser.parse_args()

    with open(args.input_list, 'r') as f:
        input_files = [line.strip() for line in f]

    sweep_audio = AudioSegment.from_file(args.sweep_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generate_long_audios(input_files, sweep_audio, args.output_dir, args.sweep_probability, args.root_dir, args.output_length)

if __name__ == "__main__":
    main()