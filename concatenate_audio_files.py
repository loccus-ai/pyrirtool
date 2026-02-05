import os
import random
import argparse
import subprocess
import tempfile
from pydub import AudioSegment
import jsonlines

# Target audio format: 16kHz, 16-bit, mono
TARGET_FRAME_RATE = 16000
TARGET_SAMPLE_WIDTH = 2  # 16 bits = 2 bytes
TARGET_CHANNELS = 1


def normalize_audio(audio):
    """
    Converts an audio segment to linear PCM 16kHz, 16 bits/sample, mono.

    Args:
        audio (AudioSegment): The audio segment to normalize.

    Returns:
        AudioSegment: The normalized audio segment.
    """
    return audio.set_frame_rate(TARGET_FRAME_RATE).set_sample_width(TARGET_SAMPLE_WIDTH).set_channels(TARGET_CHANNELS)


def load_audio_normalized(filepath):
    """
    Loads an audio file and normalizes it to 16kHz, 16-bit, mono PCM.
    Handles floating-point WAV files by using ffmpeg for conversion.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        AudioSegment: The normalized audio segment.
    """
    # Use ffmpeg to convert to standard PCM format first
    # This handles float WAV and other non-standard formats
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
        cmd = [
            'ffmpeg', '-y', '-i', filepath,
            '-ar', str(TARGET_FRAME_RATE),
            '-ac', str(TARGET_CHANNELS),
            '-sample_fmt', 's16',  # signed 16-bit
            '-f', 'wav',
            tmp.name
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        audio = AudioSegment.from_file(tmp.name, format='wav')

    return audio


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


def append_with_sync_tone(current_audio, segment_to_add, sync_tone, file_info, segment_info, sync_tone_filename):
    """
    Appends a segment to the current audio, preceded by a sync tone, and updates file_info.

    Args:
        current_audio (AudioSegment): The current concatenated audio.
        segment_to_add (AudioSegment): The audio segment to append.
        sync_tone (AudioSegment): The synchronization tone audio segment.
        file_info (list): List of file info dictionaries to update.
        segment_info (dict): Info dict for the segment being added (filename, is_sweep).
        sync_tone_filename (str): The actual filename of the sync tone audio file.

    Returns:
        AudioSegment: The updated concatenated audio.
    """
    # Add sync tone first
    sync_start = current_audio.duration_seconds
    current_audio += sync_tone
    sync_end = current_audio.duration_seconds
    file_info.append({'filename': sync_tone_filename, 'start': sync_start, 'end': sync_end, 'is_sweep': False, 'is_sync_tone': True})

    # Add the actual segment
    segment_start = current_audio.duration_seconds
    current_audio += segment_to_add
    segment_end = current_audio.duration_seconds
    file_info.append({
        'filename': segment_info['filename'],
        'start': segment_start,
        'end': segment_end,
        'is_sweep': segment_info.get('is_sweep', False),
        'is_sync_tone': False
    })

    return current_audio

def generate_long_audios(input_files, sweep_audio, sync_tone, output_dir, sweep_probability, root_dir, output_length_seconds, sweep_filename, sync_tone_filename):
    """
    Generates audio files of specified length by concatenating multiple input audio files and inserting sweep audio segments.
    A synchronization tone is inserted before each segment (including sweeps) for later reconstruction.

    Args:
        input_files (list): List of input audio file paths relative to root_dir.
        sweep_audio (AudioSegment): The sweep audio segment.
        sync_tone (AudioSegment): The synchronization tone audio segment.
        output_dir (str): Directory to save the output audio files.
        sweep_probability (float): Probability of inserting the sweep audio segment (0 to 1).
        root_dir (str): Root directory containing the audio files.
        output_length_seconds (int): Length of the output audio files in seconds.
        sweep_filename (str): The actual filename of the sweep audio file.
        sync_tone_filename (str): The actual filename of the sync tone audio file.
    """
    file_count = 1

    while input_files:
        # Start with empty audio matching the target format (16kHz, 16-bit, mono)
        current_audio = AudioSegment.silent(
            duration=0,
            frame_rate=TARGET_FRAME_RATE
        ).set_channels(TARGET_CHANNELS).set_sample_width(TARGET_SAMPLE_WIDTH)
        file_info = []

        # Add initial sweep with sync tone
        current_audio = append_with_sync_tone(
            current_audio, sweep_audio, sync_tone, file_info,
            {'filename': sweep_filename, 'is_sweep': True},
            sync_tone_filename
        )

        random.shuffle(input_files)  # Shuffle input files for randomness

        while input_files and current_audio.duration_seconds < output_length_seconds:
            file = os.path.join(root_dir, input_files.pop(0))
            audio_segment = load_audio_normalized(file)
            sweep_segment, was_sweep_inserted = maybe_insert_sweep(sweep_audio, sweep_probability)

            # Add audio file with sync tone
            current_audio = append_with_sync_tone(
                current_audio, audio_segment, sync_tone, file_info,
                {'filename': file, 'is_sweep': False},
                sync_tone_filename
            )

            # Add sweep if randomly selected
            if was_sweep_inserted:
                current_audio = append_with_sync_tone(
                    current_audio, sweep_segment, sync_tone, file_info,
                    {'filename': sweep_filename, 'is_sweep': True},
                    sync_tone_filename
                )

        # Add the sweep audio at the end with sync tone
        current_audio = append_with_sync_tone(
            current_audio, sweep_audio, sync_tone, file_info,
            {'filename': sweep_filename, 'is_sweep': True},
            sync_tone_filename
        )

        output_file = os.path.join(output_dir, f'long_audio_{file_count:04}.wav')
        current_audio.export(output_file, format="wav")

        with jsonlines.open(os.path.join(output_dir, f'long_audio_{file_count:04}_info.jsonl'), 'w') as info_file:
            for entry in file_info:
                info_file.write(entry)

        print(f"Generated {output_file} with duration {current_audio.duration_seconds / 60.0} minutes")

        file_count += 1

def main():
    """
    Main function to parse command line arguments and call the generate_long_audios function.
    """
    parser = argparse.ArgumentParser(description="Concatenate audio files into specified length audio files, inserting sweep file with a random probability and always adding sweep at the beginning and end. A sync tone is inserted before each segment.")
    parser.add_argument('--input_list', type=str, required=True, help="Path to the file containing the list of audio files to concatenate")
    parser.add_argument('--sweep_file', type=str, required=True, help="Path to the sweep audio file to insert")
    parser.add_argument('--sync_tone_file', type=str, required=True, help="Path to the synchronization tone audio file to insert before each segment")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output audio files")
    parser.add_argument('--sweep_probability', type=float, required=True, help="Probability of inserting the sweep audio file (0 to 1)")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing the audio files")
    parser.add_argument('--output_length', type=int, required=True, help="Length of the output audio files in seconds")

    args = parser.parse_args()

    with open(args.input_list, 'r') as f:
        input_files = [line.strip() for line in f]

    sweep_audio = load_audio_normalized(args.sweep_file)
    sync_tone = load_audio_normalized(args.sync_tone_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generate_long_audios(input_files, sweep_audio, sync_tone, args.output_dir, args.sweep_probability, args.root_dir, args.output_length, args.sweep_file, args.sync_tone_file)

if __name__ == "__main__":
    main()