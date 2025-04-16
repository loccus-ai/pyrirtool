import soundfile as sf
import numpy as np
from scipy.signal import correlate

def read_audio(file_path):
    audio_data, sample_rate = sf.read(file_path)
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

    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    return audio1, audio2

def main():
    # Read two audio files
    audio1_path = "audio1.wav"
    audio2_path = "audio2.wav"

    audio1, sample_rate1 = read_audio(audio1_path)
    audio2, sample_rate2 = read_audio(audio2_path)

    # Ensure both audio files have the same sample rate
    if sample_rate1 != sample_rate2:
        raise ValueError("Sample rates do not match.")

    # If stereo, convert to mono by averaging the channels
    if audio1.ndim > 1:
        audio1 = audio1.mean(axis=1)
    if audio2.ndim > 1:
        audio2 = audio2.mean(axis=1)

    # Align and trim audio files using correlation
    aligned_audio1, aligned_audio2 = align_audio(audio1, audio2)

    # Export aligned and trimmed audio files
    sf.write("aligned_audio1.wav", aligned_audio1, sample_rate1)
    sf.write("aligned_audio2.wav", aligned_audio2, sample_rate2)

    print("Audio files have been aligned and trimmed.")

if __name__ == "__main__":
    main()