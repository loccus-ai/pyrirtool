import numpy as np
import wave
import struct
import argparse


def generate_dtmf_file(filename, digit, duration_ms=300, pause_ms=1000, sample_rate=44100):
    # DTMF Frequency Map
    dtmf_map = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
    }

    if digit not in dtmf_map:
        raise ValueError("Invalid DTMF digit")

    f1, f2 = dtmf_map[digit]
    
    # Calculate lengths
    t_tone = np.linspace(0, duration_ms / 1000, int(sample_rate * (duration_ms / 1000)), False)
    t_pause = np.zeros(int(sample_rate * (pause_ms / 1000)))

    # Generate the sine waves for the dual tones
    # We use 0.5 amplitude for each to ensure the sum doesn't exceed 1.0 (clipping)
    tone = 0.5 * np.sin(2 * np.pi * f1 * t_tone) + 0.5 * np.sin(2 * np.pi * f2 * t_tone)

    # Fade in/out (5ms) to prevent "clicks" at the start and end
    fade_len = int(sample_rate * 0.005)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    tone[:fade_len] *= fade_in
    tone[-fade_len:] *= fade_out

    # # Construct the "Double Tone" sequence: Tone + Silence + Tone
    # full_signal = np.concatenate([tone, t_pause, tone])
   
    # Construct the tone sequence: Silence + Tone + Silence
    full_signal = np.concatenate([t_pause, tone, t_pause])
    
    # Convert to 16-bit PCM format
    audio_data = (full_signal * 32767).astype(np.int16)

    # Write to WAV file
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)  # Mono
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        for sample in audio_data:
            f.writeframes(struct.pack('<h', sample))

    print(f"✅ Success: '{filename}' generated for digit '{digit}'")

def main():
    valid_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '#']

    parser = argparse.ArgumentParser(description="Generate DTMF tone audio files")
    parser.add_argument('--output', '-o', type=str, default='dtmf_tone.wav',
                        help="Output WAV filename (default: dtmf_tone.wav)")
    parser.add_argument('--digit', '-d', type=str, default='5', choices=valid_digits,
                        help="DTMF digit to generate (default: 5)")
    parser.add_argument('--duration', type=int, default=300,
                        help="Tone duration in milliseconds (default: 300)")
    parser.add_argument('--pause', type=int, default=1000,
                        help="Pause duration in milliseconds (default: 1000)")
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help="Sample rate in Hz (default: 44100)")

    args = parser.parse_args()

    generate_dtmf_file(args.output, args.digit, args.duration, args.pause, args.sample_rate)


if __name__ == "__main__":
    main()