# Hiya: tools to generate concatenated audio files for easier transmission and recording.

This toolbox concatenates lists of audio files into single audio files of a maximum, given duration, using DTMF tones for separation and later synchronization. 
It also randomly insersts "sweep" signals, for later estimation of the transmission channel IR.

## Concatenate audio files
```
python concatenate_audio_files.py --input_list kk.lst --sweep_file sweep_16kHz.wav --sync_tone_file dtmf_sync_signal.freq_hash.16kHz.wav --output_dir kk --sweep_probability 0.5 --root_dir / --output_length 3600
```

## Create sync DTMF signal
```
python concatenate_audio_files.py --input_list kk.lst --sweep_file sweep_16kHz.wav --sync_tone_file dtmf_sync_signal.freq_hash.16kHz.wav --output_dir kk --sweep_probability 0.5 --root_dir / --output_length 3600
```

## Generate sweep signal
```
python generate_sweep.py --fs 16000 -o sweep_16kHz.wav
```
The configuration of the sweep signal is stored as a JSON file, this is needed when reconstructing the Impulse Response.


#### Measuring room impulse responses with ```python``` and ```sounddevice```
See **README.orig.md**