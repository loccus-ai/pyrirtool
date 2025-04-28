#!/bin/bash

SUFFIXES=(
    "20250424_112112_plus0000_RE5292d71d0c42dfdc79b594a1d9445dac"
    "20250424_120531_plus0000_REee2be08ee3d288952d3eb99a7f01b037"
    "20250424_143318_plus0000_REe1154787a2dd664a74ba2f74913c0418"
    "20250424_151734_plus0000_RE0c8599d36d8af7912e214790bc6d9f30"
    "20250424_160149_plus0000_RE97612a7110810e3c9b81a84e6458d303"
    "20250424_164603_plus0000_REafdc4c65a70ba6e10847e2ef61a3c6c0"
    "20250425_132539_plus0000_RE16a2ba592b2d81c465df71d080bc5e98"
    "20250425_141003_plus0000_RE0c9f59f63951ea9a8f3ddaab5f75335e"
    "20250425_152558_plus0000_RE62c028ed146fb9363f467c4df494c1a0"
    "20250425_172028_plus0000_RE84763b9fcbee4fb2f2fdf603d95e9e00"
)

for suffix in "${SUFFIXES[@]}"; do
    # Nokia 3310 recordings (mono)
    # echo "python align_audio.py --original /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_original.wav \
    #     --recorded /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_nokia3310.wav \
    #     --jsonl /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_info.jsonl \
    #     --output_dir /data/audio/EXP28-tel/recordings/ \
    #     --output_suffix .${suffix}.nokia3310.wav \
    #     --basedir /data/audio/EXP28-tel/ &> /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_align.log"

    # Twilio recordings (stereo, get channel 1)
    echo "python align_audio.py --original /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_original.wav \
        --recorded /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_twilio.wav \
        --jsonl /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_info.jsonl \
        --output_dir /data/audio/EXP28-tel/recordings/ \
        --output_suffix .${suffix}.twilio.wav \
        --channel 1 \
        --basedir /data/audio/EXP28-tel/ &> /data/audio/EXP28-tel/concatenated_sweep/recordings/${suffix}_align_twilio.log"
done