# ================================================================
# Room impulse response measurement with an exponential sine sweep
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# ================================================================
import os
import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import soundfile as sf
import json
from scipy.io.wavfile import write as wavwrite

# modules from this software
import stimulus as stim
from deconvolve import replace_extension

def process(fs, duration, amplitude, reps, startsilence, endsilence, sweeprange):
    print(f'Generating sweep {fs}Hz, duration {duration}s, {reps} reps, start silence {startsilence}s, end silence {endsilence}, sweep range {sweeprange}')
    # Create a test signal object, and generate the excitation
    testStimulus = stim.stimulus('sinesweep', fs)
    testStimulus.generate(fs, duration, amplitude,reps,startsilence, endsilence, sweeprange)
    return testStimulus

    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Setting the parameters for RIR measurement using exponential sine sweep \n ----------------------------------------------------------------------')
    #---
    parser.add_argument("-f", "--fs", type = int, help=" The sampling rate (make sure it matches that of your audio interface). Default: 44100 Hz.", default = 44100)
    #---
    parser.add_argument("-dur", "--duration", type = int, help=" The duration of a single sweep. Default: 15 seconds.", default = 10)
    #---
    parser.add_argument("-r", "--reps", type = int, help = "Number of repetitions of the sinesweep. Default: 1.", default = 1)
    #---
    parser.add_argument("-a", "--amplitude", type = float, help = "Amplitude of the sine. Default: 0.7",default = 0.2)
    #---
    parser.add_argument("-ss", "--startsilence", type = int, help = "Duration of silence at the start of a sweep, in seconds. Default: 1.", default = 3)

    parser.add_argument("-frange", "--sweeprange", nargs='+', type=int, help = "Frequency range of the sweep", default = [0, 0])
    #---
    parser.add_argument("-es", "--endsilence", type = int, help = "Duration of silence at the end of a sweep, in seconds. Default: 1.", default = 6)
    #---
    parser.add_argument("-o", "--output", type = str, help = "Output sweep audio file")

    return parser.parse_args()


def main():
    args = parse_arguments()
    d = vars(args)
    testStimulus = process(fs=args.fs, 
                           duration=args.duration,
                           amplitude=args.amplitude,
                           reps=args.reps,
                           startsilence=args.startsilence,
                           endsilence=args.endsilence,
                           sweeprange=args.sweeprange)
    
    wavwrite(filename=args.output, rate=args.fs, data=testStimulus.signal)
    # sf.write(args.output, testStimulus.signal, args.fs)
    
    json_output = replace_extension(args.output, '.json')
    with open(json_output, 'w') as json_file:
        json.dump(d, json_file, indent=4)
    print(f'Sweep saved to {args.output}, parameters to {json_output}')
    
if __name__ == "__main__":
    main()