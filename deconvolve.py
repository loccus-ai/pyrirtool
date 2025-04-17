import os
import sys
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import json
from matplotlib import pyplot as plt

# modules from this software
import stimulus as stim
import _parseargs as parse
import utils as utils


# 
def replace_extension(filename, new_extension):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    # Concatenate the name with the new extension
    new_filename = name + new_extension
    return new_filename

# 
def compute_tdecay(impulse_response_ori, sample_rate, DBdecay, plot=False, title=''):
    
    max_position = np.argmax(impulse_response_ori)

    
    impulse_response = impulse_response_ori[max_position:, :]
    # Step 1: Calculate the energy of the impulse response
    energy = impulse_response ** 2

    # Step 2: Compute the cumulative sum in reverse
    cumulative_energy = np.cumsum(energy[::-1])[::-1]

    # Step 3: Convert the cumulative sum to a decay curve in dB
    decay_curve = 10 * np.log10(cumulative_energy / np.max(cumulative_energy))

    # Step 4: Find the time point where the decay curve crosses -Tdecay dB
    t_Tdecay_point = np.where(decay_curve <= -DBdecay)[0]

    if len(t_Tdecay_point) == 0:
        print("Warning: The decay curve does not reach -Tdecay dB")
        return None

    # Step 5: Interpolate to get a more accurate TTdecay
    index_Tdecay = t_Tdecay_point[0]

    # Calculate the exact point with linear interpolation
    if index_Tdecay > 0:
        t1 = index_Tdecay - 1
        t2 = index_Tdecay
        y1 = decay_curve[t1]
        y2 = decay_curve[t2]
        t_Tdecay = t1 + (y1 + DBdecay) / (y1 - y2)
    else:
        t_Tdecay = index_Tdecay

    # Convert samples to time
    TTdecay = t_Tdecay / sample_rate
    print(f"Reverberation time (T{DBdecay}) is {TTdecay} seconds")

    TTdecay_sample = int(TTdecay * sample_rate)
    RIRtrimmed = impulse_response_ori.copy()
    RIRtrimmed = RIRtrimmed[max_position-TTdecay_sample:max_position+TTdecay_sample,:]
    RIRtrimmed0 = impulse_response_ori.copy()
    RIRtrimmed0[0:max_position-TTdecay_sample,:] = 0
    RIRtrimmed0[max_position+TTdecay_sample:,:] = 0

    if plot:
        fig = plt.figure(figsize = (9,3))
        t = np.arange(0, impulse_response.shape[0]) / sample_rate
        plt.plot(t, decay_curve)
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Decay curve (dB)')
        plt.axhline(y=-DBdecay, color='r', linestyle='--', label=f'-{DBdecay} dB')
        plt.legend()
        plt.title(title)
        plt.show()

    return TTdecay, decay_curve, RIRtrimmed, RIRtrimmed0


def process(recorded_audio, sweep_conf_json, plot, Treverb=[30, 60]):
    # 
    # recorded_audio='../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1_nokia_recording.wav'
    # original_audio='../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1.wav'
    # sys.argv = 'measure.py --fs 16000 -ss 3 -es 1'.split()
    with open(sweep_conf_json, 'r') as f:
        sweep_params = json.load(f)
    # 
    # sys.argv = command.split()
    # flag_defaultsInitialized = parse._checkdefaults()
    # args = parse._parse()
    # parse._defaults(args)

    # Create a test signal object, and generate the excitation
    testStimulus = stim.stimulus('sinesweep', sweep_params['fs'])
    testStimulus.generate(sweep_params['fs'], sweep_params['duration'], sweep_params['amplitude'],sweep_params['reps'],sweep_params['startsilence'], sweep_params['endsilence'], sweep_params['sweeprange'])

    # 
    # Load recorded signal
    x, fs = librosa.load(recorded_audio, sr=sweep_params['fs'])
    x = np.expand_dims(x, 1)

    # 
    # Deconvolve
    impulse_response = testStimulus.deconvolve(x)

    # 
    maxval = np.max(impulse_response)
    minval = np.min(impulse_response)
    taxis = np.arange(0,impulse_response.shape[0]/fs,1/fs)

    if plot:
        # Plot all on a single figure
        plt.figure(figsize = (10,6))
        plt.plot(taxis,impulse_response)
        plt.ylim((minval+0.05*minval,maxval+0.05*maxval))
        plt.title(f'RIR {recorded_audio}')

    # 
    output_RIR = replace_extension(recorded_audio, '_RIR.wav')
    sf.write(output_RIR, impulse_response, fs)
    
    for T in Treverb:
        T60, decay_curve_T, RIRtrimmed_T, RIRtrimmed0_T = compute_tdecay(impulse_response, fs, T, plot=plot, title=f'Decay curve RIR T{T}')
        
        output_RIR_trimmed_T = replace_extension(recorded_audio, f'_RIR_trimmed_T{T}.wav')
        output_RIR_trimmed0_T = replace_extension(recorded_audio, f'_RIR_trimmed0_T{T}.wav')

        sf.write(output_RIR_trimmed_T, RIRtrimmed_T, fs)
        sf.write(output_RIR_trimmed0_T, RIRtrimmed0_T, fs)


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument('--recorded_audio', type=str, help='Path to the recorded audio file. Example: "../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1_nokia_recording.wav')
    parser.add_argument('--sweep_json', type=str, help='JSON file with sweep parameters. Example: "../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1.json"')
    parser.add_argument('--plot', action='store_true', help='Flag to enable plotting. Default: False')
    parser.add_argument('--Treverb', type=int, nargs='+', default=[30, 60], help='List of integers for reverberation times. Default: [30, 60]')

    return parser.parse_args()

def main():
    args = parse_arguments()
    process(
        recorded_audio=args.recorded_audio,
        sweep_conf_json=args.sweep_json,        
        plot=args.plot,
        Treverb=args.Treverb
    )

if __name__ == "__main__":
    main()