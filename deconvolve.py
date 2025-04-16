# %%
import os
import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt

# modules from this software
import stimulus as stim
import _parseargs as parse
import utils as utils
import sys
import soundfile as sf
import librosa

# %%
# --- Parse command line arguments and check defaults
recording='../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1_nokia_recording.wav'
original='../recordings/REcbdb0b438839daebf0f87bb84af9d989_sigtest_fs16000_ss3_es1/sigtest_fs16000_ss3_es1.wav'
sys.argv = 'measure.py --fs 16000 -ss 3 -es 1'.split()
flag_defaultsInitialized = parse._checkdefaults()
args = parse._parse()
parse._defaults(args)
# ------------------------

# %%
# Create a test signal object, and generate the excitation
testStimulus = stim.stimulus('sinesweep', args.fs);
testStimulus.generate(args.fs, args.duration, args.amplitude,args.reps,args.startsilence, args.endsilence, args.sweeprange)

# %%
x, fs = librosa.load(recording, sr=args.fs)

# %%
x = np.expand_dims(x, 1)
x.shape

# %%
# Deconvolve
impulse_response = testStimulus.deconvolve(x)

# %%
maxval = np.max(impulse_response)
minval = np.min(impulse_response)
taxis = np.arange(0,impulse_response.shape[0]/fs,1/fs)

# Plot all on a single figure
# plt.figure(figsize = (10,6))
# plt.plot(taxis,RIR)
# plt.ylim((minval+0.05*minval,maxval+0.05*maxval))

# Plot them as subplots
numplots = impulse_response.shape[1]
#height = numplots*3
#fig = plt.figure(figsize = (10,height))
for idx in range(numplots):
    fig = plt.figure(figsize = (9,3))
    plt.plot(impulse_response[:,idx])
    plt.ylim((minval+0.05*minval,maxval+0.05*maxval))
    plt.title('RIR Microphone '+ str(idx + 1))
    #ax = fig.add_subplot(numplots,1,idx+1)
    #plt.plot(taxis,RIR[:,idx])

# %%
sf.write('RIR.wav', impulse_response, fs)

# %%
x_original, fs = librosa.load('../sample_1min.wav', sr=args.fs)

# %%
y_simulated = np.convolve(impulse_response[:,0], x_original)
sf.write('simulated.wav', y_simulated, fs)

# %% [markdown]
# # Reverberation time T60

# %%
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

    TTdecay_sample = int(TTdecay * fs)
    RIRtrimmed = impulse_response_ori.copy()
    RIRtrimmed = RIRtrimmed[max_position-TTdecay_sample:max_position+TTdecay_sample,:]
    RIRtrimmed0 = impulse_response_ori.copy()
    RIRtrimmed0[0:max_position-TTdecay_sample,:] = 0
    RIRtrimmed0[max_position+TTdecay_sample:,:] = 0


    if plot:
        fig = plt.figure(figsize = (9,3))
        t = np.arange(0, impulse_response.shape[0]) / fs
        plt.plot(t, decay_curve)
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Decay curve (dB)')
        plt.axhline(y=-DBdecay, color='r', linestyle='--', label=f'-{DBdecay} dB')
        plt.legend()
        plt.title(title)
        plt.show()


    return TTdecay, decay_curve, RIRtrimmed, RIRtrimmed0

# %%
T60, decay_curve, RIRtrimmed, RIRtrimmed0 = compute_tdecay(impulse_response, fs, 60, plot=True, title='Decay curve RIR T60')

# %%

sf.write('RIR_trimmed.wav', RIRtrimmed, fs)
y_simulated_trimmed = np.convolve(RIRtrimmed[:,0], x_original)
sf.write('simulated_trimmed.wav', y_simulated_trimmed, fs)

sf.write('RIR_trimmed0.wav', RIRtrimmed0, fs)
y_simulated_trimmed0 = np.convolve(RIRtrimmed0[:,0], x_original)
sf.write('simulated_trimmed0.wav', y_simulated_trimmed0, fs)


# %%
T30, decay_curve_T30, RIRtrimmed_T30, RIRtrimmed0_T30 = compute_tdecay(impulse_response, fs, 30, plot=True, title='Decay curve RIR T30')

# %%
sf.write('RIR_trimmed_T30.wav', RIRtrimmed_T30, fs)
y_simulated_trimmed_T30 = np.convolve(RIRtrimmed_T30[:,0], x_original)
sf.write('simulated_trimmed_T30.wav', y_simulated_trimmed_T30, fs)

sf.write('RIR_trimmed0_T30.wav', RIRtrimmed0_T30, fs)
y_simulated_trimmed0_T30 = np.convolve(RIRtrimmed0_T30[:,0], x_original)
sf.write('simulated_trimmed0_T30.wav', y_simulated_trimmed0_T30, fs)



import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument('input_directory', type=str, help='Path to the input directory')
    parser.add_argument('original_audio', type=str, help='Path to the original audio file')
    parser.add_argument('recorded_audio', type=str, help='Path to the recorded audio file')
    parser.add_argument('command', type=str, help='Command to execute')

    return parser.parse_args()

def process(input_directory, original_audio, recorded_audio, command):
    print(f"Input Directory: {input_directory}")
    print(f"Original Audio: {original_audio}")
    print(f"Recorded Audio: {recorded_audio}")
    print(f"Command: {command}")

    # Implement your processing logic here
    if command == "play":
        print("Playing audio files...")
        # Add logic to play the audio files
    elif command == "compare":
        print("Comparing audio files...")
        # Add logic to compare the audio files
    else:
        print(f"Unknown command: {command}")

def main():
    args = parse_arguments()
    process(
        input_directory=args.input_directory,
        original_audio=args.original_audio,
        recorded_audio=args.recorded_audio,
        command=args.command
    )

if __name__ == "__main__":
    main()