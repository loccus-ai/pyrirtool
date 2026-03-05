#!/usr/bin/env python3
"""
Obtain Room Impulse Response (RIR) from original and recorded sweep signals
via FFT-based deconvolution with Tikhonov regularisation.

Usage:
    python obtain_rir_from_sweep.py --sweep wav/_sine_sweep.wav --recorded wav/_rec_sine_sweep.wav -o rir_output.wav
    python obtain_rir_from_sweep.py --sweep wav/_sine_sweep.wav --recorded wav/_rec_sine_sweep.wav --reference wav/RIR.wav -o rir_output.wav
"""

import argparse
import os
import numpy as np
import soundfile as sf


def compute_rir_deconv(sweep, recorded, fs, reg_dB=-30):
    """
    Compute room impulse response via frequency-domain deconvolution.

    H(f) = Y(f) / X(f)  with optional Tikhonov regularisation.

    Parameters
    ----------
    sweep : array_like
        Original excitation signal.
    recorded : array_like
        Recorded signal (system output).
    fs : int
        Sample rate.
    reg_dB : float or None
        Regularisation floor in dB relative to max |X(f)|^2.
        Set to None for plain spectral division (no regularisation).
    """
    n_fft = 2 ** int(np.ceil(np.log2(len(sweep) + len(recorded) - 1)))

    X = np.fft.rfft(sweep, n=n_fft)
    Y = np.fft.rfft(recorded, n=n_fft)

    if reg_dB is not None:
        eps = 10 ** (reg_dB / 10) * np.max(np.abs(X) ** 2)
        H = (Y * np.conj(X)) / (np.abs(X) ** 2 + eps)
    else:
        H = Y / (X + 1e-30)

    rir = np.fft.irfft(H, n=n_fft)
    return rir[:len(sweep) + len(recorded) - 1]


def trim_rir(rir, fs, pre_s=0.01, post_s=1.0):
    """Normalise and trim RIR around its peak."""
    rir_norm = rir / np.max(np.abs(rir))
    peak = np.argmax(np.abs(rir_norm))
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    start = max(0, peak - pre)
    end = min(len(rir_norm), peak + post)
    return rir_norm, rir_norm[start:end], peak


def normalised_xcorr(a, b):
    """Peak normalised cross-correlation between two signals."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm < 1e-30:
        return 0.0
    return np.max(np.correlate(a, b, "full")) / norm


def log_spectral_distance(a, b, n_fft=4096):
    """Mean log-spectral distance (dB) between two impulse responses."""
    n = min(len(a), len(b))
    A = np.maximum(np.abs(np.fft.rfft(a[:n], n=n_fft)), 1e-12)
    B = np.maximum(np.abs(np.fft.rfft(b[:n], n=n_fft)), 1e-12)
    return np.sqrt(np.mean((20 * np.log10(A) - 20 * np.log10(B)) ** 2))


def find_optimal_reg_dB(sweep, recorded, fs, rir_ref_trimmed,
                        reg_range=(-80, 0), step=1.0):
    """
    Sweep reg_dB and find the value that maximises normalised
    cross-correlation with the reference RIR.

    Returns (best_reg_dB, best_ncc, results_dict).
    """
    reg_values = np.arange(reg_range[0], reg_range[1] + step, step)
    ncc_scores = np.empty(len(reg_values))
    lsd_scores = np.empty(len(reg_values))

    for i, reg in enumerate(reg_values):
        _, rir_cand, _ = trim_rir(compute_rir_deconv(sweep, recorded, fs, reg_dB=reg), fs)
        ncc_scores[i] = normalised_xcorr(rir_cand, rir_ref_trimmed)
        lsd_scores[i] = log_spectral_distance(rir_cand, rir_ref_trimmed)

    best_idx = np.argmax(ncc_scores)
    best_reg = reg_values[best_idx]
    return best_reg, ncc_scores[best_idx], {
        "reg_values": reg_values,
        "ncc_scores": ncc_scores,
        "lsd_scores": lsd_scores,
    }


def energy_decay_curve(ir, fs):
    """Schroeder backward integration of squared impulse response."""
    energy = ir ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    return 10 * np.log10(edc / np.max(edc) + 1e-12)


def estimate_rt(edc_dB, fs, db_decay):
    """Estimate reverberation time from EDC for a given dB decay threshold."""
    idx = np.where(edc_dB <= -db_decay)[0]
    if len(idx) == 0:
        return None
    cross_idx = idx[0]
    if cross_idx > 0:
        y1, y2 = edc_dB[cross_idx - 1], edc_dB[cross_idx]
        frac = (y1 + db_decay) / (y1 - y2)
        return (cross_idx - 1 + frac) / fs
    return cross_idx / fs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Obtain RIR from original and recorded sweep signals "
                    "via FFT-based deconvolution with Tikhonov regularisation.")
    parser.add_argument("--sweep", required=True,
                        help="Path to the original sweep WAV file")
    parser.add_argument("--recorded", required=True,
                        help="Path to the recorded sweep WAV file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output RIR WAV file path")
    parser.add_argument("--reference",
                        help="Optional reference RIR WAV for optimal reg_dB search")
    parser.add_argument("--reg-db", type=float, default=-30,
                        help="Regularisation floor in dB (default: -30). "
                             "Ignored when --reference is provided (optimal value is searched).")
    parser.add_argument("--pre-s", type=float, default=0.01,
                        help="Seconds before RIR peak to keep (default: 0.01)")
    parser.add_argument("--post-s", type=float, default=1.0,
                        help="Seconds after RIR peak to keep (default: 1.0)")
    parser.add_argument("--no-trim", action="store_true",
                        help="Save full-length (untrimmed) RIR")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load signals
    sweep, fs = sf.read(args.sweep)
    rec, fs_rec = sf.read(args.recorded)
    assert fs == fs_rec, (
        f"Sample rate mismatch: sweep={fs} Hz, recorded={fs_rec} Hz")

    print(f"Sample rate:     {fs} Hz")
    print(f"Original sweep:  {len(sweep)} samples ({len(sweep)/fs:.3f} s)")
    print(f"Recorded sweep:  {len(rec)} samples ({len(rec)/fs:.3f} s)")

    # Determine regularisation
    reg_dB = args.reg_db

    if args.reference:
        rir_ref, fs_ref = sf.read(args.reference)
        assert fs == fs_ref, (
            f"Sample rate mismatch: sweep={fs} Hz, reference={fs_ref} Hz")

        # Trim reference around its peak
        rir_ref_norm = rir_ref / np.max(np.abs(rir_ref))
        peak_ref = np.argmax(np.abs(rir_ref_norm))
        pre = int(args.pre_s * fs)
        post = int(args.post_s * fs)
        start_ref = max(0, peak_ref - pre)
        end_ref = min(len(rir_ref_norm), peak_ref + post)
        rir_ref_trimmed = rir_ref_norm[start_ref:end_ref]

        print(f"Reference RIR:   {len(rir_ref)} samples ({len(rir_ref)/fs:.3f} s)")
        print("Searching for optimal reg_dB...")

        reg_dB, best_ncc, results = find_optimal_reg_dB(
            sweep, rec, fs, rir_ref_trimmed)
        best_lsd_idx = np.argmin(results["lsd_scores"])
        best_lsd = results["reg_values"][best_lsd_idx]

        print(f"Optimal reg_dB (NCC): {reg_dB:.0f} dB  (NCC = {best_ncc:.6f})")
        print(f"Optimal reg_dB (LSD): {best_lsd:.0f} dB  "
              f"(LSD = {results['lsd_scores'][best_lsd_idx]:.2f} dB)")

    # Compute RIR
    rir_full = compute_rir_deconv(sweep, rec, fs, reg_dB=reg_dB)
    rir_norm, rir_trimmed, peak_idx = trim_rir(
        rir_full, fs, pre_s=args.pre_s, post_s=args.post_s)

    print(f"RIR peak at:     {peak_idx/fs:.3f} s (sample {peak_idx})")
    print(f"Trimmed RIR:     {len(rir_trimmed)} samples ({len(rir_trimmed)/fs:.3f} s)")

    # Reverberation time estimates
    rir_from_peak = rir_trimmed[np.argmax(np.abs(rir_trimmed)):]
    edc_dB = energy_decay_curve(rir_from_peak, fs)
    for db in (30, 60):
        rt = estimate_rt(edc_dB, fs, db)
        if rt is not None:
            print(f"RT{db}:            {rt:.4f} s")
        else:
            print(f"RT{db}:            decay does not reach -{db} dB")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_rir = rir_norm if args.no_trim else rir_trimmed
    sf.write(args.output, output_rir.astype(np.float32), fs)
    print(f"RIR saved to:    {args.output} (reg_dB={reg_dB:.0f})")


if __name__ == "__main__":
    main()
