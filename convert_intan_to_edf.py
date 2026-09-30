# -*- coding: utf-8 -*-
"""Convert Intan RHX 'one file per channel' recording to EDF for sleep-staging tools.

Channel map (per user):
    amp-A-001.dat -> EMG1
    amp-A-002.dat -> EMG2
    amp-A-003.dat -> EEG1 (primary, named with 'RF' for REST)
    amp-A-004.dat -> EEG2

Outputs:
    edf/rest_mouse02.edf      channels: RF_EEG, EEG2, EMG, EMG2        (for REST)
    edf/faster2_mouse02.edf   channels: EEG01, EMG01                   (for FASTER2)
    results/qa_overview.png   10 s traces + PSD sanity figure
"""
import os
import numpy as np
from scipy import signal
from datetime import datetime, timedelta

import pyedflib

REC_DIR = r"D:\code\eeg-emg\02_260924_161606"
OUT_DIR = r"D:\code\eeg-emg\edf"
FIG_DIR = r"D:\code\eeg-emg\results"
FS_IN = 30000
FS_OUT = 1000
UV_PER_BIT = 0.195  # Intan amplifier conversion factor (RHS, gain index 0)
START = datetime(2026, 9, 24, 16, 16, 6)  # from folder name 02_260924_161606

CHANNELS = {  # file -> (label, kind)
    # User-confirmed mapping (verified 2026-09-28 by delta NREM/Wake ratio,
    # same-channel delta-HF correlation, and cross-channel structure):
    # A-001/A-002 = EMG, A-003/A-004 = EEG.
    "amp-A-001.dat": ("EMG1", "emg"),
    "amp-A-002.dat": ("EMG2", "emg"),
    "amp-A-003.dat": ("EEG1", "eeg"),
    "amp-A-004.dat": ("EEG2", "eeg"),
}


def resample_stream(path, fs_in, fs_out, up=1, down=30, block_sec=10.0, ov_sec=1.0):
    """Chunked anti-aliased decimation (int16 file -> float32 µV at fs_out)."""
    n_samples = os.path.getsize(path) // 2
    block = int(block_sec * fs_in)
    ov = int(ov_sec * fs_in)
    n_out = int(np.ceil(n_samples * fs_out / fs_in))
    out = np.empty(n_out, dtype=np.float32)
    pos_in = 0
    pos_out = 0
    first = True
    while pos_in < n_samples:
        start = max(0, pos_in - ov)
        end = min(n_samples, pos_in + block)
        x = np.fromfile(path, dtype=np.int16, count=end - start, offset=start * 2).astype(np.float64)
        y = signal.resample_poly(x, up, down)
        # samples of this block that are new (not the overlap head)
        head = 0 if first else int(ov * fs_out / fs_in)
        y_new = y[head:]
        i0 = int(np.floor(start * fs_out / fs_in)) + head
        i1 = min(n_out, i0 + len(y_new))
        out[i0:i1] = y_new[: i1 - i0]
        pos_out = i1
        pos_in = end
        first = False
    return out[:pos_out]


def write_edf(path, sigs, fs, start, ch_names):
    n = len(sigs[0])
    f = pyedflib.EdfWriter(path, len(sigs), file_type=pyedflib.FILETYPE_EDFPLUS)
    ch_headers = []
    for name in ch_names:
        ch_headers.append({
            "label": name,
            "dimension": "mV",
            "sample_frequency": fs,
            "physical_min": -6.388,   # ±32768 × 0.195 µV (Intan full scale)
            "physical_max": 6.388,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "Intan RHS",
            "prefilter": "DSP 1Hz HP + Bessel 0.1-7500Hz",
        })
    f.setStartdatetime(start)
    f.setSignalHeaders(ch_headers)
    data = np.vstack(sigs) / 1000.0  # µV -> mV
    f.writeSamples(data)
    f.close()
    return n / fs


def notch50(sigs_uv, fs=FS_OUT):
    """Zero-phase notch at 50 Hz (+ harmonic) — line noise inflates EMG-band
    features in tools that don't notch internally (FASTER2)."""
    out = {}
    for k, x in sigs_uv.items():
        y = x.astype(np.float64)
        for f0 in (50.0, 100.0):
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            y = signal.filtfilt(b, a, y)
        out[k] = y.astype(np.float32)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    sigs_uv = {}
    cache_dir = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    for fname, (label, kind) in CHANNELS.items():
        cache = os.path.join(cache_dir, f"{os.path.splitext(fname)[0]}_{FS_OUT}.npy")
        if os.path.exists(cache):
            sigs_uv[label] = np.load(cache)
            print(f"cached {label}: {len(sigs_uv[label])} samples", flush=True)
            continue
        p = os.path.join(REC_DIR, fname)
        print(f"resampling {fname} -> {label} ...", flush=True)
        sigs_uv[label] = resample_stream(p, FS_IN, FS_OUT, down=30)  # 30 kHz -> 1 kHz
        np.save(cache, sigs_uv[label])
        print(f"  done: {len(sigs_uv[label])} samples ({len(sigs_uv[label])/FS_OUT:.1f} s), "
              f"std={sigs_uv[label].std():.1f} uV", flush=True)

    n = min(len(v) for v in sigs_uv.values())
    for k in sigs_uv:
        sigs_uv[k] = sigs_uv[k][:n]
    sigs_uv = notch50(sigs_uv)
    dur = n / FS_OUT

    # REST EDF: EEG channel must contain 'RF', EMG must contain 'EMG'
    order_rest = ["EEG1", "EEG2", "EMG1", "EMG2"]
    names_rest = ["RF_EEG", "EEG2", "EMG", "EMG2"]
    d_rest = os.path.join(OUT_DIR, "rest_mouse02.edf")
    dur = write_edf(d_rest, [sigs_uv[k] for k in order_rest], FS_OUT, START, names_rest)
    print(f"wrote {d_rest}  ({dur:.1f} s)")

    # FASTER2 EDF: channels exactly EEG<devid> / EMG<devid>, devid = 01
    order_f2 = ["EEG1", "EMG1"]
    names_f2 = ["EEG01", "EMG01"]
    d_f2 = os.path.join(OUT_DIR, "faster2_mouse02.edf")
    write_edf(d_f2, [sigs_uv[k] for k in order_f2], FS_OUT, START, names_f2)
    print(f"wrote {d_f2}")

    end = START + timedelta(seconds=dur)
    print(f"start {START}  end {end}  duration {dur:.1f} s ({dur/3600:.2f} h)")

    # QA figure: 10 s traces + PSD per channel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = slice(int(3600 * FS_OUT), int(3610 * FS_OUT))  # a 10 s window 1 h in
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for j, label in enumerate(["EMG1", "EMG2", "EEG1", "EEG2"]):
        x = sigs_uv[label][sel]
        f, pxx = signal.welch(x, fs=FS_OUT, nperseg=4 * FS_OUT)
        axes[0, j].plot(np.arange(len(x)) / FS_OUT, x, lw=0.5)
        axes[0, j].set_title(f"{label} ({kind})")
        axes[1, j].semilogy(f, pxx)
        axes[1, j].set_xlim(0, 100)
        axes[1, j].set_xlabel("Hz")
    fig.suptitle(f"QA: mouse02  {START:%Y-%m-%d %H:%M:%S} + 1 h  (10 s trace & PSD)")
    fig.tight_layout()
    qa = os.path.join(FIG_DIR, "qa_overview.png")
    fig.savefig(qa, dpi=120)
    print(f"wrote {qa}")


if __name__ == "__main__":
    main()
