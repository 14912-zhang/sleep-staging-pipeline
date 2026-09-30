# -*- coding: utf-8 -*-
"""Publication-style raw EEG & EMG trace figure (single column, 90 mm)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta

OUT = r"D:\code\eeg-emg\results"
CACHE = r"D:\code\eeg-emg\edf\cache"
START = datetime(2026, 9, 24, 16, 16, 6)
FS = 1000
T0 = datetime(2026, 9, 24, 17, 21, 0)      # window start (clock time)
T1 = datetime(2026, 9, 24, 17, 21, 40)     # window end
GRAY = {"eeg": "#696969", "emg": "#8C8C8C"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def load_notched(fname):
    x = np.load(os.path.join(CACHE, fname)).astype(np.float64)
    for f0 in (50.0, 100.0):
        b, a = signal.iirnotch(f0, Q=30, fs=FS)
        x = signal.filtfilt(b, a, x)
    return x


i0 = int((T0 - START).total_seconds() * FS)
i1 = int((T1 - START).total_seconds() * FS)
eeg = load_notched("amp-A-003_1000.npy")[i0:i1]   # EEG = A-003
emg = load_notched("amp-A-001_1000.npy")[i0:i1]   # EMG = A-001
t = np.arange(len(eeg)) / FS                       # seconds within window

fig = plt.figure(figsize=(3.5, 2.4))               # 90 mm single column
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.32,
                      left=0.13, right=0.965, top=0.93, bottom=0.16)

axE = fig.add_subplot(gs[0])
axE.plot(t[::2], eeg[::2], color=GRAY["eeg"], lw=0.4)
axE.set_ylim(-1000, 1000)
axE.set_ylabel("EEG ($\\mu$V)")
axE.set_xticklabels([])

axM = fig.add_subplot(gs[1])
axM.plot(t[::2], emg[::2], color=GRAY["emg"], lw=0.4)
axM.set_ylim(-1000, 1000)
axM.set_ylabel("EMG ($\\mu$V)")

tick0 = T0.replace(second=0, microsecond=0)
if tick0 < T0:
    tick0 += timedelta(seconds=10)
ticks = [(tk - T0).total_seconds() for tk in
         [tick0 + timedelta(seconds=10 * k) for k in range(5)]
         if tk <= T0 + timedelta(seconds=(t[-1] - 4))]
for ax in (axE, axM):
    ax.set_xlim(0, t[-1])
    ax.set_xticks(ticks)
axM.set_xticklabels([(T0 + timedelta(seconds=s)).strftime("%H:%M:%S") for s in ticks])
axM.set_xlabel("Clock time")

for ax, letter in [(axE, "a"), (axM, "b")]:
    ax.text(-0.115, 1.05, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

fig.savefig(os.path.join(OUT, "fig_raw_traces.png"), dpi=350)
fig.savefig(os.path.join(OUT, "fig_raw_traces.pdf"))
print("saved fig_raw_traces.png / .pdf")
