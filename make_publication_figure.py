# -*- coding: utf-8 -*-
"""Publication-quality multi-panel figure: hypnogram, EEG spectrogram,
slow-wave activity (delta power) and EMG amplitude on a shared time axis."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.io import loadmat
from scipy import signal
from datetime import datetime, timedelta

OUT = r"D:\code\eeg-emg\results"
CACHE = r"D:\code\eeg-emg\edf\cache"
START = datetime(2026, 9, 24, 16, 16, 6)
EP = 4.0
FS = 1000
TOOLS = ["REST", "somnotate", "AccuSleePy"]

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

STATE_COLORS = {1: "#DC267F", 2: "#648FFF", 3: "#FFB000"}
STATE_NAMES = {1: "Wake", 2: "NREM", 3: "REM"}
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "eeg_blue", ["#04072E", "#0B2FA6", "#1E78E0", "#35C8F0", "#F2EE5C"])
WAVE_GRAY_EEG = "#696969"
WAVE_GRAY_EMG = "#8C8C8C"


def load_rest():
    d = loadmat(r"D:\code\eeg-emg\edf\rest\rest_mouse02_REST_V2.0.mat")
    s = d["score"].flatten().astype(int)
    m = {1: 1, 2: 2, 3: 3, 4: 0}
    return np.array([m[x] for x in s])


def load_somnotate():
    ends, labels, prev = [], [], 0.0
    for line in open(r"D:\code\eeg-emg\somnotate_run\data\processed\mouse02_automated.hyp"):
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        try:
            end = float(parts[-1])
        except ValueError:
            continue
        ends.append(end); labels.append(parts[0]); prev = end
    m = {"awake": 1, "non-REM": 2, "REM": 3}
    n = int(prev / EP)
    out = np.zeros(n, dtype=int)
    prev = 0.0
    for lab, end in zip(labels, ends):
        out[int(prev / EP):min(int(end / EP), n)] = m.get(lab, 0)
        prev = end
    return out


def load_accusleepy():
    df = pd.read_csv("D:/code/eeg-emg/accusleepy_run/mouse02_accusleepy_labels.csv")
    m = {1: 3, 2: 1, 3: 2}
    return np.array([m.get(x, 0) for x in df["brain_state"]])


def load_notched(fname):
    x = np.load(os.path.join(CACHE, fname)).astype(np.float64)
    for f0 in (50.0, 100.0):
        b, a = signal.iirnotch(f0, Q=30, fs=FS)
        x = signal.filtfilt(b, a, x)
    return x


rest = load_rest()
som = load_somnotate()
acc = load_accusleepy()
n = min(len(rest), len(som), len(acc))
vecs = {"REST": rest[:n], "somnotate": som[:n], "AccuSleePy": acc[:n]}
hours = np.arange(n) * EP / 3600.0
dur_h = hours[-1]

eeg = load_notched("amp-A-003_1000.npy")[:n * 4000]
emg = load_notched("amp-A-001_1000.npy")[:n * 4000]

# ---- spectrogram (1 s window, 0.5 s hop) ------------------------------------
nwin = int(2 * FS)                    # 2 s window
hop = int(0.5 * FS)                   # 0.5 s hop
nframes = (len(eeg) - nwin) // hop + 1
idx = np.arange(nwin)[None, :] + hop * np.arange(nframes)[:, None]
freqs, Pxx = signal.welch(eeg[idx], fs=FS, nperseg=nwin, axis=1)
mask = (freqs >= 0.5) & (freqs <= 30)
S = np.log10(Pxx[:, mask].T)
t_spec = (np.arange(nframes) * hop + nwin / 2) / FS / 3600.0
vmin, vmax = np.percentile(S, [2, 98])

# ---- waveform envelopes (1 s min/max — visible form of the raw traces) ------
w = FS
m_env = len(eeg) // w * w
t_env = np.arange(m_env // w) * 1.0 / 3600.0
E_env = eeg[:m_env].reshape(-1, w)
M_env = emg[:m_env].reshape(-1, w)
eeg_lo, eeg_hi = E_env.min(1), E_env.max(1)
emg_lo, emg_hi = M_env.min(1), M_env.max(1)

# =============================================================================
fig = plt.figure(figsize=(7.09, 5.6))          # 180 mm double column
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 2.3, 1.05, 1.05],
                      hspace=0.24, left=0.095, right=0.895, top=0.955, bottom=0.09)

# ---- a: hypnogram -----------------------------------------------------------
axH = fig.add_subplot(gs[0])
order = [(3, 0, "REM"), (2, 1, "NREM"), (1, 2, "Wake")]   # y position per state
for s, ypos, _ in order:
    axH.fill_between(hours, ypos, ypos + 0.92, where=(vecs["REST"] == s),
                     step="mid", color=STATE_COLORS[s], lw=0)
axH.set_yticks([0.46, 1.46, 2.46])
axH.set_yticklabels(["REM", "NREM", "Wake"])
axH.set_ylim(-0.15, 3.4)
axH.set_xlim(0, dur_h)
axH.set_xticklabels([])
for sp in ["left", "bottom"]:
    axH.spines[sp].set_visible(False)
axH.tick_params(length=0)

# ---- b: spectrogram ---------------------------------------------------------
axS = fig.add_subplot(gs[1])
im = axS.imshow(S, aspect="auto", origin="lower", cmap=SPEC_CMAP,
                extent=[0, dur_h, freqs[mask][0], freqs[mask][-1]],
                vmin=vmin, vmax=vmax, interpolation="nearest")
axS.set_ylabel("Frequency (Hz)")
axS.set_yticks([1, 5, 10, 20, 30])
axS.set_xlim(0, dur_h)
axS.set_xticklabels([])
cax = axS.inset_axes([1.012, 0.0, 0.022, 1.0])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"log$_{10}$ EEG power ($\mu$V$^2$/Hz)", fontsize=7)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)

# ---- c: EEG raw waveform (1 s min/max envelope) ------------------------------
axD = fig.add_subplot(gs[2])
axD.fill_between(hours, -1500, 1500, where=(vecs["REST"] == 2), step="mid",
                 color="#648FFF", alpha=0.12, lw=0)
axD.fill_between(t_env, eeg_lo, eeg_hi, color=WAVE_GRAY_EEG, lw=0)
axD.set_ylim(-1500, 1500)
axD.set_xlim(0, dur_h)
axD.set_ylabel("EEG ($\\mu$V)\nA-003")
axD.set_yticks([-1000, 0, 1000])
axD.set_xticklabels([])
axD.text(dur_h * 0.995, 1250, "NREM", ha="right", fontsize=6, color="#3D6BD6")

# ---- d: EMG raw waveform (1 s min/max envelope) ------------------------------
axE = fig.add_subplot(gs[3])
axE.fill_between(hours, -2500, 2500, where=(vecs["REST"] == 1), step="mid",
                 color="#DC267F", alpha=0.12, lw=0)
axE.fill_between(t_env, emg_lo, emg_hi, color=WAVE_GRAY_EMG, lw=0)
axE.set_ylim(-2500, 2500)
axE.set_xlim(0, dur_h)
axE.set_ylabel("EMG ($\\mu$V)\nA-001")
axE.set_yticks([-2000, 0, 2000])
axE.set_xlabel("Time since recording start (h)")
axE.text(dur_h * 0.995, 2100, "Wake", ha="right", fontsize=6, color="#B02163")

# ---- clock-time secondary axis on top of hypnogram --------------------------
sec = axH.secondary_xaxis("top", functions=(lambda h: h, lambda h: h))
tk = np.arange(0, dur_h, 0.5)
sec.set_xticks(tk)
sec.set_xticklabels([(START + timedelta(hours=float(t))).strftime("%H:%M") for t in tk],
                    fontsize=6.5)
sec.tick_params(length=2, width=0.5, pad=1)
sec.spines["top"].set_visible(False)

# ---- panel letters ----------------------------------------------------------
for ax, letter in [(axH, "a"), (axS, "b"), (axD, "c"), (axE, "d")]:
    ax.text(-0.078, 1.02, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

fig.savefig(os.path.join(OUT, "fig_publication.png"), dpi=350)
fig.savefig(os.path.join(OUT, "fig_publication.pdf"))
print("saved fig_publication.png / .pdf")
