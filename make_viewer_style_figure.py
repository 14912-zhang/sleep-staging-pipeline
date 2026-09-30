# -*- coding: utf-8 -*-
"""Viewer-style figure: confidence + state rasters + EEG spectrogram
+ EMG envelope (10 min overview), and raw EEG/EMG + rasters (40 s detail)."""
import os
import sys
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
# optional fixed upper limit for the spectrogram log-power colorbar;
# pass as the first command-line argument (e.g. `python make_viewer_style_figure.py 10`
# -> color bar 0-10). Without it the colorbar auto-ranges over the data percentiles.
CBAR_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
TOOLS = ["AccuSleePy", "REST", "somnotate"]
# reference-style state colors: NREM=green, Wake=orange, REM=blue
STATE_COLORS = {1: "#FF7F0E", 2: "#2CA02C", 3: "#1F77B4"}  # Wake, NREM, REM
STATE_NAMES = {1: "Wake", 2: "NREM", 3: "REM"}
# blue spectrogram colormap: dark navy -> blue -> cyan -> yellow (reference style)
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "eeg_blue", ["#04072E", "#0B2FA6", "#1E78E0", "#35C8F0", "#F2EE5C"])


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
    return (np.array([m.get(x, 0) for x in df["brain_state"]]),
            df["confidence_score"].values)


def load_notched(fname):
    x = np.load(os.path.join(CACHE, fname)).astype(np.float64)
    for f0 in (50.0, 100.0):
        b, a = signal.iirnotch(f0, Q=30, fs=FS)
        x = signal.filtfilt(b, a, x)
    return x


def spans(vec, state):
    """contiguous [start_h, dur_h] spans of `state` (4 s epochs)"""
    idx = np.where(vec == state)[0]
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(g[0] * EP / 3600.0, len(g) * EP / 3600.0) for g in groups]


def raster(ax, vec, h0, h1, label):
    for s in (3, 1, 2):  # draw REM under Wake under NREM edges
        ax.broken_barh(spans(vec, s), (0.08, 0.84),
                       facecolors=STATE_COLORS[s], edgecolors="none")
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlim(h0, h1)
    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
    ax.set_xticklabels([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def clock_ticks(h0, h1, every):
    t0 = START + timedelta(hours=h0)
    t = t0.replace(second=0, microsecond=0)
    if t < t0:
        t += timedelta(minutes=1)
    while (t - t0).total_seconds() % every:
        t += timedelta(minutes=1)
    out = []
    while t <= START + timedelta(hours=h1):
        out.append((t - START).total_seconds() / 3600.0)
        t += timedelta(minutes=every)
    return out


def clock_ticks_sec(h0, h1, every_sec):
    t0 = START + timedelta(hours=h0)
    t = t0.replace(microsecond=0)
    while (t - t0).total_seconds() % every_sec:
        t += timedelta(seconds=1)
    out = []
    while t <= START + timedelta(hours=h1):
        out.append((t - START).total_seconds() / 3600.0)
        t += timedelta(seconds=every_sec)
    return out


def fmt(h):
    return (START + timedelta(hours=h)).strftime("%H:%M:%S")


rest = load_rest()
som = load_somnotate()
acc, conf = load_accusleepy()
n = min(len(rest), len(som), len(acc))
vecs = {"AccuSleePy": acc[:n], "REST": rest[:n], "somnotate": som[:n]}
conf = conf[:n]

eeg = load_notched("amp-A-003_1000.npy")[:n * 4000]
emg = load_notched("amp-A-001_1000.npy")[:n * 4000]

# ---- window: 17:15 - 17:25 ; zoom: 17:21:00 - 17:21:40
w0 = (datetime(2026, 9, 24, 17, 15) - START).total_seconds() / 3600
w1 = (datetime(2026, 9, 24, 17, 25) - START).total_seconds() / 3600
z0 = (datetime(2026, 9, 24, 17, 21, 0) - START).total_seconds() / 3600
z1 = (datetime(2026, 9, 24, 17, 21, 40) - START).total_seconds() / 3600

fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(
    12, 1,
    height_ratios=[0.9, 0.32, 0.32, 0.32, 2.6, 1.1, 0.3,
                   1.8, 1.4, 0.32, 0.32, 0.32],
    hspace=0.30)

# ================= block A: 10 min overview =================
axc = fig.add_subplot(gs[0])
h = np.arange(n) * EP / 3600.0
m = (h >= w0) & (h <= w1)
axc.fill_between(h[m], 0, conf[m], where=conf[m] < 0.8, color="#FFB6C1", step="mid")
axc.plot(h[m], conf[m], color="#555555", lw=0.8)
axc.set_xlim(w0, w1); axc.set_ylim(0, 1.02)
axc.set_yticks([0, 0.5, 1]); axc.set_ylabel("Conf.", fontsize=9, rotation=0, ha="right", va="center")
axc.set_xticklabels([])
for sp in ["top", "right"]:
    axc.spines[sp].set_visible(False)

for i, tool in enumerate(TOOLS):
    ax = fig.add_subplot(gs[1 + i])
    raster(ax, vecs[tool], w0, w1, tool)

axS = fig.add_subplot(gs[4])
i0, i1 = int(w0 * 3600 * FS), int(w1 * 3600 * FS)
Ew = eeg[i0:i1].reshape(-1, 1000)          # 1 s windows
f, Pxx = signal.welch(Ew, fs=FS, nperseg=1000, axis=1)
mask = (f >= 0.5) & (f <= 30)
S = np.log10(Pxx[:, mask].T)                # log power
if CBAR_MAX > 0:
    im = axS.imshow(S, aspect="auto", origin="lower", cmap=SPEC_CMAP,
                    extent=[w0, w1, f[mask][0], f[mask][-1]],
                    vmin=0, vmax=CBAR_MAX)
else:
    im = axS.imshow(S, aspect="auto", origin="lower", cmap=SPEC_CMAP,
                    extent=[w0, w1, f[mask][0], f[mask][-1]],
                    vmin=np.percentile(S, 2), vmax=np.percentile(S, 98))
axS.set_ylabel("EEG\nfrequency (Hz)", fontsize=9)
axS.set_yticks([1, 5, 10, 20, 30])
axS.set_xticklabels([])
cax = axS.inset_axes([1.01, 0.0, 0.015, 1.0])   # colorbar outside the axes: parent keeps full width
cb = fig.colorbar(im, cax=cax)
cb.set_label("log power", fontsize=8)

axM = fig.add_subplot(gs[5])
t_env = np.arange(0, len(emg), 200) / FS / 3600.0
rms = np.sqrt(np.convolve(emg ** 2, np.ones(200) / 200, mode="same")[::200])
axM.plot(t_env, rms, color="#696969", lw=0.8)
axM.set_xlim(w0, w1)
axM.set_ylim(0, np.percentile(rms, 99) * 1.5)   # clip artifact spikes so the envelope stays readable
axM.set_ylabel("EMG (µV)\nrms", fontsize=9)
axM.grid(alpha=.25)
tk = clock_ticks(w0, w1, 60)
axM.set_xticks(tk); axM.set_xticklabels([fmt(t) for t in tk], fontsize=8)
axM.set_xlabel("clock time", fontsize=9)

# ================= block B: 40 s detail =================
axE = fig.add_subplot(gs[7])
j0, j1 = int(z0 * 3600 * FS), int(z1 * 3600 * FS)
axE.plot(np.arange(j0, j1, 2) / FS / 3600, eeg[j0:j1:2], color="#696969", lw=0.4)
axE.set_xlim(z0, z1); axE.set_ylim(-1000, 1000)
axE.set_ylabel("EEG (µV)", fontsize=9)
axE.set_xticklabels([])

axM2 = fig.add_subplot(gs[8])
axM2.plot(np.arange(j0, j1, 2) / FS / 3600, emg[j0:j1:2], color="#8C8C8C", lw=0.4)
axM2.set_xlim(z0, z1); axM2.set_ylim(-1500, 1500)
axM2.set_ylabel("EMG (µV)", fontsize=9)
axM2.set_xticklabels([])

for i, tool in enumerate(TOOLS):
    ax = fig.add_subplot(gs[9 + i])
    raster(ax, vecs[tool], z0, z1, tool)
    if i == len(TOOLS) - 1:
        tkz = clock_ticks_sec(z0, z1, 10)
        ax.set_xticks(tkz)
        ax.set_xticklabels([fmt(t) for t in tkz], fontsize=8)
        ax.set_xlabel("clock time", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(axis="x", length=0)

if CBAR_MAX > 0:
    tag = f"  |  color bar 0-{CBAR_MAX:g}"
    outname = f"fig_viewer_style_cbar0-{int(CBAR_MAX)}.png"
else:
    tag = ""
    outname = "fig_viewer_style.png"
fig.suptitle("mouse02 — viewer-style comparison (top: 17:15–17:25; bottom: 17:21:00–17:21:40 detail)  "
             "NREM=green  Wake=orange  REM=blue" + tag, fontsize=11, y=0.985)
fig.savefig(os.path.join(OUT, outname), dpi=135, bbox_inches="tight")
print("saved", outname)
