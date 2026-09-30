# -*- coding: utf-8 -*-
"""Visualization suite for mouse02 sleep staging — REST / somnotate / AccuSleePy.

Elements: raw waveforms (full-recording envelope + 10 min zoom),
EEG spectrogram heatmaps, sleep-state rasters (hypnograms) per tool.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from scipy import signal
from datetime import datetime, timedelta

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

OUT = r"D:\code\eeg-emg\results"
CACHE = r"D:\code\eeg-emg\edf\cache"
START = datetime(2026, 9, 24, 16, 16, 6)
EP = 4.0
FS = 1000
STATE_COLORS = {1: "#DC267F", 2: "#648FFF", 3: "#FFB000"}
STATE_NAMES = {1: "Wake", 2: "NREM", 3: "REM"}
HYPMAP = ListedColormap(["#DDDDDD", "#DC267F", "#648FFF", "#FFB000"])
# blue spectrogram colormap: dark navy -> blue -> cyan -> yellow (reference style)
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "eeg_blue", ["#04072E", "#0B2FA6", "#1E78E0", "#35C8F0", "#F2EE5C"])
WAVE_GRAY = {"eeg": "#696969", "emg": "#8C8C8C"}
TOOLS = ["REST", "somnotate", "AccuSleePy"]

os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- loaders
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
    m = {1: 3, 2: 1, 3: 2}  # 1=REM 2=Wake 3=NREM -> 1=Wake 2=NREM 3=REM
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

eeg = load_notched("amp-A-003_1000.npy")[:n * int(EP * FS)]  # A-003 = EEG (user-confirmed)
emg = load_notched("amp-A-001_1000.npy")[:n * int(EP * FS)]  # A-001 = EMG (user-confirmed)


def clock_ticks(every_min):
    t = START.replace(minute=0, second=0, microsecond=0)
    while (t - START).total_seconds() <= 0:
        t += timedelta(minutes=every_min)
    end = START + timedelta(hours=dur_h)
    ticks = []
    while t <= end:
        ticks.append((t - START).total_seconds() / 3600.0)
        t += timedelta(minutes=every_min)
    return ticks


def set_clock_axis(ax, every_min=30):
    tk = clock_ticks(every_min)
    ax.set_xticks(tk)
    ax.set_xticklabels([(START + timedelta(hours=h)).strftime("%H:%M") for h in tk])


def hyp_strip(ax, vec, ylab=None):
    img = np.vstack([vec, vec])
    ax.imshow(img, aspect="auto", interpolation="nearest", cmap=HYPMAP,
              vmin=0, vmax=3, extent=[0, dur_h, 0, 1])
    ax.set_yticks([])
    if ylab:
        ax.set_ylabel(ylab, fontsize=10, rotation=0, ha="right", va="center")


def spectrogram(ax, x, t0_h=0.0, t1_h=None):
    t1_h = t1_h if t1_h is not None else dur_h
    E = x.reshape(-1, int(EP * FS))
    f, Pxx = signal.welch(E, fs=FS, nperseg=500, axis=1)
    mask = (f >= 0.5) & (f <= 30)
    S = np.log10(Pxx[:, mask].T)
    vmin, vmax = np.percentile(S, [2, 98])
    im = ax.imshow(S, aspect="auto", origin="lower", cmap=SPEC_CMAP,
                   extent=[t0_h, t1_h, f[mask][0], f[mask][-1]], vmin=vmin, vmax=vmax)
    ax.set_ylabel("EEG frequency (Hz)")
    cax = ax.inset_axes([1.01, 0.0, 0.015, 1.0])   # colorbar outside: parent keeps full width
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log power")
    ax.set_yticks([1, 4, 8, 12, 20, 30])
    return ax


def envelope(x, win_sec=1):
    w = int(win_sec * FS)
    m = len(x) // w * w
    X = x[:m].reshape(-1, w)
    return np.arange(m // w) * win_sec / 3600.0, X.min(1), X.max(1)


# ================================================================ master figure
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(6, 1, height_ratios=[1.4, 1.4, 3.2, 0.55, 0.55, 0.55],
                      hspace=0.14)

axE = fig.add_subplot(gs[0])
t, lo, hi = envelope(eeg)
axE.fill_between(t, lo, hi, color=WAVE_GRAY["eeg"], lw=0)
axE.set_xlim(0, dur_h)   # same time span as spectrogram/rasters (no autoscale margins)
axE.set_ylim(-1500, 1500)
axE.set_ylabel("EEG (µV)\nA-003", fontsize=9)
axE.grid(alpha=.25); axE.set_xticklabels([])

axM = fig.add_subplot(gs[1])
t, lo, hi = envelope(emg)
axM.fill_between(t, lo, hi, color=WAVE_GRAY["emg"], lw=0)
axM.set_xlim(0, dur_h)
axM.set_ylim(-2500, 2500)
axM.set_ylabel("EMG (µV)\nA-001", fontsize=9)
axM.grid(alpha=.25); axM.set_xticklabels([])

axS = fig.add_subplot(gs[2])
spectrogram(axS, eeg)

for i, tool in enumerate(TOOLS):
    ax = fig.add_subplot(gs[3 + i])
    hyp_strip(ax, vecs[tool], ylab=tool)
    ax.set_xlim(0, dur_h)
    if i == len(TOOLS) - 1:
        set_clock_axis(ax)
        ax.set_xlabel("clock time")
    else:
        ax.set_xticklabels([])
    tot = (vecs[tool] > 0).sum()
    txt = "  ".join(f"{STATE_NAMES[s]} {100*(vecs[tool]==s).sum()/tot:.0f}%"
                    for s in (1, 2, 3) if (vecs[tool] == s).any())
    ax.text(1.002, 0.5, txt, transform=ax.transAxes, fontsize=8, va="center")

fig.suptitle("mouse02 — EEG/EMG waveforms, EEG spectrogram and sleep-state rasters "
             "(2026-09-24 16:16–19:05, EEG=A-003, EMG=A-001)", y=0.955)
fig.savefig(os.path.join(OUT, "fig_sleep_overview.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("fig_sleep_overview.png")

# ================================================================ zoom figure
t_zoom0 = datetime(2026, 9, 24, 17, 15)   # clock time: contains longest REM bout (17:21, 96 s)
t_zoom1 = datetime(2026, 9, 24, 17, 25)
z0 = (t_zoom0 - START).total_seconds() / 3600.0   # elapsed hours since recording start
z1 = (t_zoom1 - START).total_seconds() / 3600.0
i0, i1 = int(z0 * 3600 * FS), int(z1 * 3600 * FS)
tk = [z0 + k / 5 * (z1 - z0) for k in range(6)]
tk_lab = [(START + timedelta(hours=h)).strftime("%H:%M") for h in tk]

fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(6, 1, height_ratios=[1.6, 1.2, 2.8, 0.5, 0.5, 0.5],
                      hspace=0.14)

axE = fig.add_subplot(gs[0])
axE.plot(np.arange(i0, i1, 3) / FS / 3600, eeg[i0:i1:3], lw=0.25, color=WAVE_GRAY["eeg"])
axE.set_xlim(z0, z1); axE.set_ylim(-1200, 1200)
axE.set_ylabel("EEG (µV)", fontsize=9)
axE.grid(alpha=.25); axE.set_xticklabels([])

axM = fig.add_subplot(gs[1])
axM.plot(np.arange(i0, i1, 3) / FS / 3600, emg[i0:i1:3], lw=0.25, color=WAVE_GRAY["emg"])
axM.set_xlim(z0, z1); axM.set_ylim(-1500, 1500)
axM.set_ylabel("EMG (µV)", fontsize=9)
axM.grid(alpha=.25); axM.set_xticklabels([])

axS = fig.add_subplot(gs[2])
spectrogram(axS, eeg[i0:i1], t0_h=z0, t1_h=z1)
axS.set_xticks(tk); axS.set_xticklabels([])

for i, tool in enumerate(TOOLS):
    ax = fig.add_subplot(gs[3 + i])
    hyp_strip(ax, vecs[tool], ylab=tool)
    ax.set_xlim(z0, z1)
    ax.set_xticks(tk)
    if i == len(TOOLS) - 1:
        ax.set_xticklabels(tk_lab)
        ax.set_xlabel("clock time")
    else:
        ax.set_xticklabels([])

fig.suptitle("mouse02 — 10 min detail (17:15–17:25): raw waveforms, EEG spectrogram, state rasters",
             y=0.96)
fig.savefig(os.path.join(OUT, "fig_sleep_zoom.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("fig_sleep_zoom.png")

# ================================================================ agreement
def agreement(a, b):
    ok = (a > 0) & (b > 0)
    return 100 * (a[ok] == b[ok]).mean()

overall = np.zeros((len(TOOLS), len(TOOLS)))
for i, a in enumerate(TOOLS):
    for j, b in enumerate(TOOLS):
        if i == j:
            overall[i, j] = 100.0
        elif i < j:
            overall[i, j] = overall[j, i] = agreement(vecs[a], vecs[b])

pair_list = [("REST", "somnotate"), ("REST", "AccuSleePy"), ("somnotate", "AccuSleePy")]
per_state = np.zeros((len(pair_list), 3))
for r, (a, b) in enumerate(pair_list):
    for ci, s in enumerate((1, 2, 3)):
        either = (vecs[a] == s) | (vecs[b] == s)
        both = (vecs[a] == s) & (vecs[b] == s)
        per_state[r, ci] = 100 * both.sum() / max(either.sum(), 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1, 1.1]})
im = ax1.imshow(overall, cmap="Blues", vmin=0, vmax=100)
for i in range(len(TOOLS)):
    for j in range(len(TOOLS)):
        ax1.text(j, i, f"{overall[i, j]:.0f}", ha="center", va="center",
                 color="w" if overall[i, j] > 60 else "k", fontsize=11)
ax1.set_xticks(range(len(TOOLS))); ax1.set_yticks(range(len(TOOLS)))
ax1.set_xticklabels(TOOLS, rotation=20, ha="right"); ax1.set_yticklabels(TOOLS)
ax1.set_title("overall epoch agreement (%)")
plt.colorbar(im, ax=ax1, fraction=.046)

im2 = ax2.imshow(per_state, cmap="Blues", vmin=0, vmax=100)
for r in range(len(pair_list)):
    for ci in range(3):
        ax2.text(ci, r, f"{per_state[r, ci]:.0f}", ha="center", va="center",
                 color="w" if per_state[r, ci] > 60 else "k", fontsize=10)
ax2.set_xticks(range(3)); ax2.set_xticklabels([STATE_NAMES[s] for s in (1, 2, 3)])
ax2.set_yticks(range(len(pair_list)))
ax2.set_yticklabels([f"{a} vs {b}" for a, b in pair_list], fontsize=9)
ax2.set_title("per-state overlap (Jaccard, %)")
plt.colorbar(im2, ax=ax2, fraction=.046)
fig.suptitle("Cross-tool agreement — REST / somnotate / AccuSleePy")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_agreement.png"), dpi=130)
plt.close(fig)
print("fig_agreement.png")

# ================================================================ transitions
fig, axt = plt.subplots(figsize=(6.5, 5.5))
ok = vecs["REST"] > 0
a, b = vecs["REST"][:-1][ok[:-1]], vecs["REST"][1:][ok[:-1]]
keep = (a > 0) & (b > 0)
T = np.zeros((3, 3))
for x, y in zip(a[keep].astype(int), b[keep].astype(int)):
    T[x - 1, y - 1] += 1
Tn = T / T.sum(axis=1, keepdims=True)
im = axt.imshow(Tn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        axt.text(j, i, f"{Tn[i, j]*100:.0f}%\n({int(T[i, j])})", ha="center", va="center",
                 color="w" if Tn[i, j] > .5 else "k", fontsize=11)
axt.set_xticks(range(3)); axt.set_yticks(range(3))
axt.set_xticklabels([STATE_NAMES[s] for s in (1, 2, 3)])
axt.set_yticklabels([STATE_NAMES[s] for s in (1, 2, 3)])
axt.set_xlabel("next state"); axt.set_ylabel("current state")
axt.set_title("state transition matrix (REST)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_transitions.png"), dpi=130)
plt.close(fig)
print("fig_transitions.png")

# ================================================================ REM timing
fig, ax = plt.subplots(figsize=(15, 3.4))
for yi, tool in enumerate(TOOLS):
    idx = np.where(vecs[tool] == 3)[0]
    ax.scatter(hours[idx], np.full(len(idx), yi), marker="|", s=260,
               color=STATE_COLORS[3], lw=2.5)
    idxw = np.where(vecs[tool] == 1)[0]
    ax.scatter(hours[idxw], np.full(len(idxw), yi - .22), marker="|", s=8,
               color=STATE_COLORS[1], lw=.8, alpha=.45)
ax.set_yticks(range(len(TOOLS))); ax.set_yticklabels(TOOLS)
ax.set_ylim(-0.7, len(TOOLS) - 0.3)
ax.set_xlabel("clock time"); ax.set_title("REM episodes (thick) and wake epochs (thin) per tool")
set_clock_axis(ax, every_min=30)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_rem_timing.png"), dpi=130)
plt.close(fig)
print("fig_rem_timing.png")
