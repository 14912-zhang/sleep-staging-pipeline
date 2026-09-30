# -*- coding: utf-8 -*-
"""Aggregate staging results from REST / FASTER2 / somnotate on a common 4 s grid,
plot comparison hypnograms + agreement, and export summary statistics."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime, timedelta

OUT = r"D:\code\eeg-emg\results"
START = datetime(2026, 9, 24, 16, 16, 6)
EP = 4.0  # seconds

STATE_COLORS = {0: "#BBBBBB", 1: "#DC267F", 2: "#648FFF", 3: "#FFB000"}  # unk/wake/nrem/rem
STATE_NAMES = {0: "unk", 1: "Wake", 2: "NREM", 3: "REM"}


def load_rest():
    d = loadmat(r"D:\code\eeg-emg\edf\rest\rest_mouse02_REST_V2.0.mat")
    s = d["score"].flatten().astype(int)  # 1=Wake 2=NREM 3=REM 4=Artifact
    m = {1: 1, 2: 2, 3: 3, 4: 0}
    return np.array([m[x] for x in s])


def load_somnotate():
    ends = []
    labels = []
    prev = 0.0
    for line in open(r"D:\code\eeg-emg\somnotate_run\data\processed\mouse02_automated.hyp"):
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        try:
            end = float(parts[-1])
        except ValueError:
            continue
        ends.append(end)
        labels.append(parts[0])
        prev = end
    m = {"awake": 1, "non-REM": 2, "REM": 3}
    n = int(prev / EP)
    out = np.zeros(n, dtype=int)
    prev = 0.0
    for lab, end in zip(labels, ends):
        i0, i1 = int(prev / EP), min(int(end / EP), n)
        out[i0:i1] = m.get(lab, 0)
        prev = end
    return out




def load_accusleepy():
    df = pd.read_csv("D:/code/eeg-emg/accusleepy_run/mouse02_accusleepy_labels.csv")
    m = {1: 3, 2: 1, 3: 2}  # accusleepy: 1=REM 2=Wake 3=NREM -> internal 1=Wake 2=NREM 3=REM
    return np.array([m.get(x, 0) for x in df["brain_state"]])




rest = load_rest()
som = load_somnotate()
acc = load_accusleepy()
n = min(len(rest), len(som), len(acc))
rest, som, acc = rest[:n], som[:n], acc[:n]
hours = np.arange(n) * EP / 3600.0

# ---- agreement (epochs where both tools have a definite state) ----
def agreement(a, b):
    ok = (a > 0) & (b > 0)
    return 100 * (a[ok] == b[ok]).mean(), ok.sum()

agr_rs, n_rs = agreement(rest, som)
agr_ra, n_ra = agreement(rest, acc)
agr_sa, n_sa = agreement(som, acc)

# ---- summary stats ----
def stats(vec, name):
    rows = []
    total = (vec > 0).sum()
    for st in (1, 2, 3):
        frac = 100 * (vec == st).sum() / max(total, 1)
        # bouts
        d = np.diff(np.r_[0, (vec == st).astype(int), 0])
        starts = np.where(d == 1)[0]
        stops = np.where(d == -1)[0]
        bouts = stops - starts
        rows.append({
            "tool": name, "state": STATE_NAMES[st],
            "percent_of_scored": round(frac, 1),
            "total_min": round((vec == st).sum() * EP / 60, 1),
            "n_bouts": len(bouts),
            "median_bout_s": int(np.median(bouts) * EP) if len(bouts) else 0,
        })
    return rows

allrows = stats(rest, "REST") + stats(som, "somnotate") + stats(acc, "AccuSleePy")
statsdf = pd.DataFrame(allrows)
statsdf.to_csv(f"{OUT}\\sleep_stats_summary.csv", index=False)
print(statsdf.to_string(index=False))
print(f"\nEpoch agreement: REST~somnotate {agr_rs:.1f}% | somnotate~AccuSleePy {agr_sa:.1f}% | "
      f"REST~AccuSleePy {agr_ra:.1f}%")

# ---- figure ----
fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(4, 2, height_ratios=[1.6, 1.6, 1.6, 1.3], hspace=0.45, wspace=0.12)

axes = []
tool_meta = {
    "REST": "REST (pretrained Rodent EEG Sleep Transformer)",
    "somnotate": "somnotate (LDA+HMM, trained on pilot consensus)",
    "AccuSleePy": "AccuSleePy (SSANN, OSF pretrained)",
}
for i, (vec, name) in enumerate([(rest, "REST"),
                                 (som, "somnotate"),
                                 (acc, "AccuSleePy")]):
    ax = fig.add_subplot(gs[i, :])
    for st in (1, 2, 3):
        mask = vec == st
        ax.fill_between(hours, 0, 1, where=mask, color=STATE_COLORS[st], step="mid",
                        label=STATE_NAMES[st])
    ax.set_xlim(0, hours[-1]); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_ylabel(tool_meta[name], fontsize=9, rotation=0,
                                     ha="right", va="center")
    if i == 0:
        ax.legend(loc="upper right", ncol=3, frameon=False)
        ax.set_title(f"mouse02 — 2026-09-24 {START:%H:%M} start, 2 h 49 min, EEG=A-002, EMG=A-003 (corrected mapping)")
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("time since recording start (h)")
    axes.append(ax)

# agreement matrix
axc = fig.add_subplot(gs[3, 0])
pairs = [("REST vs somnotate", agr_rs), ("somnotate vs AccuSleePy", agr_sa), ("REST vs AccuSleePy", agr_ra)]
axc.bar(range(len(pairs)), [p[1] for p in pairs], color="#648FFF")
axc.set_xticks(range(len(pairs))); axc.set_xticklabels([p[0] for p in pairs], fontsize=7)
axc.set_ylabel("epoch agreement (%)"); axc.set_ylim(0, 100)
for i, p in enumerate(pairs):
    axc.text(i, p[1] + 2, f"{p[1]:.0f}%", ha="center", fontsize=8)

# state distribution
axd = fig.add_subplot(gs[3, 1])
tools = ["REST", "somnotate", "AccuSleePy"]
bottoms = np.zeros(len(tools))
for st, stn in [(1, "Wake"), (2, "NREM"), (3, "REM")]:
    vals = []
    for tname in tools:
        row = statsdf[(statsdf.tool == tname) & (statsdf.state == stn)]
        vals.append(row["percent_of_scored"].iloc[0] if len(row) else 0)
    vals = np.array(vals)
    axd.bar(tools, vals, bottom=bottoms, color=STATE_COLORS[st], label=stn)
    for i, v in enumerate(vals):
        if v > 3:
            axd.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", fontsize=9, color="w")
    bottoms += vals
axd.set_ylabel("% of scored epochs"); axd.set_ylim(0, 108); axd.legend(frameon=False, fontsize=8)

fig.suptitle("Automated mouse sleep staging — tool comparison", fontsize=13)
fig.savefig(f"{OUT}\\hypnogram_comparison.png", dpi=130, bbox_inches="tight")
print("saved", f"{OUT}\\hypnogram_comparison.png")
