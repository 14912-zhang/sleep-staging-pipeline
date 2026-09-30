# -*- coding: utf-8 -*-
"""Headless AccuSleePy scoring of mouse02 using the OSF pretrained 4 s model.

Calibration uses pseudo-labels from REST ∩ somnotate epoch agreement
(AccuSleePy requires a calibration file per recording; it is used only for
feature-range normalization, not model training).
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

from accusleepy.fileio import Recording, EMGFilter
from accusleepy.brain_state_set import BrainState, BrainStateSet
from accusleepy.services import create_calibration, score_recording_list
from accusleepy.models import load_model
from accusleepy import constants as c

RUN = r"D:\code\eeg-emg\accusleepy_run"
CACHE = r"D:\code\eeg-emg\edf\cache"
FS = 1000
EPOCH = 4.0

os.makedirs(RUN, exist_ok=True)


def load_notched(fname):
    x = np.load(os.path.join(CACHE, fname)).astype(np.float64)
    for f0 in (50.0, 100.0):
        b, a = signal.iirnotch(f0, Q=30, fs=FS)
        x = signal.filtfilt(b, a, x)
    return x.astype(np.float32)


def main():
    # ---- 1. recording file (parquet, columns eeg/emg) ----
    eeg = load_notched("amp-A-003_1000.npy")  # user-confirmed: A-003 = EEG
    emg = load_notched("amp-A-001_1000.npy")  # A-001 = EMG
    n = min(len(eeg), len(emg))
    n_epochs = int(n / (FS * EPOCH))          # 2539 epochs, matches REST output
    n = n_epochs * int(FS * EPOCH)            # trim to whole epochs
    eeg, emg = eeg[:n], emg[:n]
    rec_path = os.path.join(RUN, "mouse02.parquet")
    pd.DataFrame({"eeg": eeg, "emg": emg}).to_parquet(rec_path)
    print(f"recording: {n} samples = {n/FS:.1f} s")

    # ---- 2. pseudo-label file from REST ∩ somnotate agreement ----
    rest = loadmat(r"D:\code\eeg-emg\edf\rest\rest_mouse02_REST_V2.0.mat")["score"].flatten().astype(int)
    # REST: 1=Wake 2=NREM 3=REM  -> AccuSleePy digits: 1=REM 2=Wake 3=NREM
    rest_map = {1: 2, 2: 3, 3: 1, 4: -1}
    som_digits = np.zeros(len(rest), dtype=int)
    prev, labels = 0.0, []
    for line in open(r"D:\code\eeg-emg\somnotate_run\data\processed\mouse02_automated.hyp"):
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        try:
            end = float(parts[-1])
        except ValueError:
            continue
        som_digits[int(prev / EPOCH):int(end / EPOCH)] = {
            "awake": 2, "non-REM": 3, "REM": 1}[parts[0]]
        prev = end
    agree = (som_digits > 0) & (rest <= 3) & (np.array([rest_map[x] for x in rest]) == som_digits)
    pseudo = np.where(agree, som_digits, -1)
    pseudo = pseudo[:n_epochs]

    # ---- 3. model & brain state set ----
    model, model_epoch, epochs_per_img, model_type, brain_state_dicts = load_model(
        os.path.join(RUN, "ssann_4s.pth"))
    assert model_epoch == EPOCH, f"model epoch {model_epoch}"
    print("model brain states:", brain_state_dicts)
    # scored states only; the non-scored 'none' entry is the undefined label
    undefined_label = c.UNDEFINED_LABEL  # label files always use -1 for undefined
    brain_states = [BrainState(name=d["name"], digit=d["digit"],
                               is_scored=d["is_scored"], frequency=d["frequency"])
                    for d in brain_state_dicts]
    bss = BrainStateSet(brain_states, undefined_label)
    label_path = os.path.join(RUN, "labels_pseudo.csv")
    pd.DataFrame({"brain_state": pseudo}).to_csv(label_path, index=False)
    for d in (1, 2, 3):
        print(f"pseudo-labels digit {d}: {(pseudo == d).sum()} epochs")

    emg_filter = EMGFilter(order=c.DEFAULT_EMG_FILTER_ORDER,
                           bp_lower=c.DEFAULT_EMG_BP_LOWER,
                           bp_upper=c.DEFAULT_EMG_BP_UPPER)

    recording = Recording(name=1, recording_file=rec_path,
                          label_file=label_path, calibration_file="",
                          sampling_rate=FS)

    # ---- 4. calibration from pseudo-labels ----
    cal_path = os.path.join(RUN, "mouse02_calibration.csv")
    r = create_calibration(recording, EPOCH, bss, emg_filter, cal_path)
    if not r.success:
        raise RuntimeError(r.error)
    print("calibration:", r.messages, r.warnings)
    recording.calibration_file = cal_path

    # ---- 5. score ----
    from accusleepy.services import LoadedModel
    loaded = LoadedModel(model=model, epoch_length=model_epoch,
                         epochs_per_img=epochs_per_img)
    label_out = os.path.join(RUN, "mouse02_accusleepy_labels.csv")
    recording.label_file = label_out
    r = score_recording_list([recording], loaded, EPOCH,
                             only_overwrite_undefined=False,
                             save_confidence_scores=True,
                             min_bout_length=EPOCH,
                             brain_state_set=bss,
                             emg_filter=emg_filter)
    print("scoring:", r.success, r.messages, r.warnings, r.error)
    if r.success:
        df = pd.read_csv(label_out)
        print(df["brain_state"].value_counts())


if __name__ == "__main__":
    main()
