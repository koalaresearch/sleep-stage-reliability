#!/usr/bin/env python3
"""Read-only two-session EEG feature pilot for Sleep-EDF SC v1.2.

Loads one verified EDF from the existing ZIP into a small ephemeral file, reads
Fpz-Cz in bounded batches without preloading, aligns 30-second epoch centers
to independently loaded hypnogram annotations, selects the predeclared main
sleep episode window, and extracts the ten specified Welch spectral features.

No training, no whole-dataset extraction, no edits to the protocol or input ZIP.
Outputs a small pilot CSV and provenance JSON. Only for validation of the
pipeline; not the main analysis and not a basis for changing parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mne
import numpy as np
import scipy
from scipy.signal import welch

try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML is required to read the frozen YAML. Check: python -c "import yaml"')

CLASSES = ("W", "N1", "N2", "N3", "REM")
FEATURE_NAMES = [f"log10_{b}_uv2" for b in ("delta", "theta", "alpha", "sigma", "beta")] + [f"rel_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_archive_member_verified(z: ZipFile, member: str, path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with z.open(member) as src, path.open("wb") as out:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
            out.write(block)
    if h.hexdigest() != expected:
        raise RuntimeError(f"Transient copy has wrong SHA-256: {member}")


def epoch_labels(annotations: mne.Annotations, n_epochs: int, stage_map: dict, excluded: list[str]) -> list[str]:
    if annotations.orig_time is not None:
        raise ValueError("Hypnogram orig_time is set: manual time-base alignment required")
    labels = np.empty(n_epochs, dtype=object)
    labels[:] = None
    coverage = np.zeros(n_epochs, dtype=np.uint8)
    center = (np.arange(n_epochs) + 0.5) * 30.0
    approved = set(stage_map) | set(excluded)
    for onset, duration, label in zip(annotations.onset, annotations.duration, annotations.description):
        label = str(label)
        if label not in approved:
            raise ValueError(f"Unknown annotation label: {label!r}")
        if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid annotation range for {label}: {onset}, {duration}")
        selected = (center >= float(onset)) & (center < float(onset + duration))
        if np.any(coverage[selected] != 0):
            raise ValueError("Overlapping annotated epoch")
        coverage[selected] = 1
        labels[selected] = label
    if np.any(coverage != 1):
        raise ValueError(f"Epoch annotation coverage has {int(np.sum(coverage != 1))} gaps")
    return [stage_map.get(str(label)) if label not in excluded else None for label in labels]


def main_sleep_window(stages: list[str | None], gap_min: float, context_min: float) -> tuple[int, int, dict]:
    """Merge sleep runs separated by strictly less than gap_min, choose most sleep epochs.

    Ties are resolved by earliest occurrence; boundaries denote [start, end).
    """
    sleep = np.array([(s in CLASSES and s != "W") for s in stages], dtype=bool)
    if not np.any(sleep):
        raise ValueError("No sleep epochs in record")
    idx = np.flatnonzero(sleep)
    run_starts = np.r_[idx[0], idx[1:][np.diff(idx) > 1]]
    run_ends = np.r_[idx[:-1][np.diff(idx) > 1] + 1, idx[-1] + 1]
    gap_max_epochs = gap_min * 60.0 / 30.0
    episodes = []
    for start, end in zip(run_starts, run_ends):
        start, end = int(start), int(end)
        if episodes and start - episodes[-1][1] < gap_max_epochs:
            episodes[-1][1] = end
            episodes[-1][2] += end - start
        else:
            episodes.append([start, end, end - start])
    best = max(episodes, key=lambda ep: ep[2])  # stable earliest tie-break
    pad = int(round(context_min * 60 / 30))
    lo = max(0, best[0] - pad)
    hi = min(len(stages), best[1] + pad)
    return lo, hi, {
        "episode_start_epoch": best[0], "episode_end_epoch_exclusive": best[1],
        "episode_sleep_epochs": best[2], "episode_count": len(episodes),
        "wake_context_epochs_each_side": pad,
    }


def welch_features(batch_microvolts: np.ndarray, cfg: dict) -> np.ndarray:
    p = cfg["preprocessing"]
    wp = p["welch"]
    fs = float(wp["fs"])
    nperseg = int(wp["nperseg"])
    nfft = int(wp["nfft"])
    if fs != 100.0 or nperseg != 400 or int(wp["noverlap"]) != 200 or nfft != 400:
        raise ValueError("This pilot is v1.2-specific; expected Welch 100 Hz / 400 / 200 / 400")
    if wp["scaling"] != "density" or wp["detrend"] != "constant" or wp["window"] != "hann" or wp["average"] != "mean":
        raise ValueError("Unrecognized Welch policy")
    freqs, psd = welch(batch_microvolts, fs=fs, window="hann", nperseg=nperseg,
                       noverlap=int(wp["noverlap"]), nfft=nfft,
                       detrend="constant", scaling="density", average="mean", axis=-1)
    df = float(freqs[1] - freqs[0])
    if not math.isclose(df, 0.25, abs_tol=1e-12):
        raise ValueError(f"Unexpected Welch bin spacing: {df}")
    if p["band_power_integration"] != "rectangle_sum" or p["total_power"] != "sum_of_five_bands":
        raise ValueError("Unexpected band integration/normalization policy")
    band_items = list(p["bands_hz"].items())
    if [name for name, _ in band_items] != ["delta", "theta", "alpha", "sigma", "beta"]:
        raise ValueError("Unexpected bands or band order")
    masks = []
    for j, (_name, (lo, hi)) in enumerate(band_items):
        mask = (freqs >= float(lo)) & ((freqs <= float(hi)) if j == len(band_items)-1 else (freqs < float(hi)))
        if not np.any(mask):
            raise ValueError("One spectral band has no FFT bins")
        masks.append(mask)
    counts = np.stack(masks, axis=0).sum(axis=0)
    if np.any(counts > 1):
        raise ValueError("Bands overlap")
    powers = np.stack([psd[:, m].sum(axis=1) * df for m in masks], axis=1)
    if not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("Invalid PSD-derived band powers")
    total = powers.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Nonzero EEG variance but no 0.5-30 Hz power: manual review required")
    rel = powers / total[:, None]
    if not np.allclose(rel.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Relative powers do not sum to 1")
    epsilon = float(p["absolute_log10_epsilon_uV2"])
    return np.hstack((np.log10(powers + epsilon), rel))


def process_record(z: ZipFile, row: dict, cfg: dict, tmp_base: Path, batch_epochs: int = 64) -> tuple[list[dict], dict]:
    p = cfg["preprocessing"]
    with TemporaryDirectory(prefix="sleepedf-features-", dir=str(tmp_base)) as temp:
        psg = Path(temp) / "record-PSG.edf"
        hyp = Path(temp) / "record-Hypnogram.edf"
        copy_archive_member_verified(z, row["psg_member"], psg, row["sha256_actual_psg"])
        copy_archive_member_verified(z, row["hypnogram_member"], hyp, row["sha256_actual_hyp"])
        raw = mne.io.read_raw_edf(str(psg), include=[str(p["main_channel"])], preload=False, verbose="ERROR")
        if raw.ch_names != [str(p["main_channel"])]:
            raise ValueError(f"Expected only one Fpz-Cz channel, got {raw.ch_names}")
        sfreq = float(raw.info["sfreq"])
        if not math.isclose(sfreq, float(p["sample_rate_hz"]), abs_tol=0.01):
            raise ValueError(f"Unexpected sample rate {sfreq}")
        samples_per_epoch = int(round(sfreq * float(p["epoch_length_seconds"])))
        if samples_per_epoch != 3000 or raw.n_times % samples_per_epoch != 0:
            raise ValueError(f"Non-integer number of 30-second epochs: {raw.n_times=}")
        n_epochs = raw.n_times // samples_per_epoch
        ann = mne.read_annotations(str(hyp))  # deliberately do not set on raw; preserves original overhang
        stages = epoch_labels(ann, n_epochs, p["stage_map"], p["exclude_labels"])
        low, high, episode_info = main_sleep_window(
            stages,
            gap_min=float(p["analysis_window"]["episode_merge_max_wake_gap_minutes"]),
            context_min=float(p["analysis_window"]["wake_context_minutes_each_side"]),
        )
        all_sleep_idx = [i for i,s in enumerate(stages) if s in ("N1", "N2", "N3", "REM")]
        pad = int(round(float(p["analysis_window"]["wake_context_minutes_each_side"]) * 2))
        sensitivity_start = max(0, all_sleep_idx[0]-pad)
        sensitivity_end = min(n_epochs, all_sleep_idx[-1]+1+pad)
        feature_rows = []
        rejected_nonfinite = rejected_zero = 0
        for start_idx in range(low, high, batch_epochs):
            stop_idx = min(high, start_idx + batch_epochs)
            data_v = raw.get_data(picks=[0], start=start_idx * samples_per_epoch,
                                  stop=stop_idx * samples_per_epoch)[0]
            data_uv = np.asarray(data_v, dtype=float).reshape(stop_idx-start_idx, samples_per_epoch) * 1e6
            labels_batch = stages[start_idx:stop_idx]
            eligible = np.array([s in CLASSES for s in labels_batch], dtype=bool)
            finite = np.isfinite(data_uv).all(axis=1)
            # Non-finite check first, then detect zero variance only among finite epochs
            variances = np.full(len(labels_batch), np.nan, dtype=float)
            variances[finite] = np.var(data_uv[finite], axis=1)
            nonfinite_reject = eligible & ~finite
            zero_reject = eligible & finite & (variances == 0)
            rejected_nonfinite += int(nonfinite_reject.sum())
            rejected_zero += int(zero_reject.sum())
            valid = eligible & finite & (variances > 0)
            if not np.any(valid):
                continue
            feats = welch_features(data_uv[valid], cfg)
            valid_indexes = np.flatnonzero(valid)
            for row_i, vec in zip(valid_indexes, feats):
                idx = start_idx + int(row_i)
                entry = {"subject_id": row["subject_id"],
                         "record_id": row["record_id"],
                         "night_index": int(row["night_index"]),
                         "epoch_index": idx, "epoch_start_sec": idx * 30,
                         "stage": stages[idx]}
                entry.update({name: float(value) for name, value in zip(FEATURE_NAMES, vec)})
                feature_rows.append(entry)
        raw.close()
    excluded_by_label = sum(s is None for s in stages[low:high])
    summary = {
        "excluded_by_label": excluded_by_label,
        "record_id": row["record_id"], "night_index": int(row["night_index"]),
        "all_recording_epochs": int(n_epochs),
        "main_sleep_window_start_epoch": low, "main_sleep_window_end_epoch_exclusive": high,
        "main_sleep_window_length_epochs": high-low,
        "sensitivity_window_start_epoch": sensitivity_start,
        "sensitivity_window_end_epoch_exclusive": sensitivity_end,
        "sensitivity_window_length_epochs": sensitivity_end-sensitivity_start,
        "window_reference_stage_counts": dict(Counter(str(s) for s in stages[low:high])),
        "feature_rows": len(feature_rows),
        "rejected_nonfinite": rejected_nonfinite,
        "rejected_zero_variance": rejected_zero,
        **episode_info,
    }
    expected = sum(s in CLASSES for s in stages[low:high])
    if expected + excluded_by_label != high - low:
        raise AssertionError("Inconsistent scorable vs. excluded epoch count")
    if len(feature_rows)+rejected_nonfinite+rejected_zero != expected:
        raise AssertionError("Inconsistent extracted+rejected epoch counts")
    return feature_rows, summary


def json_numpy_default(obj):
    """Serialize NumPy scalar metadata without altering analysis values."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot JSON serialize {type(obj).__name__}")

def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--zip", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--subject", default="00")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--tmp", type=Path, default=None,
                     help="Existing directory with space for one temporary PSG; default OS temporary folder")
    args = cli.parse_args()
    subject = str(args.subject).zfill(2)
    if not args.zip.exists() or not args.manifest.exists() or not args.config.exists():
        raise FileNotFoundError("ZIP, verified manifest, or configuration is missing")
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["study"]["protocol"] != "v1.2":
        raise ValueError("Expected protocol v1.2")
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["subject_id"] == subject]
    rows.sort(key=lambda r: int(r["night_index"]))
    if len(rows) != 2 or [int(row["night_index"]) for row in rows] != [1, 2]:
        raise ValueError(f"Expected two sessions, night indices 1 and 2, for subject {subject}")
    if any(r["sha256_match"] != "matched" for r in rows):
        raise ValueError("Manifest is not fully hash verified")
    # Ensure extraction of a single member cannot exhaust remaining local storage.
    temp_base = args.tmp or Path(__import__('tempfile').gettempdir())
    if not temp_base.is_dir():
        raise FileNotFoundError(f"Temporary directory does not exist: {temp_base}")
    with ZipFile(args.zip) as z:
        max_transient = max(z.getinfo(r["psg_member"]).file_size + z.getinfo(r["hypnogram_member"]).file_size for r in rows)
        available = shutil.disk_usage(temp_base).free
        reserve = 512 * 1024**2
        if available < max_transient + reserve:
            raise RuntimeError(f"Not enough scratch space. Required one PSG + 512 MiB reserve; available={available / 1024**3:.2f} GiB")
        output_rows, reports = [], []
        for row in rows:
            current_rows, report = process_record(z, row, cfg, temp_base)
            output_rows.extend(current_rows)
            reports.append(report)
            print(f"{report['record_id']}: {report['main_sleep_window_length_epochs']} epochs in analysis window; "
                  f"{report['feature_rows']} feature rows; excluded_by_label="
                  f"{report['excluded_by_label']}; rejected_signal="
                  f"{report['rejected_nonfinite'] + report['rejected_zero_variance']}")
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"pilot_features_subject{subject}.csv"
    header = ["subject_id", "record_id", "night_index", "epoch_index", "epoch_start_sec", "stage"] + FEATURE_NAMES
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "purpose": "Two-record feature extraction pilot; no model predictions or metrics",
        "status": "PASS", "subject_id": subject, "feature_count": len(FEATURE_NAMES),
        "dataset_path": str(args.zip),
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "feature_csv": str(csv_path), "feature_csv_sha256": sha256_file(csv_path),
        "versions": {"python": sys.version.split()[0], "mne": mne.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "yaml": yaml.__version__},
        "records": reports,
    }
    audit_path = args.out / f"pilot_features_subject{subject}_audit.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    default=json_numpy_default, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS — feature pilot completed; no models trained")
    print("Features:", csv_path)
    print("Audit:", audit_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
