#!/usr/bin/env python3
"""Smoke test two Sleep-EDF SC sessions directly from an existing ZIP.

No training and no whole-dataset extraction. Each PSG/hypnogram pair is copied
into a temporary directory, read with MNE without preloading signals, and then
removed. Reports EDF/annotation metadata and reads only one 30-s EEG segment
per session. Writes a small JSON audit into the project results directory.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--subject", default="00", help="2-digit Sleep-Cassette subject ID")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    try:
        import mne
        import numpy as np
        import scipy
        from scipy.signal import welch
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}. Install MNE in the project venv.", file=sys.stderr)
        return 2

    a.subject = str(a.subject).zfill(2)
    with a.manifest.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["subject_id"] == a.subject]
    rows.sort(key=lambda r: int(r["night_index"]))
    if len(rows) != 2 or [int(r["night_index"]) for r in rows] != [1, 2]:
        raise ValueError(f"Subject {a.subject} does not have two sessions: {len(rows)} found")
    if any(r.get("sha256_match") != "matched" for r in rows):
        raise ValueError("Manifest does not show verified SHA-256 for both PSG/hypnogram pairs")
    if not a.zip.is_file():
        raise FileNotFoundError(a.zip)

    print(f"Subject {a.subject}: two records. Only first 30 s of Fpz-Cz will be read per record.")
    records = []
    with ZipFile(a.zip) as z:
        names = set(z.namelist())
        for r in rows:
            psg_member, hyp_member = r["psg_member"], r["hypnogram_member"]
            if psg_member not in names or hyp_member not in names:
                raise FileNotFoundError(f"ZIP member missing for record {r['record_id']}")
            with TemporaryDirectory(prefix="sleepedf-smoke-") as tmpdir:
                tmp = Path(tmpdir)
                psg = tmp / "record-PSG.edf"
                hyp = tmp / "record-Hypnogram.edf"
                for member, dest, expected_field in (
                    (psg_member, psg, "sha256_actual_psg"),
                    (hyp_member, hyp, "sha256_actual_hyp"),
                ):
                    sha = hashlib.sha256()
                    with z.open(member) as source, dest.open("wb") as sink:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            sha.update(chunk)
                            sink.write(chunk)
                    if sha.hexdigest() != r[expected_field]:
                        raise ValueError(f"Hash mismatch on transient copy: {member}")

                raw = mne.io.read_raw_edf(
                    str(psg), include=["EEG Fpz-Cz"], preload=False, verbose="ERROR"
                )
                annotations = mne.read_annotations(str(hyp))
                raw.set_annotations(annotations)
                if raw.ch_names != ["EEG Fpz-Cz"]:
                    raise ValueError(f"Unexpected channels: {raw.ch_names}")
                sfreq = float(raw.info["sfreq"])
                if not math.isclose(sfreq, 100.0, abs_tol=0.01):
                    raise ValueError(f"Unexpected sampling frequency: {sfreq}")
                if raw.n_times < int(30 * sfreq):
                    raise ValueError("PSG has less than 30 seconds of EEG")
                segment = raw.get_data(picks=[0], start=0, stop=int(30 * sfreq))[0]
                if not np.isfinite(segment).all():
                    raise ValueError("Nonfinite samples in first 30 s")
                _, psd = welch(
                    segment * 1e6, fs=sfreq, window="hann", nperseg=400,
                    noverlap=200, nfft=400, detrend="constant", scaling="density",
                    average="mean",
                )
                labels = Counter(str(x) for x in annotations.description)
                totals = Counter()
                for duration, label in zip(annotations.duration, annotations.description):
                    totals[str(label)] += float(duration)
                output = {
                    "record_id": r["record_id"],
                    "subject_id": r["subject_id"],
                    "night_index": int(r["night_index"]),
                    "scorer_code": r["scorer_code"],
                    "edf_start_date": r["edf_start_date"],
                    "sfreq_hz": sfreq,
                    "n_samples": int(raw.n_times),
                    "duration_h": round(raw.n_times / sfreq / 3600, 3),
                    "annotation_count": len(annotations),
                    "annotation_labels": dict(sorted(labels.items())),
                    "annotation_hours_by_label": {
                        k: round(v / 3600, 4) for k, v in sorted(totals.items())
                    },
                    "first_30s_eeg_std_microvolts": round(float(np.std(segment) * 1e6), 4),
                    "first_30s_psd_finite": bool(np.isfinite(psd).all()),
                    "edf_psg_sha256_verified_again": True,
                }
                records.append(output)
                print(f"Night {output['night_index']}: {output['record_id']} | "
                      f"{output['duration_h']} h | {output['annotation_count']} annotations | "
                      f"EEG std {output['first_30s_eeg_std_microvolts']} µV")
                print("  Labels:", ", ".join(f"{k}={v}" for k,v in sorted(labels.items())))
                del raw, segment, annotations

    a.out.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "Two-night PSG+annotation reader smoke test; no model or full-night feature extraction",
        "source_archive": str(a.zip),
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "versions": {"python": sys.version.split()[0], "mne": mne.__version__, "scipy": scipy.__version__},
        "records": records,
    }
    outfile = a.out / "smoke_test_two_nights.json"
    outfile.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS — two PSGs read and annotation labels inspected; 30-s segments sampled.")
    print("Output:", outfile)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
