#!/usr/bin/env python3
"""Read-only scientific preflight for the complete Sleep-EDF SC feature matrix.

Runs locally, reading *already extracted features*; it never accesses the dataset ZIP,
reads EEG samples, fits a model, or edits the existing input files.

Usage (from the repository root):
  python src/validate_sleep_edf_features_v1_2.py \
    --features results/features_v1_2/features_all_153.csv \
    --audit results/features_v1_2/features_all_153_audit.json \
    --manifest results/manifest_inventory.csv \
    --window-audit results/window_audit_all_153.csv \
    --config config/study_v1_2.yaml \
    --out results/features_v1_2/pre_model_QA_v1_2.json

All checks fail closed. Produces a JSON record of checks and the exact hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

STAGES = ("W", "N1", "N2", "N3", "REM")
BANDS = ("delta", "theta", "alpha", "sigma", "beta")
FEATURES = [f"log10_{b}_uv2" for b in BANDS] + [f"rel_{b}" for b in BANDS]
COLUMNS = ["subject_id", "record_id", "night_index", "epoch_index",
           "epoch_start_sec", "stage", *FEATURES]
EXPECTED_SUBJECTS, EXPECTED_RECORDS, EXPECTED_PAIRS = 78, 153, 75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def must(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject_id": str, "record_id": str},
                       keep_default_na=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("features", "audit", "manifest", "window-audit", "config", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()

    inputs = (args.features, args.audit, args.manifest, args.window_audit, args.config)
    for path in inputs:
        must(path.is_file(), f"Missing input: {path}")
    must(args.out.parent.is_dir(), f"Output folder does not exist: {args.out.parent}")
    must(not args.out.exists(), f"Output exists: {args.out}. Never overwrite a QA record.")
    paths_sha = {str(path): sha256(path) for path in inputs}

    agg = json.loads(args.audit.read_text(encoding="utf-8"))
    must(agg.get("status") == "PASS", "Combined feature audit is not PASS")
    must(agg.get("aggregate_sha256") == sha256(args.features), "Combined CSV SHA-256 mismatch")
    h = agg.get("input_hashes", {})
    must(h.get("manifest_sha256") == sha256(args.manifest), "Manifest SHA mismatch")
    must(h.get("window_csv_sha256") == sha256(args.window_audit), "Window audit SHA mismatch")
    must(h.get("config_sha256") == sha256(args.config), "Configuration SHA mismatch")

    df = read_csv(args.features)
    manifest = read_csv(args.manifest)
    windows = read_csv(args.window_audit)
    must(list(df.columns) == COLUMNS, f"Unexpected feature columns: {list(df.columns)}")
    must(df.shape[0] == int(agg.get("feature_rows", -1)), "Feature count vs audit differs")
    must(df.shape[0] > 0, "Empty feature matrix")
    must(len(manifest) == EXPECTED_RECORDS and len(windows) == EXPECTED_RECORDS,
         "Manifest or window audit does not contain 153 records")
    must(df["record_id"].nunique() == EXPECTED_RECORDS,
         "Feature matrix does not contain 153 records")
    must(df["subject_id"].nunique() == EXPECTED_SUBJECTS,
         "Feature matrix does not contain 78 subjects")
    must(len(df) == len(df[["record_id", "epoch_index"]].drop_duplicates()),
         "Duplicate (record_id,epoch_index) rows")
    must(set(df["stage"]).issubset(STAGES), "Unexpected stage label")
    must(df["stage"].notna().all(), "Empty stage label")

    for key in ("night_index", "epoch_index", "epoch_start_sec"):
        numeric = pd.to_numeric(df[key], errors="coerce")
        must(numeric.notna().all(), f"Non-numeric value in {key}")
        must((numeric == numeric.astype(np.int64)).all(), f"Fractional value in {key}")
        df[key] = numeric.astype(np.int64)
    must((df["epoch_index"] >= 0).all(), "Negative epoch index")
    must((df["epoch_start_sec"] == 30 * df["epoch_index"]).all(),
         "Epoch times are not on a 30 s grid")
    must(df["night_index"].isin([1, 2]).all(), "Invalid night index")

    # Check feature values and their mathematical invariant.
    matrix = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    must(np.isfinite(matrix).all(), "Missing / infinite feature values")
    relative = matrix[:, 5:]
    must(np.logical_and(relative >= -1e-12, relative <= 1 + 1e-12).all(),
         "Relative powers outside [0,1]")
    must(np.allclose(relative.sum(axis=1), 1, rtol=0, atol=1e-8),
         "Relative power does not sum to one")

    # Join each record to the frozen manifest and window: never infer identities
    # by slicing the filename.
    must(manifest["record_id"].is_unique and windows["record_id"].is_unique,
         "Duplicated records in manifest or window audit")
    must(set(df["record_id"]) == set(manifest["record_id"]) == set(windows["record_id"]),
         "Record IDs disagree across sources")
    m = manifest.set_index("record_id")
    w = windows.set_index("record_id")
    per_rec = df.groupby("record_id", sort=False)
    per_rec_count = per_rec.size()
    must((per_rec["subject_id"].nunique() == 1).all(),
         "Record assigned to multiple subjects")
    must((per_rec["night_index"].nunique() == 1).all(),
         "Record assigned to multiple night indices")
    for rec_id, group in per_rec:
        subject = str(group["subject_id"].iloc[0])
        night = int(group["night_index"].iloc[0])
        must(subject == str(m.at[rec_id, "subject_id"]),
             f"Subject ID differs from manifest for {rec_id}")
        must(night == int(m.at[rec_id, "night_index"]),
             f"Night index differs from manifest for {rec_id}")
        start = int(w.at[rec_id, "main_start_epoch"])
        end = int(w.at[rec_id, "main_end_epoch_exclusive"])
        must(((group["epoch_index"] >= start) & (group["epoch_index"] < end)).all(),
             f"Epoch outside approved main window for {rec_id}")
        must(len(group) <= int(w.at[rec_id, "main_window_epochs"]),
             f"Too many epochs in {rec_id}")
        rec_audit = args.features.parent / "records" / rec_id / "audit.json"
        rec_csv = args.features.parent / "records" / rec_id / "features.csv"
        must(rec_audit.is_file() and rec_csv.is_file(), f"Missing saved record: {rec_id}")
        ra = json.loads(rec_audit.read_text(encoding="utf-8"))
        must(ra.get("extraction", {}).get("feature_rows") == len(group),
             f"Per-record audit row count differs for {rec_id}")
        expected_hash = agg.get("per_record_hashes", {}).get(rec_id)
        must(bool(expected_hash) and sha256(rec_csv) == expected_hash,
             f"Per-record CSV hash differs for {rec_id}")

    observed_counts = Counter(df["stage"])
    audit_counts = Counter(agg.get("stage_counts", {}))
    must(all(observed_counts.get(s, 0) == audit_counts.get(s, 0) for s in STAGES),
         "Stage counts differ from combined audit")
    must(all(observed_counts.get(s, 0) > 0 for s in STAGES),
         "At least one global stage has no support")
    subjects = manifest.groupby("subject_id")["record_id"].nunique()
    must(len(subjects) == EXPECTED_SUBJECTS,
         "Manifest missing subjects")
    must((subjects == 2).sum() == EXPECTED_PAIRS and (subjects == 1).sum() == 3,
         "Unexpected 75 pairs / 3 singleton subjects")
    per_subject_stages = pd.crosstab(df["subject_id"], df["stage"]).reindex(columns=STAGES, fill_value=0)
    # Also audit the exact 78-fold schedule without fitting a single model.
    subject_ids = df["subject_id"].to_numpy()
    splits = LeaveOneGroupOut()
    seen = np.zeros(len(df), dtype=np.uint8)
    for train_i, test_i in splits.split(matrix, df["stage"].to_numpy(), subject_ids):
        train_groups = set(subject_ids[train_i])
        test_groups = set(subject_ids[test_i])
        must(len(test_groups) == 1 and train_groups.isdisjoint(test_groups),
             "LOSO subject overlap / unexpected test subject count")
        must(len(train_groups) == 77, "Unexpected LOSO training subject count")
        seen[test_i] += 1
        train_stages = set(df["stage"].to_numpy()[train_i])
        must(train_stages == set(STAGES), "A LOSO training fold lacks a stage")
    must((seen == 1).all(), "LOSO test coverage is not exactly once per epoch")

    report = {
        "status": "PASS",
        "scope": "read-only matrix preflight and fold schedule, no model training",
        "records": int(df["record_id"].nunique()),
        "subjects": int(df["subject_id"].nunique()),
        "paired_subjects": int((subjects == 2).sum()),
        "feature_rows": int(len(df)),
        "feature_names": FEATURES,
        "stage_counts": {s: int(observed_counts[s]) for s in STAGES},
        "subject_rows_min": int(df.groupby("subject_id").size().min()),
        "subject_rows_max": int(df.groupby("subject_id").size().max()),
        "subjects_missing_stage": {s: int((per_subject_stages[s] == 0).sum()) for s in STAGES},
        "loso_folds": 78,
        "train_groups_per_fold": 77,
        "test_coverage_exactly_once": True,
        "input_sha256": paths_sha,
    }
    # Atomic creation of a new audit; no overwrites of previous artifacts.
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".part",
                                         prefix=".qa-", dir=args.out.parent, delete=False) as t:
            temp_name = t.name
            json.dump(report, t, indent=2, ensure_ascii=False, allow_nan=False)
            t.write("\n")
        os.rename(temp_name, args.out)
        temp_name = None
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)

    print("PASS — feature matrix / provenance / 78 LOSO folds verified; no model trained")
    print(f"Subjects: {report['subjects']} | Records: {report['records']} | Rows: {report['feature_rows']}")
    print("Stages:", report["stage_counts"])
    print("Subjects missing each stage:", report["subjects_missing_stage"])
    print("Subject row range:", report["subject_rows_min"], "to", report["subject_rows_max"])
    print("QA report:", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import sys
        print(f"FAILED — {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
