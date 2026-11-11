#!/usr/bin/env python3
"""Read-only inventory of Sleep-EDF Expanded / Sleep-Cassette from its ZIP.

Produces CSV metadata and a JSON audit; does not extract EDF recordings or
require MNE, pandas, or third-party packages. Run with --verify-hashes for a
streaming comparison against the dataset's SHA256SUMS.txt (slower, still no
extraction). No physiological signal or hypnogram annotations are inspected.

Source: PhysioNet Sleep-EDF Expanded v1.0.0, DOI 10.13026/C2X676.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from zipfile import BadZipFile, ZipFile

# SC4 + 2-digit subject + session 1/2 + tape identifier + scorer/PSG suffix.
PATTERN = re.compile(
    r"^(?P<stem>SC4(?P<subject>\d{2})(?P<night>[12])(?P<tape>[A-Z]))"
    r"(?P<suffix>[A-Z0-9])-(?P<kind>PSG|Hypnogram)\.edf$"
)
CHECKSUM = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")
EXPECTED_SINGLETONS = {"13": [1], "36": [2], "52": [2]}
EXPECTED_DISCORDANT = {"07", "17", "27", "32", "37", "60", "62", "72"}
MANIFEST_FIELDS = [
    "subject_id", "record_id", "night_index", "scorer_code", "same_scorer",
    "date_concordant", "edf_start_date", "edf_start_time", "duration_h",
    "psg_member", "hypnogram_member", "psg_uncompressed_bytes", "hyp_uncompressed_bytes",
    "sha256_expected_psg", "sha256_expected_hyp", "sha256_actual_psg",
    "sha256_actual_hyp", "sha256_match", "edf_header_status",
    "age", "sex", "raw_label_counts", "n_epochs_window", "n_epochs_rejected",
    "exclusion_reason",
]
PAIR_FIELDS = [
    "subject_id", "n_records", "record_1", "record_2", "scorer_1",
    "scorer_2", "same_scorer", "date_1", "date_2", "date_delta_days",
    "date_concordant",
]


def relative_member(name: str) -> str:
    """Drop a ZIP's enclosing directory, leaving PhysioNet-relative paths."""
    pos = name.find("sleep-cassette/")
    if pos >= 0:
        return name[pos:]
    return name.split("/", 1)[-1] if "/" in name else name


def read_checksums(z: ZipFile) -> dict[str, str]:
    matches = [n for n in z.namelist() if n.endswith("/SHA256SUMS.txt") or n == "SHA256SUMS.txt"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one SHA256SUMS.txt in ZIP; found {len(matches)}")
    text = z.read(matches[0]).decode("utf-8-sig")
    result = {}
    for line in text.splitlines():
        m = CHECKSUM.match(line.strip())
        if not m:
            if line.strip():
                raise ValueError(f"Unrecognized checksum line: {line[:100]!r}")
            continue
        checksum, path = m.group(1).lower(), m.group(2).strip().removeprefix("./")
        if path in result:
            raise ValueError(f"Duplicate checksum path: {path}")
        result[path] = checksum
    return result


def parse_edf_date(value: str) -> str:
    # EDF uses dd.mm.yy. The SC cohort is from the late 1980s/early 1990s.
    parts = value.split(".")
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise ValueError(f"Bad EDF date: {value!r}")
    d, m, yy = (int(p) for p in parts)
    year = 1900 + yy if yy >= 85 else 2000 + yy
    return date(year, m, d).isoformat()


def read_edf_header(z: ZipFile, member: str) -> dict[str, object]:
    # EDF fixed header offsets; only 256 bytes read, never all EEG samples.
    with z.open(member) as f:
        data = f.read(256)
    if len(data) != 256:
        raise ValueError(f"EDF header shorter than 256 bytes: {member}")
    field = lambda a, b: data[a:b].decode("ascii", errors="replace").strip()
    raw_date, raw_time = field(168, 176), field(176, 184)
    n_records, seconds_per_record = int(field(236, 244)), float(field(244, 252))
    if n_records < 0 or seconds_per_record <= 0:
        raise ValueError("Invalid n_records or duration in EDF header")
    return {
        "edf_start_date": parse_edf_date(raw_date),
        "edf_start_time": raw_time,
        "duration_h": round(n_records * seconds_per_record / 3600, 5),
    }


def sha256_zip_member(z: ZipFile, member: str) -> str:
    sha = hashlib.sha256()
    with z.open(member) as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_csv(dest: Path, rows: list[dict], fields: list[str]) -> None:
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(zip_path: Path, outdir: Path, verify_hashes: bool) -> int:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Cannot find ZIP: {zip_path}")
    outdir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as z:
        checksum = read_checksums(z)
        selected: dict[tuple[str, int], dict] = defaultdict(dict)
        unrecognized = []
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "sleep-cassette/" not in name or not name.endswith(".edf"):
                continue
            basename = Path(name).name
            m = PATTERN.fullmatch(basename)
            if m is None:
                unrecognized.append(name)
                continue
            role = m.group("kind")
            if role == "PSG" and m.group("suffix") != "0":
                raise ValueError(f"Unexpected PSG suffix: {basename}")
            key = (m.group("subject"), int(m.group("night")))
            if role in selected[key]:
                raise ValueError(f"Duplicate {role} for subject/night {key}")
            selected[key][role] = (info, m)

        rows = []
        missing_mates = []
        for (subject, night), entries in sorted(selected.items()):
            if set(entries) != {"PSG", "Hypnogram"}:
                missing_mates.append({"subject": subject, "night": night,
                                      "present": sorted(entries)})
                continue
            psg, psgmatch = entries["PSG"]
            hyp, hypmatch = entries["Hypnogram"]
            if psgmatch.group("stem") != hypmatch.group("stem"):
                raise ValueError(f"PSG/hypnogram mismatch for {subject}, {night}")
            psg_rel, hyp_rel = relative_member(psg.filename), relative_member(hyp.filename)
            expected_psg, expected_hyp = checksum.get(psg_rel), checksum.get(hyp_rel)
            data = {
                "subject_id": subject,
                "record_id": psgmatch.group("stem"),
                "night_index": night,
                "scorer_code": hypmatch.group("suffix"),
                "psg_member": psg.filename,
                "hypnogram_member": hyp.filename,
                "psg_uncompressed_bytes": psg.file_size,
                "hyp_uncompressed_bytes": hyp.file_size,
                "sha256_expected_psg": expected_psg or "",
                "sha256_expected_hyp": expected_hyp or "",
                "sha256_actual_psg": "",
                "sha256_actual_hyp": "",
                "sha256_match": "not_verified",
                "age": "", "sex": "", "raw_label_counts": "",
                "n_epochs_window": "", "n_epochs_rejected": "",
                "exclusion_reason": "", "same_scorer": "", "date_concordant": "",
                "edf_start_date": "", "edf_start_time": "", "duration_h": "",
                "edf_header_status": "",
            }
            try:
                data.update(read_edf_header(z, psg.filename))
                data["edf_header_status"] = "ok"
            except (ValueError, OSError) as exc:
                data["edf_header_status"] = f"ERROR: {exc}"
            if verify_hashes:
                data["sha256_actual_psg"] = sha256_zip_member(z, psg.filename)
                data["sha256_actual_hyp"] = sha256_zip_member(z, hyp.filename)
                data["sha256_match"] = (
                    "matched" if expected_psg == data["sha256_actual_psg"]
                    and expected_hyp == data["sha256_actual_hyp"]
                    else "MISMATCH_OR_MISSING_REFERENCE"
                )
            rows.append(data)

    by_subj = defaultdict(dict)
    for row in rows:
        by_subj[row["subject_id"]][row["night_index"]] = row
    pairs = []
    for subject, nights in sorted(by_subj.items()):
        a, b = nights.get(1), nights.get(2)
        same = (a["scorer_code"] == b["scorer_code"]) if a and b else None
        daydelta = None
        if a and b and a["edf_start_date"] and b["edf_start_date"]:
            daydelta = (date.fromisoformat(b["edf_start_date"]) -
                        date.fromisoformat(a["edf_start_date"])).days
        concordant = daydelta == 1 if daydelta is not None else None
        pair = {
            "subject_id": subject, "n_records": len(nights),
            "record_1": a["record_id"] if a else "",
            "record_2": b["record_id"] if b else "",
            "scorer_1": a["scorer_code"] if a else "",
            "scorer_2": b["scorer_code"] if b else "",
            "same_scorer": same if same is not None else "",
            "date_1": a["edf_start_date"] if a else "",
            "date_2": b["edf_start_date"] if b else "",
            "date_delta_days": daydelta if daydelta is not None else "",
            "date_concordant": concordant if concordant is not None else "",
        }
        pairs.append(pair)
        for row in nights.values():
            row["same_scorer"] = same if same is not None else ""
            row["date_concordant"] = concordant if concordant is not None else ""

    complete = [x for x in pairs if x["n_records"] == 2]
    singles = {x["subject_id"]: [int(r["night_index"]) for r in rows
                                if r["subject_id"] == x["subject_id"]]
               for x in pairs if x["n_records"] == 1}
    discordant = sorted(x["subject_id"] for x in complete if x["date_concordant"] is False)
    unknown_dates = sorted(x["subject_id"] for x in complete if x["date_concordant"] == "")
    same_count = sum(x["same_scorer"] is True for x in complete)
    diff_count = sum(x["same_scorer"] is False for x in complete)
    counts = Counter(x["scorer_code"] for x in rows)
    tests = {
        "recordings_153": len(rows) == 153,
        "subjects_78": len(by_subj) == 78,
        "complete_pairs_75": len(complete) == 75,
        "singletons_13_36_52": singles == EXPECTED_SINGLETONS,
        "discordant_8_subjects": set(discordant) == EXPECTED_DISCORDANT,
        "date_readable_everywhere": not unknown_dates and all(x["edf_header_status"] == "ok" for x in rows),
        "same_scorer_26": same_count == 26,
        "different_scorer_49": diff_count == 49,
        "scorer_codes_11": len(counts) == 11,
        "complete_psg_hyp_pairs": not missing_mates,
        "no_unrecognized_sc_edf": not unrecognized,
        "all_expected_sha256_present": all(x["sha256_expected_psg"] and x["sha256_expected_hyp"] for x in rows),
    }
    if verify_hashes:
        tests["all_sha256_matched"] = all(x["sha256_match"] == "matched" for x in rows)

    manifest = outdir / "manifest_inventory.csv"
    pair_file = outdir / "pair_audit.csv"
    write_csv(manifest, rows, MANIFEST_FIELDS)
    write_csv(pair_file, pairs, PAIR_FIELDS)
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "dataset": "PhysioNet Sleep-EDF Expanded 1.0.0 / Sleep-Cassette",
        "source_archive_name": zip_path.name,
        "source_archive_bytes": zip_path.stat().st_size,
        "inspection": "ZIP central directory + PSG EDF fixed headers only",
        "sha256_entry_verification": "complete" if verify_hashes else "not performed",
        "age_and_sex": "not extracted; SC-subjects.xls pending",
        "hypnogram_annotations": "not read",
        "number_of_subjects": len(by_subj),
        "number_of_recordings": len(rows),
        "number_of_complete_pairs": len(complete),
        "single_night_subjects": singles,
        "date_discordant_subjects": discordant,
        "date_unreadable_subjects": unknown_dates,
        "same_scorer_pairs": same_count,
        "different_scorer_pairs": diff_count,
        "scorer_counts": dict(sorted(counts.items())),
        "unrecognized_edf_members": unrecognized,
        "unpaired_recordings": missing_mates,
        "manifest_sha256": manifest_digest,
        "checks": tests,
        "overall_pass": all(tests.values()),
    }
    (outdir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SLEEP-EDF Sleep-Cassette inventory")
    print("  Subjects / records / paired subjects:", len(by_subj), len(rows), len(complete))
    print("  Matching vs different scorers:", same_count, diff_count)
    print("  Discordant header dates:", ", ".join(discordant))
    print("  Manifest SHA-256:", manifest_digest)
    print("  Output:", manifest, pair_file, outdir / "audit_summary.json", sep="\n    ")
    failed = [k for k, ok in tests.items() if not ok]
    print("  Checks:", "PASS" if not failed else "FAILED: " + ", ".join(failed))
    return 0 if not failed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True,
                        help="Path to existing sleep-edfx-1.0.0.zip")
    parser.add_argument("--out", dest="outdir", type=Path, required=True,
                        help="Output directory for three small files")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="SHA-256 check all 306 EDF entries in streaming mode; no extraction")
    args = parser.parse_args()
    try:
        return build(args.zip_path.expanduser(), args.outdir.expanduser(), args.verify_hashes)
    except (OSError, ValueError, BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
