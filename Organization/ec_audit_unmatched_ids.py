#!/usr/bin/env python3
"""
EC-HTR - audit of the scans the organizer filed by the Climate ID first-digit fallback.

Read-only: nothing on the drive is moved, renamed or written. The report goes to stdout, the file
list to --out.

ec_organizer_enhanced.py looks each Climate ID up in the 2022 then the 2014 station inventory and
only for a 7-digit ID in neither falls back to its first-digit province_mapping. That mapping is off
by one (2 Alberta ... 6 Quebec, 7 Atlantic, 8 Territories; EC uses 1 BC, 2 YT/NT/NU, 3 AB, 4 SK,
5 MB, 6 ON, 7 QC, 8 NB/NS/PE/NL), so those files may sit in a real but wrong province folder.

This walks PROCESSED_DATA/Organized, parses each name by the organizer's parse_filename() rules,
loads the inventories as its load_station_data() does, and lists every file whose Climate ID is in
neither inventory, with the province the corrected first-digit scheme gives it. It also checks that
it reproduces the organizer's placement: an inventory ID should sit in its inventory province, an
unmatched one in the as-run fallback's.

Output (--out): filename, climate_id, current_folder, correct_province, one row per unmatched file.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

ORGANIZED = "Organized"
IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")   # what the organizer picked up

# The organizer's first-digit fallback as it ran over the drive.
AS_RUN_PREFIX = {"1": "British_Columbia", "2": "Alberta", "3": "Saskatchewan", "4": "Manitoba",
                 "5": "Ontario", "6": "Quebec", "7": "Atlantic", "8": "Territories"}
# EC's scheme. 2 and 8 each cover several provinces/territories, so they stay at region level here.
CORRECTED_PREFIX = {"1": "British_Columbia", "2": "Territories", "3": "Alberta", "4": "Saskatchewan",
                    "5": "Manitoba", "6": "Ontario", "7": "Quebec", "8": "Atlantic"}

# ECFileOrganizer.load_station_data()'s province_normalize, keys uppercase.
PROVINCE_NORMALIZE = {
    "BC": "British_Columbia", "ALTA": "Alberta", "SASK": "Saskatchewan", "MAN": "Manitoba",
    "ONT": "Ontario", "QUE": "Quebec", "NB": "New_Brunswick", "NS": "Nova_Scotia",
    "PEI": "Prince_Edward_Island", "NFLD": "Newfoundland_and_Labrador", "NWT": "Northwest_Territories",
    "NU": "Nunavut", "YT": "Yukon", "SD": "South_Dakota", "MT": "Montana", "ND": "North_Dakota",
    "WA": "Washington", "OTHR": "Other",
    "BRITISH COLUMBIA": "British_Columbia", "ALBERTA": "Alberta", "SASKATCHEWAN": "Saskatchewan",
    "MANITOBA": "Manitoba", "ONTARIO": "Ontario", "QUEBEC": "Quebec", "NEW BRUNSWICK": "New_Brunswick",
    "NOVA SCOTIA": "Nova_Scotia", "PRINCE EDWARD ISLAND": "Prince_Edward_Island",
    "NEWFOUNDLAND": "Newfoundland_and_Labrador", "NEWFOUNDLAND AND LABRADOR": "Newfoundland_and_Labrador",
    "NORTHWEST TERRITORIES": "Northwest_Territories", "NUNAVUT": "Nunavut", "YUKON": "Yukon",
    "YUKON TERRITORY": "Yukon",
}


def csv_field(row, *names):
    """ECFileOrganizer._get_csv_field(): the first of names the row has, even if it is empty."""
    return next((str(row[n]) for n in names if row.get(n) is not None), "")


def load_inventories(path_2022, path_2014):
    """Climate ID -> (province folder, source), as ECFileOrganizer.load_station_data() builds it:
    every 2022 row, then 2014 rows for IDs the 2022 file lacks. Unlike the organizer, a missing
    file is an error rather than a warning, since it would make every ID look unmatched."""
    stations = {}
    for path, source, id_names, only_new in ((path_2022, "2022", ("climate_id", "Climate ID", "CLIMATE ID"), False),
                                             (path_2014, "2014", ("Climate ID", "climate_id", "CLIMATE ID"), True)):
        if not path.is_file():
            raise SystemExit(f"Station inventory not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = csv_field(row, *id_names).strip()
                if cid and not (only_new and cid in stations):
                    raw = csv_field(row, "Province", "PROVINCE").strip()
                    stations[cid] = (PROVINCE_NORMALIZE.get(raw.upper()) or raw.replace(" ", "_"), source)
    return stations


def parse_climate_id(filename):
    """Climate ID by ECFileOrganizer.parse_filename()'s rules; None where it would send the file to
    Manual_QC (not 4 parts, non-integer or zero year, month outside 1-12)."""
    stem = os.path.splitext(filename)[0]
    if len(stem) > 2 and stem[-2] == "_" and stem[-1].isalpha():
        stem = stem[:-2]   # page suffix _A.._Z
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    try:
        year, month = int(parts[2]), int(parts[3])
    except ValueError:
        return None
    if not 1 <= month <= 12 or not year or not parts[1]:
        return None
    return parts[1]


def walk_organized(organized):
    """[(folder, filename)] for every image file under Organized/<folder>/, as the organizer's
    rglob over image extensions would see them."""
    files = []
    for dirpath, dirnames, filenames in os.walk(organized):
        dirnames.sort()
        rel = Path(dirpath).relative_to(organized).parts
        if not rel:
            continue   # loose files directly in Organized
        files += [(rel[0], f) for f in sorted(filenames) if f.lower().endswith(IMAGE_EXTS)]
    return files


def print_table(title, rows, cols, row_label):
    """rows: {row: Counter(col -> n)}. Prints with row and column totals."""
    width = max([len(row_label), 5] + [len(r) for r in rows]) + 2
    print(f"\n{title}")
    print(f"  {row_label:<{width}}" + "".join(f"{c:>9}" for c in cols) + f"{'total':>10}")
    for r in sorted(rows):
        print(f"  {r:<{width}}" + "".join(f"{rows[r][c]:>9,}" for c in cols) + f"{sum(rows[r].values()):>10,}")
    col_totals = Counter()
    for counts in rows.values():
        col_totals.update(counts)
    print(f"  {'total':<{width}}" + "".join(f"{col_totals[c]:>9,}" for c in cols) + f"{sum(col_totals.values()):>10,}")


def main():
    ap = argparse.ArgumentParser(description="EC-HTR audit of Climate IDs in neither station inventory (read-only)")
    ap.add_argument("processed_dir", type=Path, help=r"e.g. E:\PROCESSED_DATA")
    ap.add_argument("--inventory-2014", type=Path, required=True)
    ap.add_argument("--inventory-2022", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("unmatched_ids.csv"))
    args = ap.parse_args()

    processed = args.processed_dir.resolve()
    organized = processed / ORGANIZED
    out = args.out.resolve()
    if not organized.is_dir():
        raise SystemExit(f"Not found: {organized}")
    if processed in out.parents:
        raise SystemExit("--out must be outside PROCESSED_DATA")

    stations = load_inventories(args.inventory_2022, args.inventory_2014)
    by_source = Counter(src for _, src in stations.values())
    print(f"Station inventories: {by_source['2022']:,} IDs from 2022, {by_source['2014']:,} more from 2014, "
          f"{len(stations):,} in all")

    print(f"Indexing {organized} ...", flush=True)
    files = walk_organized(organized)
    print(f"  {len(files):,} image files in {len({f for f, _ in files})} folders")

    unmatched, unparsed = [], []
    per_folder = Counter()
    misfiled_matched = Counter()     # inventory ID not in its inventory province's folder
    misfiled_unmatched = Counter()   # unmatched ID not in the as-run fallback's folder
    for folder, name in files:
        per_folder[folder] += 1
        cid = parse_climate_id(name)
        if cid is None:
            unparsed.append(f"{folder}/{name}")
        elif cid in stations:
            if stations[cid][0] != folder:
                misfiled_matched[(folder, stations[cid][0])] += 1
        else:
            unmatched.append((name, cid, folder, CORRECTED_PREFIX.get(cid[0], "") if len(cid) == 7 and cid.isdigit() else ""))
            if AS_RUN_PREFIX.get(cid[0]) != folder:
                misfiled_unmatched[(folder, AS_RUN_PREFIX.get(cid[0], "none"))] += 1

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "climate_id", "current_folder", "correct_province"])
        w.writerows(unmatched)

    by_folder_digit = defaultdict(Counter)
    ids_by_digit = defaultdict(set)
    moves = defaultdict(lambda: [0, set()])   # (current, correct) -> [files, ids]
    for name, cid, folder, correct in unmatched:
        by_folder_digit[folder][cid[0]] += 1
        ids_by_digit[cid[0]].add(cid)
        if correct != folder:
            moves[(folder, correct)][0] += 1
            moves[(folder, correct)][1].add(cid)
    digits = sorted(ids_by_digit)

    print(f"\nFiles in Organized whose Climate ID is in neither inventory: {len(unmatched):,} "
          f"of {len(files):,} ({100 * len(unmatched) / max(len(files), 1):.2f}%)")
    print(f"Distinct unmatched Climate IDs: {len({cid for _, cid, _, _ in unmatched}):,}")
    print_table("Unmatched files by current folder and first digit of Climate ID",
                {f: by_folder_digit[f] for f in per_folder if by_folder_digit[f]}, digits, "folder")
    print_table("Distinct unmatched Climate IDs by first digit",
                {"IDs": Counter({d: len(ids_by_digit[d]) for d in digits})}, digits, "")

    print("\nUnmatched files in the wrong folder under the corrected scheme")
    if moves:
        print(f"  {'current folder':<20}{'correct province':<20}{'files':>9}{'IDs':>7}")
        for (folder, correct), (n, ids) in sorted(moves.items()):
            print(f"  {folder:<20}{correct:<20}{n:>9,}{len(ids):>7,}")
        print(f"  {'total':<40}{sum(n for n, _ in moves.values()):>9,}"
              f"{len(set().union(*(ids for _, ids in moves.values()))):>7,}")
    else:
        print("  none")
    print(f"Unmatched files already in the right folder: {len(unmatched) - sum(n for n, _ in moves.values()):,}")

    print("\nChecks that this reproduces the organizer's placement (all should be 0)")
    print(f"  names that don't parse:                              {len(unparsed):,}")
    for p in unparsed[:10]:
        print(f"    {p}")
    print(f"  inventory IDs not in their inventory province folder: {sum(misfiled_matched.values()):,}")
    for (folder, expected), n in sorted(misfiled_matched.items()):
        print(f"    {n:,} in {folder}, inventory says {expected}")
    print(f"  unmatched IDs not in the as-run fallback's folder:    {sum(misfiled_unmatched.values()):,}")
    for (folder, expected), n in sorted(misfiled_unmatched.items()):
        print(f"    {n:,} in {folder}, fallback gives {expected}")
    print(f"\nFile list: {out}")


if __name__ == "__main__":
    main()
