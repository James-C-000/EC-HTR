#!/usr/bin/env python3
"""
EC-HTR - audit of the pre-1940 scans the organizer left in RAW_DATA.

Read-only: hash every pre-1940 image still in RAW_DATA and compare it with its namesake in
PROCESSED_DATA/Organized. Walk, parse and year cutoff follow ec_organizer_enhanced.py.

ec_organizer_enhanced.py counts a pre-1940 file before it moves it, and skips the move when the
destination name already exists. So 571,712 pre-1940 files were counted and 567,917 reached Organized;
the other 3,795 are still in RAW_DATA, each next to a namesake in Organized. On the drive (22 Oct 2026):
275 are byte-identical, 3,520 differ, none lacks a namesake. The 619 that differ in PNG_230317_B0012
are corrupt RAW copies; the other 2,901 are real alternative scans and became Benchmark/extras_paths.txt
(the RAW_DATA path of every leftovers_differ_or_missing.csv row outside PNG_230317_B0012, sorted).

Writes leftovers_all.csv and leftovers_differ_or_missing.csv to OUT_DIR and the report to stdout.
REPO_ORG_DIR is this Organization folder, for ec_audit_unmatched_ids's inventory loader.
Recovered from the Claude Code session that wrote and ran it; the code below this docstring is unchanged.

usage: python -I Organization/ec_leftover_audit.py E:/RAW_DATA E:/PROCESSED_DATA OUT_DIR INV_2022 INV_2014 \
           Organization > leftover_report.txt
"""
import csv
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

raw, processed, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, sys.argv[6])
from ec_audit_unmatched_ids import IMAGE_EXTS, load_inventories  # noqa: E402

PNG_SIG = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {0: "gray", 2: "RGB", 3: "palette", 4: "gray+alpha", 6: "RGBA"}

stations = load_inventories(Path(sys.argv[4]), Path(sys.argv[5]))
organized = processed / "Organized"
if processed in out.resolve().parents or raw in out.resolve().parents:
    raise SystemExit("OUT_DIR must be outside the drive folders")
out.mkdir(parents=True, exist_ok=True)


def parse(filename):
    """(climate_id, year) by ECFileOrganizer.parse_filename()'s rules, None if it would go to Manual_QC."""
    stem = os.path.splitext(filename)[0]
    if len(stem) > 2 and stem[-2] == "_" and stem[-1].isalpha():
        stem = stem[:-2]
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    try:
        year, month = int(parts[2]), int(parts[3])
    except ValueError:
        return None
    if not 1 <= month <= 12 or not year or not parts[1]:
        return None
    return parts[1], year


print("Indexing Organized ...", flush=True)
by_name, by_lower = defaultdict(list), defaultdict(list)
for folder in sorted(os.scandir(organized), key=lambda e: e.name):
    if folder.is_dir():
        for e in os.scandir(folder.path):
            by_name[e.name].append(folder.name)
            by_lower[e.name.lower()].append((folder.name, e.name))
print(f"  {sum(map(len, by_name.values())):,} files", flush=True)

print("Walking RAW_DATA ...", flush=True)
pre, counts, xml_images = [], Counter(), 0
for dirpath, dirnames, filenames in os.walk(raw):
    in_xml = "XML" in Path(dirpath).relative_to(raw).parts
    dirnames.sort()
    for f in sorted(filenames):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        if in_xml:
            xml_images += 1   # the organizer never looked in XML folders
            continue
        p = Path(dirpath, f)
        parsed = parse(f)
        if parsed is None:
            counts["unparsed"] += 1
        elif parsed[1] > 1939:
            counts["post_1939"] += 1
        else:
            counts["pre_1940"] += 1
            pre.append((p, *parsed))
print(f"  image files left: pre-1940 {counts['pre_1940']:,}, post-1939 {counts['post_1939']:,}, "
      f"unparsed {counts['unparsed']:,}, in XML folders {xml_images:,}", flush=True)


SOF_MARKERS = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def image_header(data):
    """'FORMAT WxH detail' from the magic bytes and header, without decoding: PNG from IHDR, JPEG
    from its first SOF marker. Anything else is 'unknown' plus its first 8 bytes."""
    if data[:8] == PNG_SIG and data[12:16] == b"IHDR":
        w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        return f"PNG {w}x{h} {data[24]}-bit {COLOR_TYPES.get(data[25], data[25])}"
    if data[:3] == b"\xff\xd8\xff":
        i = 2
        while i + 9 < len(data) and data[i] == 0xFF:
            m = data[i + 1]
            if m in SOF_MARKERS:
                h, w = int.from_bytes(data[i + 5:i + 7], "big"), int.from_bytes(data[i + 7:i + 9], "big")
                return f"JPEG {w}x{h} {data[i + 9]}-component"
            if m == 0xFF:
                i += 1
            elif 0xD0 <= m <= 0xD9 or m == 0x01:
                i += 2
            else:
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
        return "JPEG (no SOF found)"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return f"unknown {data[:8].hex()}" if data else "empty"


def png_chunks(data):
    """[(start, end, type)] for each chunk, start..end covering length, type, data and CRC."""
    out, i = [], 8
    while i + 8 <= len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        out.append((i, i + 12 + n, data[i + 4:i + 8].decode("latin-1")))
        i += 12 + n
    return out


def bytes_differing(a, b):
    if a == b:
        return 0
    xor = (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")
    return len(a) - xor.count(0)


def diff_chunks(a, b):
    """For two same-length files: differing byte count per PNG chunk type of a when both are PNGs
    (e.g. 'tIME 3; IDAT 51,220'), else the total and the first differing offset."""
    chunks = png_chunks(a)
    if a[:8] != PNG_SIG or b[:8] != PNG_SIG or not chunks:
        first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        return f"{bytes_differing(a, b):,} bytes, first at offset {first:,}"
    per = Counter()
    for start, end, kind in chunks:
        per[kind] += bytes_differing(a[start:end], b[start:end])
    per["(after last chunk)"] += bytes_differing(a[chunks[-1][1]:], b[chunks[-1][1]:])
    return "; ".join(f"{k} {n:,}" for k, n in per.items() if n)


def fmt(header):
    return header.split(" ")[0]


def px(header):
    parts = header.split(" ")
    return parts[1] if len(parts) > 1 and "x" in parts[1] and parts[1].replace("x", "").isdigit() else ""


def area(header):
    w, _, h = px(header).partition("x")
    return int(w) * int(h) if w else 0


def compare(item):
    p, cid, year = item
    expected = stations.get(cid, ("", ""))[0]
    folders = by_name.get(p.name, [])
    match = "exact"
    if not folders:
        alt = by_lower.get(p.name.lower(), [])
        folders, match = [a[0] for a in alt], ("case_only" if alt else "")
        twin_name = alt[0][1] if alt else ""
    else:
        twin_name = p.name
    a = p.read_bytes()
    row = {"raw_path": p.relative_to(raw.parent).as_posix(), "batch": p.relative_to(raw).parts[0],
           "year": year, "climate_id": cid, "inventory_province": expected,
           "raw_bytes": len(a), "raw_sha256": hashlib.sha256(a).hexdigest(), "raw_header": image_header(a),
           "organized_path": "", "organized_bytes": "", "organized_sha256": "", "organized_header": "",
           "same_size_diff": "", "namesakes": len(folders), "match": match}
    if not folders:
        row["status"] = "no_namesake"
        return row
    folder = expected if expected in folders else folders[0]
    b = (organized / folder / twin_name).read_bytes()
    row.update(organized_path=f"Organized/{folder}/{twin_name}", organized_bytes=len(b),
               organized_sha256=hashlib.sha256(b).hexdigest(), organized_header=image_header(b))
    row["status"] = "identical" if row["organized_sha256"] == row["raw_sha256"] else "differs"
    if row["status"] == "differs" and len(a) == len(b):
        row["same_size_diff"] = diff_chunks(a, b)
    return row


total_bytes = sum(p.stat().st_size for p, _, _ in pre)
print(f"Hashing {len(pre):,} leftovers ({total_bytes / 1e9:.1f} GB) and their namesakes ...", flush=True)
rows, t0, last = [], time.monotonic(), 0.0
with ThreadPoolExecutor(8) as ex:
    for r in ex.map(compare, pre):
        rows.append(r)
        now = time.monotonic()
        if now - last > 30:
            last = now
            print(f"  {len(rows):,}/{len(pre):,} ({now - t0:.0f}s)", flush=True)

fields = ["status", "raw_path", "batch", "year", "climate_id", "inventory_province", "raw_bytes", "raw_header",
          "raw_sha256", "organized_path", "organized_bytes", "organized_header", "organized_sha256", "same_size_diff",
          "namesakes", "match"]
with open(out / "leftovers_differ_or_missing.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(r for r in rows if r["status"] != "identical")
with open(out / "leftovers_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

status = Counter(r["status"] for r in rows)
print(f"\nPre-1940 files still in RAW_DATA: {len(rows):,}")
for s in ("identical", "differs", "no_namesake"):
    print(f"  {s:<12}{status[s]:>7,}")
print("\nBy batch")
by_batch = defaultdict(Counter)
for r in rows:
    by_batch[r["batch"]][r["status"]] += 1
print(f"  {'batch':<56}{'identical':>10}{'differs':>9}{'none':>7}")
for b in sorted(by_batch):
    c = by_batch[b]
    print(f"  {b:<56}{c['identical']:>10,}{c['differs']:>9,}{c['no_namesake']:>7,}")
print("\nNamesake checks")
print(f"  matched only case-insensitively: {sum(r['match'] == 'case_only' for r in rows):,}")
print(f"  namesake in more than one Organized folder: {sum(r['namesakes'] > 1 for r in rows):,}")
print(f"  namesake not in the inventory province's folder: "
      f"{sum(1 for r in rows if r['organized_path'] and r['organized_path'].split('/')[1] != r['inventory_province']):,}")
print(f"  leftover names occurring more than once in RAW_DATA: "
      f"{sum(n for n in Counter(r['raw_path'].rsplit('/', 1)[1] for r in rows).values() if n > 1):,}")
diff = [r for r in rows if r["status"] == "differs"]
if diff:
    readable = {"PNG", "JPEG", "TIFF"}
    print(f"\nOf the {len(diff):,} that differ")
    print("  format (RAW copy -> Organized copy)")
    for (f1, f2), n in sorted(Counter((fmt(r["raw_header"]), fmt(r["organized_header"])) for r in diff).items()):
        print(f"    {n:>6,}  {f1} -> {f2}")
    both = [r for r in diff if fmt(r["raw_header"]) in readable and fmt(r["organized_header"]) in readable]
    print(f"  both copies a recognised image: {len(both):,}")
    print(f"    same pixel dimensions:        {sum(px(r['raw_header']) == px(r['organized_header']) != '' for r in both):,}")
    print(f"    RAW copy has more pixels:     {sum(area(r['raw_header']) > area(r['organized_header']) for r in both):,}")
    print(f"    RAW copy has fewer pixels:    {sum(area(r['raw_header']) < area(r['organized_header']) for r in both):,}")
    print(f"  only the RAW copy recognised:   "
          f"{sum(fmt(r['raw_header']) in readable and fmt(r['organized_header']) not in readable for r in diff):,}")
    print(f"  only the Organized copy recognised: "
          f"{sum(fmt(r['raw_header']) not in readable and fmt(r['organized_header']) in readable for r in diff):,}")
    print(f"  neither recognised:             "
          f"{sum(fmt(r['raw_header']) not in readable and fmt(r['organized_header']) not in readable for r in diff):,}")
    same_size = [r for r in diff if r["raw_bytes"] == r["organized_bytes"]]
    print(f"  same byte size: {len(same_size):,}, by format pair: "
          f"{dict(Counter((fmt(r['raw_header']), fmt(r['organized_header'])) for r in same_size))}")
print(f"\nCSV: {out / 'leftovers_differ_or_missing.csv'}")
