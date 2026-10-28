#!/usr/bin/env python3
"""
EC-HTR - integrity check of PROCESSED_DATA on the external HDD, run before the H200 production run.

Read-only: nothing on the drive is moved, renamed or written. Reports go to --out-dir.

1. Counts. Files per Organized/<province>, Irrelevant_Post1939 and Manual_QC against the organizer's
   organization_stats.json. The organizer counted every file it found, including the ones it then
   skipped because a file of the same name was already at the destination. Those are still in
   RAW_DATA, so each is matched to its same-name twin, subtracted from the twin's folder, and
   byte-compared with it.
2. Names. Every file parses as 9904_CLIMATEID_YYYY_MM[_X].ext, sits in the right era
   (Organized <= 1939 < Irrelevant_Post1939), and no scan name appears twice.
3. Contents (Organized only, unless --scope all). Each file is read once, SHA-256 hashed, Pillow
   verify()'d (chunk CRCs, truncation) and fully decoded, as the H200 pipeline does with
   Image.open(path).convert("RGB"). Flags unreadable, empty and undecodable files, images over
   Pillow's decompression-bomb limit, 16-bit modes that convert("RGB") clips, multi-frame files,
   tiny images and identical content under two names.

Outputs in --out-dir:
  manifest.csv       one row per content-checked file: bytes, SHA-256, format, mode, size in px, status
                     (status covers the file's name and content checks; byte-identical duplicates
                     and count problems are only in issues.csv)
  issues.csv         every FAIL / WARN / INFO finding
  raw_leftovers.csv  RAW_DATA files the organizer skipped, their twin, and whether the bytes match
  summary.json       count table, content stats, per-check totals
Exit code 0 when nothing FAILs, 1 otherwise.

Reading all 2.3 TB of Organized from the USB HDD takes about 4-6 hours (the drive tops out near
155 MB/s). Ctrl+C stops cleanly and --resume continues: files already checked OK/WARN are skipped,
FAILs are checked again.
"""

import argparse
import csv
import filecmp
import hashlib
import io
import json
import os
import re
import signal
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ORGANIZED, POST1939, MANUAL_QC = "Organized", "Irrelevant_Post1939", "Manual_QC"
IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")   # what the organizer picked up
NAME_RE = re.compile(r"9904_([0-9A-Za-z]+)_(\d{4})_(\d{2})(?:_([A-Za-z]))?\.(?:png|tiff?|jpe?g)", re.IGNORECASE)
FIRST_YEAR = 1840                    # the organizer's sanity bound
BOMB_PX = Image.MAX_IMAGE_PIXELS     # Image.open warns above this and raises above 2x; production keeps the default
Image.MAX_IMAGE_PIXELS = None        # ...but the checker decodes such images anyway, to report them
MIN_SIDE = 256                       # px; smaller than any real form scan
CLIPPED_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}   # convert("RGB") clips these to 8 bits

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"
CHECKS = {  # name: (severity, meaning). FAIL = blocks the H200 run or the counts don't add up.
    "count_mismatch":    (FAIL, "folder holds a different number of files than the organizer reported"),
    "missing_twin":      (FAIL, "RAW_DATA file the organizer skipped as already present has no copy in PROCESSED_DATA"),
    "not_an_image":      (FAIL, "non-image or OS metadata file (AppleDouble ._*, Thumbs.db, ...)"),
    "bad_name":          (FAIL, "name doesn't parse as 9904_CLIMATEID_YYYY_MM[_X].ext with month 01-12"),
    "wrong_era":         (FAIL, "post-1939 file in Organized, or pre-1940 file in Irrelevant_Post1939"),
    "read_error":        (FAIL, "file could not be read from the drive"),
    "empty_file":        (FAIL, "0-byte file"),
    "undecodable":       (FAIL, "Pillow cannot open, verify or fully decode the file"),
    "bomb_error":        (FAIL, f"over {2 * BOMB_PX:,} px: Image.open raises DecompressionBombError"),
    "bomb_warning":      (WARN, f"over {BOMB_PX:,} px: Image.open emits DecompressionBombWarning"),
    "clipped_mode":      (WARN, "16-bit/float image: convert('RGB') clips it to 8 bits"),
    "multi_frame":       (WARN, "more than one frame/page: only the first is sent"),
    "tiny_image":        (WARN, f"shorter side under {MIN_SIDE} px"),
    "odd_year":          (WARN, f"year before {FIRST_YEAR} or in the future"),
    "duplicate_name":    (WARN, "same scan name (ignoring extension and case) in more than one place"),
    "duplicate_content": (WARN, "byte-identical to a file with a different name"),
    "divergent_twin":    (WARN, "differs from the same-name RAW_DATA copy the organizer skipped"),
    "leftover_count":    (WARN, "RAW_DATA leftovers don't match the organizer's skipped_existing count"),
    "unexpected_layout": (WARN, "subfolder or loose file the checks don't cover"),
    "orphan_page":       (INFO, "page-suffixed file (_A, _B, ...) whose base page isn't in the same folder"),
}
SEVERITY_ORDER = {FAIL: 0, WARN: 1, INFO: 2}
MANIFEST_FIELDS = ["path", "bytes", "sha256", "format", "mode", "width", "height", "frames", "status", "issues"]
LEFTOVER_FIELDS = ["raw_path", "batch", "year", "twin_path", "raw_bytes", "twin_bytes", "identical"]


# --------------------------------------------------------------------------- index and names

def index_processed(processed):
    """[(folder, name, bytes)] for every file in Organized/*, Irrelevant_Post1939 and Manual_QC,
    plus layout issues. Directory metadata only: no file contents are read."""
    files, issues = [], []

    def list_folder(path, folder):
        for e in os.scandir(path):
            if e.is_dir():
                issues.append((f"{folder}/{e.name}", "unexpected_layout", "subfolder (not checked)"))
            else:
                files.append((folder, e.name, e.stat().st_size))

    for e in os.scandir(processed):
        if not e.is_dir():
            continue  # organization_stats.*, resolved_non_standard_ids.csv
        if e.name == ORGANIZED:
            for p in os.scandir(e.path):
                if p.is_dir():
                    list_folder(p.path, f"{ORGANIZED}/{p.name}")
                else:
                    issues.append((f"{ORGANIZED}/{p.name}", "unexpected_layout", "loose file outside a province folder"))
        elif e.name in (POST1939, MANUAL_QC):
            list_folder(e.path, e.name)
        else:
            issues.append((e.name, "unexpected_layout", "unknown folder (not checked)"))
    files.sort()
    return files, issues


def name_issues(folder, name):
    """Checks that need only the file's name and the folder it sits in."""
    if name.startswith("._") or not name.lower().endswith(IMAGE_EXTS):
        return [("not_an_image", name)]
    m = NAME_RE.fullmatch(name)
    if not m or not 1 <= int(m[3]) <= 12:
        return [("bad_name", name)]
    year, found = int(m[2]), []
    if folder != MANUAL_QC and (year <= 1939) != folder.startswith(ORGANIZED + "/"):
        found.append(("wrong_era", f"{year} in {folder}"))
    if not FIRST_YEAR <= year <= datetime.now().year:
        found.append(("odd_year", str(year)))
    return found


def cross_name_issues(files):
    """duplicate_name and orphan_page: checks that compare names with each other."""
    found, by_stem, stems_in = [], defaultdict(list), defaultdict(set)
    for folder, name, _ in files:
        if NAME_RE.fullmatch(name):
            stem = os.path.splitext(name)[0].lower()
            by_stem[stem].append(f"{folder}/{name}")
            stems_in[folder].add(stem)
    for paths in by_stem.values():
        for p in paths if len(paths) > 1 else ():
            found.append((p, "duplicate_name", "also " + ", ".join(q for q in paths if q != p)))
    for folder, name, _ in files:
        m = NAME_RE.fullmatch(name)
        base = os.path.splitext(name)[0][:-2]
        if m and m[4] and base.lower() not in stems_in[folder]:
            found.append((f"{folder}/{name}", "orphan_page", f"no {base} in {folder}"))
    return found


# --------------------------------------------------------------------------- counts

def raw_leftovers(raw):
    """Image files still in RAW_DATA, skipping XML folders as the organizer did."""
    out = []
    for dirpath, dirnames, filenames in os.walk(raw):
        dirnames[:] = sorted(d for d in dirnames if d != "XML")
        out += [Path(dirpath, f) for f in sorted(filenames) if f.lower().endswith(IMAGE_EXTS)]
    return out


def reconcile(files, processed, stats, raw, limit, issues):
    """Count table {folder: (organizer, skipped, expected, found)} and raw_leftovers.csv rows.
    Expected = the organizer's count minus the same-name files it skipped for that folder."""
    found = Counter(folder for folder, *_ in files)
    organizer = {f"{ORGANIZED}/{p}": n for p, n in stats["by_province"].items()}
    organizer.update({POST1939: stats["post_1939"], MANUAL_QC: stats["manual_qc"]})
    skipped, where = Counter(), defaultdict(list)
    for folder, name, size in files:
        where[name.lower()].append((folder, name, size))

    rows, leftovers = [], raw_leftovers(raw) if raw.is_dir() else None
    for path in leftovers or ():
        m = NAME_RE.fullmatch(path.name)
        year = int(m[2]) if m else None
        post = year is not None and year > 1939
        twins = where.get(path.name.lower(), [])
        twin = next((t for t in twins if (t[0] == POST1939) == post), twins[0] if twins else None)
        rel = path.relative_to(raw.parent).as_posix()
        if twin:
            skipped[twin[0]] += 1
        else:
            if post:
                skipped[POST1939] += 1   # a pre-1940 one can't be attributed without the station inventory
            issues.append((rel, "missing_twin", "no file of this name anywhere in PROCESSED_DATA"))
        rows.append({"raw_path": rel, "batch": path.relative_to(raw).parts[0], "year": year,
                     "twin_path": f"{twin[0]}/{twin[1]}" if twin else "", "raw_bytes": path.stat().st_size,
                     "twin_bytes": twin[2] if twin else "", "identical": ""})

    if leftovers is None:
        issues.append((str(raw), "leftover_count", "RAW_DATA not found: per-folder counts can't be reconciled"))
    elif len(leftovers) != stats["skipped_existing"] + stats["errors"]:
        issues.append((str(raw), "leftover_count", f"{len(leftovers):,} image files left in RAW_DATA, organizer "
                       f"skipped {stats['skipped_existing']:,} and failed on {stats['errors']:,}"))

    pairs = [r for r in rows if r["twin_path"]][:limit or None]
    if pairs:
        print(f"  byte-comparing {len(pairs):,} RAW_DATA leftovers with their twins ...", flush=True)

        def same(r):
            try:
                return filecmp.cmp(raw.parent / r["raw_path"], processed / r["twin_path"], shallow=False)
            except OSError:
                return None
        with ThreadPoolExecutor(8) as ex:
            for r, identical in zip(pairs, ex.map(same, pairs)):
                r["identical"] = identical
                if identical is False:
                    issues.append((r["twin_path"], "divergent_twin",
                                   f"{r['twin_bytes']:,} bytes vs {r['raw_bytes']:,} in {r['raw_path']}"))

    table = {}
    for folder in sorted(set(organizer) | set(found)):
        row = (organizer.get(folder, 0), skipped[folder], organizer.get(folder, 0) - skipped[folder], found[folder])
        if any(row):
            table[folder] = row
            if leftovers is not None and row[2] != row[3]:
                issues.append((folder, "count_mismatch", f"expected {row[2]:,}, found {row[3]:,}"))
    table["TOTAL"] = (stats["total_files"], sum(skipped.values()), stats["moved_files"], len(files))
    if stats["moved_files"] != len(files):
        issues.append(("TOTAL", "count_mismatch", f"organizer moved {stats['moved_files']:,}, found {len(files):,}"))
    return table, rows


def print_counts(table, reconciled):
    print(f"\n  {'folder':<38}{'organizer':>11}{'skipped':>9}{'expected':>10}{'found':>10}")
    for folder, (org, skip, exp, got) in table.items():
        verdict = "" if not reconciled and folder != "TOTAL" else "ok" if exp == got else "MISMATCH"
        print(f"  {folder:<38}{org:>11,}{skip:>9,}{exp:>10,}{got:>10,}  {verdict}")
    print("  (skipped = same-name files the organizer counted but left in RAW_DATA)\n")


# --------------------------------------------------------------------------- contents

def _quiet_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C is the main process's to handle
    warnings.simplefilter("ignore")                # e.g. Pillow's corrupt-EXIF chatter


def check_content(path):
    """Read, hash and fully decode one file the way the H200 pipeline opens it. Runs in a worker."""
    rec = {"bytes": "", "sha256": "", "format": "", "mode": "", "width": "", "height": "", "frames": ""}
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return rec, [("read_error", str(e))]
    rec["bytes"] = len(data)
    if not data:
        return rec, [("empty_file", "0 bytes")]
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            rec.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                       frames=getattr(im, "n_frames", 1))
            im.verify()   # chunk CRCs and structure
        with Image.open(io.BytesIO(data)) as im:
            im.load()     # full decode
    except UnidentifiedImageError:
        return rec, [("undecodable", "not a recognised image format")]
    except Exception as e:  # Pillow raises OSError, SyntaxError, ValueError, zlib.error, EOFError, ...
        return rec, [("undecodable", f"{type(e).__name__}: {e}")]
    w, h, found = rec["width"], rec["height"], []
    if w * h > 2 * BOMB_PX:
        found.append(("bomb_error", f"{w}x{h}"))
    elif w * h > BOMB_PX:
        found.append(("bomb_warning", f"{w}x{h}"))
    if rec["mode"] in CLIPPED_MODES:
        found.append(("clipped_mode", rec["mode"]))
    if rec["frames"] > 1:
        found.append(("multi_frame", f"{rec['frames']} frames"))
    if min(w, h) < MIN_SIDE:
        found.append(("tiny_image", f"{w}x{h}"))
    return rec, found


def load_manifest(path, sizes):
    """Rows of an interrupted run worth keeping (OK/WARN, file unchanged since). Rewrites the file
    without the rest, so FAILs and changed files are checked again."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") in ("OK", WARN)
                and len(r.get("sha256") or "") == 64 and str(sizes.get(r["path"])) == r["bytes"]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def check_contents(processed, todo, workers, manifest_path, file_level):
    """Content-check todo [(folder, name, bytes)] in worker processes, appending each result to the
    manifest as it lands. Returns the new rows."""
    if not todo:
        return []
    total_bytes = sum(size for *_, size in todo) or 1
    rows, tally, done_bytes = [], Counter(), 0
    t0 = last = time.monotonic()
    queue, pending = iter(todo), {}
    new_file = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh, \
            ProcessPoolExecutor(workers, initializer=_quiet_worker) as ex:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new_file:
            w.writeheader()
        try:
            while True:
                for folder, name, size in islice(queue, workers * 4 - len(pending)):
                    pending[ex.submit(check_content, processed / folder / name)] = (folder, name, size)
                if not pending:
                    break
                # a timeout keeps Ctrl+C responsive on Windows, where an untimed wait ignores it
                finished, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    folder, name, size = pending.pop(fut)
                    path = f"{folder}/{name}"
                    rec, found = fut.result()
                    severities = [CHECKS[c][0] for c, _ in found] + file_level.get(path, [])
                    status = FAIL if FAIL in severities else WARN if WARN in severities else "OK"
                    row = {"path": path, **rec, "status": status, "issues": " | ".join(f"{c}: {d}" for c, d in found)}
                    w.writerow(row)
                    rows.append(row)
                    tally[status] += 1
                    done_bytes += size
                    if found and found[0][0] == "read_error" and not processed.is_dir():
                        raise SystemExit(f"\n{processed} has disappeared (drive disconnected?). "
                                         "Reconnect it and rerun with --resume.")
                now = time.monotonic()
                if now - last >= 30 or not pending:
                    elapsed, last = max(now - t0, 1e-3), now
                    rate = done_bytes / elapsed
                    eta = (total_bytes - done_bytes) / rate if rate else 0
                    print(f"  {len(rows):,}/{len(todo):,} files, {100 * done_bytes / total_bytes:.1f}% of "
                          f"{total_bytes / 1e9:,.0f} GB | {rate / 1e6:.0f} MB/s, {len(rows) / elapsed:.0f} files/s"
                          f" | ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m | fail {tally[FAIL]}, warn {tally[WARN]}",
                          flush=True)
                    fh.flush()
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            fh.flush()
            raise
    return rows


# --------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser(description="EC-HTR PROCESSED_DATA integrity check (read-only)")
    ap.add_argument("processed_dir", type=Path, help=r"e.g. E:\PROCESSED_DATA")
    ap.add_argument("--raw-dir", type=Path, help="Default: RAW_DATA next to processed_dir")
    ap.add_argument("--stats", type=Path, help="Default: processed_dir/organization_stats.json")
    ap.add_argument("--out-dir", type=Path, default=Path("integrity_report"))
    ap.add_argument("--scope", choices=["organized", "all"], default="organized",
                    help="Content-check Organized only (the H200 set) or Irrelevant_Post1939 too (+1.6 TB)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent readers/decoders. On the USB HDD, 16 read small scans ~1.7x faster than 8; "
                         "large ones saturate the drive (~155 MB/s) either way")
    ap.add_argument("--resume", action="store_true", help="Continue an interrupted run in --out-dir")
    ap.add_argument("--limit", type=int, help="Content-check only the first N files per folder and byte-compare "
                                              "only N RAW_DATA leftovers (smoke test)")
    args = ap.parse_args()

    processed = args.processed_dir.resolve()
    raw = (args.raw_dir or processed.parent / "RAW_DATA").resolve()
    stats_path = args.stats or processed / "organization_stats.json"
    out = args.out_dir.resolve()
    manifest_path = out / "manifest.csv"
    if not processed.is_dir():
        raise SystemExit(f"Not found: {processed}")
    if not stats_path.is_file():
        raise SystemExit(f"Organizer stats not found: {stats_path} (pass --stats)")
    if out == processed or processed in out.parents:
        raise SystemExit("--out-dir must be outside PROCESSED_DATA")
    if manifest_path.exists() and not args.resume:
        raise SystemExit(f"{manifest_path} exists: pass --resume to continue that run, or choose another --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    print(f"Indexing {processed} ...", flush=True)
    files, issues = index_processed(processed)
    print(f"  {len(files):,} files, {sum(s for *_, s in files) / 1e9:,.1f} GB")
    for folder, name, _ in files:
        issues += [(f"{folder}/{name}", c, d) for c, d in name_issues(folder, name)]
    issues += cross_name_issues(files)
    file_level = defaultdict(list)   # severities of each file's own name checks, for its manifest status
    for path, check, _ in issues:
        file_level[path].append(CHECKS[check][0])

    print(f"Reconciling counts with {stats_path.name} and {raw} ...", flush=True)
    table, leftover_rows = reconcile(files, processed, stats, raw, args.limit, issues)
    with open(out / "raw_leftovers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEFTOVER_FIELDS)
        w.writeheader()
        w.writerows(leftover_rows)
    print_counts(table, raw.is_dir())

    targets, per_folder = [], Counter()
    for f in files:
        if (f[0].startswith(ORGANIZED + "/") or args.scope == "all") and per_folder[f[0]] < (args.limit or len(files)):
            per_folder[f[0]] += 1
            targets.append(f)
    sizes = {f"{folder}/{name}": size for folder, name, size in files}
    kept = load_manifest(manifest_path, sizes) if args.resume else []
    done = {r["path"] for r in kept}
    todo = [f for f in targets if f"{f[0]}/{f[1]}" not in done]
    print(f"Checking contents of {len(todo):,} files with {args.workers} workers"
          + (f" ({len(kept):,} already done)" if kept else "") + " ...", flush=True)
    try:
        rows = kept + check_contents(processed, todo, args.workers, manifest_path, file_level)
    except KeyboardInterrupt:
        raise SystemExit(f"\nInterrupted. Progress is saved in {manifest_path}; rerun with --resume to continue.")

    for r in rows:
        for part in filter(None, (r["issues"] or "").split(" | ")):
            check, _, detail = part.partition(": ")
            issues.append((r["path"], check, detail))
    by_hash = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_hash[r["sha256"]].append(r["path"])
    for paths in by_hash.values():
        for p in paths if len(paths) > 1 else ():
            issues.append((p, "duplicate_content", "same bytes as " + ", ".join(q for q in paths if q != p)))

    issues.sort(key=lambda i: (SEVERITY_ORDER[CHECKS[i[1]][0]], i[1], i[0]))
    with open(out / "issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["severity", "check", "path", "detail"])
        w.writerows((CHECKS[c][0], c, p, d) for p, c, d in issues)

    per_check = Counter(c for _, c, _ in issues)
    formats = Counter((os.path.splitext(r["path"])[1].lower(), r["format"]) for r in rows if r["format"])
    modes = Counter(r["mode"] for r in rows if r["mode"])
    mp = sorted(int(r["width"]) * int(r["height"]) / 1e6 for r in rows if r["width"])
    organized_rows = [r for r in rows if r["path"].startswith(ORGANIZED + "/")]
    blocked = sum(r["status"] == FAIL for r in organized_rows)
    failed = [c for c in CHECKS if per_check[c] and CHECKS[c][0] == FAIL]
    complete = not args.limit and len(organized_rows) == sum(f[0].startswith(ORGANIZED + "/") for f in files)

    print(f"\nContents of {len(rows):,} files ({sum(int(r['bytes'] or 0) for r in rows) / 1e9:,.1f} GB)")
    for ext in sorted({e for e, _ in formats}):
        print(f"  {ext} files hold: " + ", ".join(f"{fmt} {n:,}" for (e, fmt), n in formats.most_common() if e == ext))
    print("  modes: " + ", ".join(f"{m} {n:,}" for m, n in modes.most_common()))
    if mp:
        print(f"  megapixels: min {mp[0]:.1f}, median {statistics.median(mp):.1f}, max {mp[-1]:.1f}")
    print("\nChecks")
    for c, (severity, meaning) in CHECKS.items():
        print(f"  {severity if per_check[c] else 'ok':<5} {c:<18}{per_check[c]:>8,}  {meaning}")

    result = "FAIL" if failed else "PASS"
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"result": result, "complete": complete, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "processed_dir": str(processed), "raw_dir": str(raw), "scope": args.scope, "limit": args.limit,
                   "counts": {k: dict(zip(("organizer", "skipped", "expected", "found"), v)) for k, v in table.items()},
                   "organized_files_checked": len(organized_rows), "organized_files_blocked": blocked,
                   "formats_by_extension": {f"{e} {fmt}": n for (e, fmt), n in formats.items()},
                   "modes": dict(modes), "checks": {c: {"severity": CHECKS[c][0], "count": per_check[c],
                                                       "meaning": CHECKS[c][1]} for c in CHECKS}}, f, indent=2)

    print(f"\nRESULT: {result}" + ("" if complete else " (partial run: --limit set or not all of Organized checked)"))
    print(f"  {len(organized_rows) - blocked:,} of {len(organized_rows):,} checked Organized files are ready for the H200"
          + (f" ({blocked:,} blocked)" if blocked else ""))
    if failed:
        print(f"  failing checks: {', '.join(failed)}")
    print(f"  details: {out / 'issues.csv'}, manifest: {manifest_path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
