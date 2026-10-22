#!/usr/bin/env python3
"""
Excel deliverable from a production JSONL: one row per scan, taken from its latest JSONL row.

Only reads the JSONL, so it can run during production.py too. Rows are sorted by source_path.
Failed files are kept with their status and error (and no output), so none goes missing silently.
Needs openpyxl (pip install openpyxl).

  python Benchmark/export_xlsx.py runs/main.jsonl                  # writes runs/main.xlsx
  python Benchmark/export_xlsx.py runs/main.jsonl runs/extras.jsonl --xlsx runs/all.xlsx
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.utils import get_column_letter

from production import FIELDS, read_jsonl

MAX_CELL = 32_767   # Excel's limit on characters in a cell


def cell(ws, key, v):
    if isinstance(v, list):                       # native_wh, sent_wh
        return " x ".join(map(str, v))
    if key in ("year", "month") and isinstance(v, str) and v.isdigit():
        return int(v)
    if isinstance(v, str):
        v = ILLEGAL_CHARACTERS_RE.sub("", v)[:MAX_CELL]
        if v.startswith("="):                     # a ditto mark is text, not a formula
            c = WriteOnlyCell(ws, v)
            c.data_type = "s"
            return c
    return v


def main():
    ap = argparse.ArgumentParser(description="Production JSONL -> Excel (latest row per source_path)")
    ap.add_argument("jsonl", type=Path, nargs="+", help="One or more production JSONL files")
    ap.add_argument("--xlsx", type=Path, help="Default: the first JSONL with .xlsx")
    args = ap.parse_args()
    xlsx = args.xlsx or args.jsonl[0].with_suffix(".xlsx")

    # source_path -> (file, byte offset) of its latest row; offsets keep memory small on 570k rows
    latest = {}
    for i, path in enumerate(args.jsonl):
        for offset, rec in read_jsonl(path):
            latest[rec["source_path"]] = (i, offset)

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Results")
    ws.freeze_panes = "A2"
    for i, key in enumerate(FIELDS, 1):
        ws.column_dimensions[get_column_letter(i)].width = {"source_path": 60, "output": 80, "error": 40}.get(key, 14)
    ws.append(FIELDS)
    status, remarks, length = Counter(), Counter(), 0
    files = [open(p, "rb") for p in args.jsonl]
    try:
        for sp in sorted(latest):
            i, offset = latest[sp]
            files[i].seek(offset)
            rec = json.loads(files[i].readline())
            status[rec["status"]] += 1
            if rec["status"] == "ok":
                remarks["with remarks" if rec["has_remarks"] else "NO_REMARKS"] += 1
                length += rec["finish_reason"] == "length"
            ws.append([cell(ws, k, rec.get(k)) for k in FIELDS])
    finally:
        for f in files:
            f.close()
    ws.auto_filter.ref = f"A1:{get_column_letter(len(FIELDS))}{len(latest) + 1}"

    s = wb.create_sheet("Summary")
    s.column_dimensions["A"].width = 30
    s.column_dimensions["B"].width = 60
    s.append(["files", len(latest)])
    for k, v in sorted(status.items()):
        s.append([f"status {k}", v])
    for k, v in sorted(remarks.items()):
        s.append([f"ok, {k}", v])
    s.append(["ok, finish_reason=length", length])
    s.append(["source", ", ".join(str(p) for p in args.jsonl)])
    s.append(["exported (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")])
    wb.save(xlsx)

    print(f"{len(latest):,} files -> {xlsx}")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in sorted(status.items())))
    not_done = sum(v for k, v in status.items() if k not in ("ok", "decode_error"))
    if not_done:
        print(f"WARNING: {not_done:,} files have no result yet (status other than ok/decode_error). "
              "Rerun production.py with the same arguments, then export again.")


if __name__ == "__main__":
    main()
