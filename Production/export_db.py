#!/usr/bin/env python3
"""
Parquet and SQLite datasets from production JSONLs, for analysis. Read-only on its inputs.

  remarks   one row per source_path from its latest JSONL row, sorted by source_path: the rows of
            export_xlsx.py's workbook, with the same added columns (province_normalized, run,
            scans_of_form, alternate_scan). native_wh and sent_wh are split into native_w, native_h,
            sent_w, sent_h. output and error are the raw JSONL text (no Excel stripping or cell cap).
  attempts  every readable JSONL row, superseded transient_error rows included, with the same columns
            plus line_no (1-based) and byte_offset. The only record of which files timed out.
  provenance  each input's path, size, SHA-256, line and readable row counts; export time; git commit.

climate_id is text (some IDs have letters). has_remarks is null for non-ok rows. utc is a UTC timestamp
in Parquet and the JSONL's ISO 8601 text in SQLite. The SQLite file also has remarks_fts, an FTS5 table
over output with the porter tokenizer, so a word matches its plural too:
  SELECT source_path FROM remarks_fts WHERE remarks_fts MATCH 'aurora'
Every file is written under a temporary name and renamed once all of them are complete.
Needs pyarrow (pip install pyarrow).

  python Production/export_db.py runs/main.jsonl runs/extras.jsonl --out-dir runs
"""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from export_xlsx import EXTRA, PROVINCE
from production import FIELDS, read_jsonl

BATCH = 20_000
COLUMNS = [c for k in FIELDS for c in ({"native_wh": ["native_w", "native_h"],
                                         "sent_wh": ["sent_w", "sent_h"]}.get(k, [k]))] + EXTRA
ATTEMPT_COLUMNS = COLUMNS + ["line_no", "byte_offset"]
ARROW = {"year": pa.int32(), "month": pa.int32(), "native_w": pa.int32(), "native_h": pa.int32(),
         "sent_w": pa.int32(), "sent_h": pa.int32(), "payload_bytes": pa.int64(), "prompt_tokens": pa.int64(),
         "output_tokens": pa.int64(), "has_remarks": pa.bool_(), "utc": pa.timestamp("s", tz="UTC"),
         "scans_of_form": pa.int32(), "alternate_scan": pa.bool_(), "line_no": pa.int64(),
         "byte_offset": pa.int64()}
INDEXES = {"remarks": [["filename"], ["climate_id"], ["province_normalized"], ["year", "month"],
                       ["has_remarks"], ["finish_reason"]],
           "attempts": [["source_path"]]}


def sql_type(c):
    return "TEXT" if ARROW.get(c, pa.string()) in (pa.string(), ARROW["utc"]) else "INTEGER"


class Sink:
    """Writes the same rows to a Parquet file and a SQLite table, in batches."""

    def __init__(self, con, table, columns, parquet):
        self.con, self.columns, self.rows, self.count = con, columns, [], 0
        self.schema = pa.schema([(c, ARROW.get(c, pa.string())) for c in columns])
        self.writer = pq.ParquetWriter(parquet, self.schema, compression="zstd")
        con.execute(f"CREATE TABLE {table} ({', '.join(f'{c} {sql_type(c)}' for c in columns)})")
        self.insert = f"INSERT INTO {table} VALUES ({', '.join('?' * len(columns))})"

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= BATCH:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        cols = {c: [r[c] for r in self.rows] for c in self.columns}
        cols["utc"] = [datetime.fromisoformat(v) if v else None for v in cols["utc"]]
        self.writer.write_table(pa.Table.from_pydict(cols, schema=self.schema))
        self.con.executemany(self.insert, ([r[c] for c in self.columns] for r in self.rows))
        self.count += len(self.rows)
        self.rows = []

    def close(self):
        self.flush()
        self.writer.close()


def row(rec, run, alternate, scans):
    """A JSONL record as a remarks row: the same values and added columns as export_xlsx.py."""
    if rec["province"] not in PROVINCE:
        raise ValueError(f"{rec['source_path']}: unmapped province {rec['province']!r}; add it to PROVINCE")
    r = {k: rec.get(k) for k in FIELDS}
    for k in ("native", "sent"):
        r[f"{k}_w"], r[f"{k}_h"] = r.pop(f"{k}_wh") or (None, None)
    r["year"], r["month"] = int(r["year"]), int(r["month"])
    if r["status"] != "ok":
        r["has_remarks"] = None
    r.update(province_normalized=PROVINCE[rec["province"]], run=run, scans_of_form=scans[rec["filename"]],
             alternate_scan=alternate)
    return r


def scan(path):
    """Size, SHA-256, line count and byte offset -> line number, split into lines the way read_jsonl does."""
    sha, line_no, offset = hashlib.sha256(), {}, 0
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            sha.update(line)
            line_no[offset] = n
            offset += len(line)
    return {"path": str(path.resolve()), "size": offset, "sha256": sha.hexdigest(), "lines": len(line_no)}, line_no


def script_commit():
    """HEAD of this script's repo, and whether the script differs from it (None, None outside git)."""
    here = Path(__file__).resolve()
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(here.parent), *a], capture_output=True, text=True,
                                         check=True).stdout.strip()
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", here.name))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Production JSONLs -> Parquet + SQLite (remarks, attempts, provenance)")
    ap.add_argument("jsonl", type=Path, nargs="+", help="Production JSONL files, main.jsonl first")
    ap.add_argument("--out-dir", type=Path, help="Default: the first JSONL's folder")
    args = ap.parse_args()
    out = args.out_dir or args.jsonl[0].parent
    final = {k: out / n for k, n in [("remarks", "ec_remarks.parquet"), ("attempts", "ec_attempts.parquet"),
                                     ("sqlite", "ec_remarks.sqlite"), ("provenance", "ec_remarks.provenance.json")]}
    tmp = {k: p.with_name(f".{p.name}.tmp") for k, p in final.items()}
    runs = [p.stem for p in args.jsonl]

    inputs, line_nos = zip(*(scan(p) for p in args.jsonl))
    # source_path -> (file, byte offset, filename) of its latest row, as in export_xlsx.py
    latest = {}
    for i, path in enumerate(args.jsonl):
        readable = 0
        for offset, rec in read_jsonl(path):
            latest[rec["source_path"]] = (i, offset, rec["filename"])
            readable += 1
        inputs[i].update(readable_rows=readable, unreadable_lines=inputs[i]["lines"] - readable)
    scans = Counter(fn for _, _, fn in latest.values())

    for p in tmp.values():
        p.unlink(missing_ok=True)
    try:
        con = sqlite3.connect(tmp["sqlite"])
        con.execute("PRAGMA journal_mode = OFF")   # a crash leaves only the temp file, which is discarded
        con.execute("PRAGMA synchronous = OFF")

        attempts = Sink(con, "attempts", ATTEMPT_COLUMNS, tmp["attempts"])
        for i, path in enumerate(args.jsonl):
            for offset, rec in read_jsonl(path):
                attempts.add(row(rec, runs[i], i > 0, scans) | {"line_no": line_nos[i][offset], "byte_offset": offset})
        attempts.close()

        remarks, status = Sink(con, "remarks", COLUMNS, tmp["remarks"]), Counter()
        files = [open(p, "rb") for p in args.jsonl]
        try:
            for sp in sorted(latest):
                i, offset, _ = latest[sp]
                files[i].seek(offset)
                rec = json.loads(files[i].readline())
                status[rec["status"]] += 1
                remarks.add(row(rec, runs[i], i > 0, scans))
        finally:
            for f in files:
                f.close()
        remarks.close()

        for p, info in zip(args.jsonl, inputs):
            if p.stat().st_size != info["size"]:
                raise RuntimeError(f"{p} changed during the export; rerun once it is no longer being written")

        con.execute("CREATE UNIQUE INDEX remarks_source_path ON remarks (source_path)")
        for table, indexes in INDEXES.items():
            for cols in indexes:
                con.execute(f"CREATE INDEX {table}_{'_'.join(cols)} ON {table} ({', '.join(cols)})")
        try:
            con.execute("CREATE VIRTUAL TABLE remarks_fts USING fts5(source_path UNINDEXED, output, "
                        "tokenize = 'porter unicode61')")
        except sqlite3.OperationalError as e:
            fts = False
            print(f"WARNING: no FTS5 in this sqlite3 ({e}); remarks_fts not built")
        else:
            fts = True
            con.execute("INSERT INTO remarks_fts SELECT source_path, output FROM remarks WHERE output IS NOT NULL")

        commit, modified = script_commit()
        meta = {"exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script": Path(__file__).name, "script_commit": commit, "script_modified": modified,
                "inputs": list(inputs), "rows": {"remarks": remarks.count, "attempts": attempts.count},
                "remarks_fts": fts}
        con.execute("CREATE TABLE provenance (path TEXT, size INTEGER, sha256 TEXT, lines INTEGER, "
                    "readable_rows INTEGER, unreadable_lines INTEGER, exported_utc TEXT, script_commit TEXT, "
                    "script_modified INTEGER)")
        con.executemany("INSERT INTO provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(d["path"], d["size"], d["sha256"], d["lines"], d["readable_rows"], d["unreadable_lines"],
                          meta["exported_utc"], commit, modified) for d in inputs])
        con.commit()
        con.close()
        tmp["provenance"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for k in final:
            os.replace(tmp[k], final[k])
    except BaseException:
        for p in tmp.values():
            p.unlink(missing_ok=True)
        raise

    print(f"remarks {remarks.count:,} rows ({', '.join(f'{k} {v:,}' for k, v in sorted(status.items()))}), "
          f"attempts {attempts.count:,} rows -> {out}")
    for d in inputs:
        print(f"  {d['path']}: {d['lines']:,} lines, {d['readable_rows']:,} readable, "
              f"{d['unreadable_lines']:,} skipped as unreadable")
    if modified:
        print("WARNING: export_db.py has uncommitted changes; provenance records script_modified = true")


if __name__ == "__main__":
    main()
