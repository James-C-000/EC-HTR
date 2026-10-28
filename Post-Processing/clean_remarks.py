#!/usr/bin/env python3
"""
Conservative, non-destructive cleaning of the EC production-run remarks: only junk lines are removed.

Inputs (read-only), from --production-dir (default: Production/, beside this folder), as written by
Production/export_db.py from the same JSONLs:
  ec_remarks.parquet   one row per scan; `output` is the raw VLM transcription
  ec_remarks.sqlite    the same rows as the table remarks, plus attempts, provenance, remarks_fts

Outputs, in --out-dir (default: this folder):
  ec_remarks_clean.parquet  every row and column of ec_remarks.parquet unchanged (utc is written at
                            microsecond precision; its values are unchanged), plus the cleaning columns
  ec_remarks_clean.sqlite   a copy of ec_remarks.sqlite, every table unchanged, with the cleaning
                            columns added to remarks (the `lines` column becomes the table remark_lines),
                            plus remarks_clean_fts (FTS5, porter tokenizer, over remarks_clean) and
                            clean_provenance (each input's size and SHA-256, the time, the git commit)

Cleaning columns:
  remarks_clean       kept lines, original order, one per distinct line; in forms that finished
                      normally, a line that occurred N>1 times ends with " [×N]" (not shown for
                      token-capped forms, where repeats are loop artifacts). Empty when nothing was
                      kept; null for failed files
  has_remarks_clean   True if any line was kept; null for failed files (has_remarks null), so a
                      failed file never reads as having no remarks
  n_lines_raw         non-empty lines in `output` of has_remarks rows (0 for NO_REMARKS rows)
  n_lines_clean       lines in remarks_clean
  n_removed_<rule>    distinct lines removed by each rule (see below)
  lines               every distinct line of the form, kept AND removed, in order, as a list of
                      (line_no, text, times_in_form, removed_by); removed_by null = kept. Empty for
                      NO_REMARKS rows, null for failed files. In SQLite: remark_lines (source_path,
                      line_no, text, times_in_form, removed_by), one row per line

Nothing is deleted: `output` is left exactly as produced, and every removed line is listed with its rule.

Lines are compared after normalisation (lowercase, accents stripped, punctuation -> space).
Only lines that are plainly not content are removed: marks with no letters or digits, empty entries
such as "None", and the cut-off fragment that ends a repetition loop. Every other line is kept,
whether or not it is in scope. Rules, applied to each distinct line of a form:
  repeat   : a line that recurs in the same form is kept once (first occurrence's text)
             and its count is shown as " [×N]". Not counted as a removal of content.
  symbols  : no letters or digits at all (stray marks, ditto marks, Vienna-congress weather symbols)
  null_entry: "No", "None", "Nil", "Nothing" (or "Remarks"), with at most numbers and single letters
             beside it ("No.", "None.", "No. 2")
  cut_off_repeat: in a token-capped output, the cut-off final line when it is the start of
             another kept line in the same form ("Qu" after "Quel Cool", "Storm" after "Stormy");
             its content is in that line. A line with no letters or digits never qualifies
Earlier versions (git bc59876 and before) also removed routine weather and statistics lines with
the rules plain_weather, reading and zero_count; they were dropped to keep every line that might
be in scope.

Both files are written under temporary names and renamed once both are complete.

Usage: python Post-Processing/clean_remarks.py [--production-dir DIR] [--out-dir DIR]
Needs: duckdb, pandas
"""
import argparse
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

def W(s):
    return frozenset(s.split())

# a line made only of these is an empty entry ("None", "Nil", "No")
NULLS = W("no none nil nothing remarks remark")

ORD = re.compile(r"^\d+(st|nd|rd|th|a|p|am|pm|h|hr|hrs|in)?$")


def rule_for(norm: str):
    """Return the removal rule for a normalised line, or None to keep it."""
    toks = norm.split()
    if not toks:
        return "symbols"
    # null_entry: "None", "Nil", "No", with at most numbers and single letters beside it
    alpha = [t for t in toks if not ORD.match(t)]
    if any(t in NULLS for t in alpha) and all(t in NULLS or len(t) < 2 for t in alpha):
        return "null_entry"
    return None


# cut_off_repeat is assigned in SQL: the last line of a token-capped output, cut mid-line,
# when it is the start of another line in the same form (the loop restarting)
RULES = ["symbols", "null_entry", "cut_off_repeat"]


def sql_path(p) -> str:
    return str(p).replace("'", "''")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


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
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Clean the production remarks (read-only on the inputs)")
    ap.add_argument("--production-dir", type=Path, default=here.parent / "Production",
                    help="Folder with ec_remarks.parquet and ec_remarks.sqlite (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=here, help="Default: %(default)s")
    args = ap.parse_args()
    src = {k: args.production_dir / k for k in ("ec_remarks.parquet", "ec_remarks.sqlite")}
    for p in src.values():
        if not p.is_file():
            sys.exit(f"missing input: {p}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = {k: args.out_dir / k for k in ("ec_remarks_clean.parquet", "ec_remarks_clean.sqlite")}
    tmp = {k: p.with_name(f".{p.name}.tmp") for k, p in final.items()}
    for p in tmp.values():
        p.unlink(missing_ok=True)
    try:
        inputs = [{"path": str(p.resolve()), "size": p.stat().st_size, "sha256": sha256(p)} for p in src.values()]
        con = clean(src["ec_remarks.parquet"])
        con.execute(f"""COPY (SELECT * EXCLUDE (_row) FROM out ORDER BY _row)
                        TO '{sql_path(tmp["ec_remarks_clean.parquet"])}' (FORMAT parquet, COMPRESSION zstd)""")
        write_sqlite(con, src["ec_remarks.sqlite"], tmp["ec_remarks_clean.sqlite"], inputs)
        for p, info in zip(src.values(), inputs):
            if p.stat().st_size != info["size"]:
                raise RuntimeError(f"{p} changed during the run; rerun once it is no longer being written")
        for k in final:
            os.replace(tmp[k], final[k])
    except BaseException:
        for p in tmp.values():
            p.unlink(missing_ok=True)
        raise
    print("wrote", final["ec_remarks_clean.parquet"], "and", final["ec_remarks_clean.sqlite"])


def clean(src: Path):
    """Run the rules on ec_remarks.parquet; returns a DuckDB connection with the tables out
    (the output rows, in input order by _row) and ulc (one row per distinct line per form)."""
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=true")
    con.execute("SET memory_limit='5GB'")
    # _row = the row's position in the input file, so the output keeps the input's order
    con.execute(f"""CREATE TABLE r AS SELECT file_row_number AS _row, * EXCLUDE (file_row_number)
                    FROM read_parquet('{sql_path(src)}', file_row_number = true)""")

    # explode outputs of has_remarks rows into lines; normalise; group repeats within a form
    con.execute(r"""
        CREATE TABLE ln AS
        WITH x AS (
            SELECT _row, unnest(string_split(output, chr(10))) AS line,
                   generate_subscripts(string_split(output, chr(10)), 1) AS line_no
            FROM r WHERE has_remarks AND output IS NOT NULL)
        SELECT _row, line_no, trim(line) AS line,
               trim(regexp_replace(regexp_replace(strip_accents(lower(trim(line))), '[^a-z0-9\s]', ' ', 'g'),
                                   '\s+', ' ', 'g')) AS norm
        FROM x WHERE trim(line) <> ''""")
    con.execute("""
        CREATE TABLE ul AS
        SELECT _row, norm, min(line_no) AS first_line_no, arg_min(line, line_no) AS text, count(*) AS times_in_form
        FROM ln GROUP BY _row, norm""")

    # classify each distinct normalised string once
    norms = [n for (n,) in con.execute("SELECT DISTINCT norm FROM ul").fetchall()]
    cls = pd.DataFrame({"norm": norms, "removed_by": [rule_for(n) for n in norms]})
    con.register("cls", cls)
    # truncated outputs: the final line is cut off by the token cap. If it occurs once and is the
    # beginning of another line in the same form that is itself kept, it is a partial repeat of
    # that line, whose full text stays in remarks_clean.
    con.execute("""
        CREATE TABLE cut AS
        WITH last AS (
            SELECT ln._row, arg_max(ln.norm, ln.line_no) AS norm
            FROM ln JOIN r USING (_row) WHERE r.finish_reason = 'length' GROUP BY ln._row)
        SELECT u._row, u.norm FROM ul u JOIN last USING (_row, norm)
        WHERE u.times_in_form = 1 AND u.norm <> '' AND EXISTS (
            SELECT 1 FROM ul v JOIN cls c ON c.norm = v.norm
            WHERE v._row = u._row AND v.norm <> u.norm AND starts_with(v.norm, u.norm) AND c.removed_by IS NULL)""")
    con.execute("""
        CREATE TABLE ulc AS
        SELECT ul.*, CASE WHEN cut._row IS NOT NULL THEN 'cut_off_repeat' ELSE cls.removed_by END AS removed_by
        FROM ul JOIN cls USING (norm) LEFT JOIN cut USING (_row, norm)""")

    # per-form aggregates
    rule_cols = ",\n".join(f"count(*) FILTER (WHERE removed_by = '{k}') AS n_removed_{k}" for k in RULES)
    con.execute(f"""
        CREATE TABLE agg AS
        SELECT ulc._row,
               string_agg(CASE WHEN ulc.times_in_form > 1 AND r.finish_reason = 'stop'
                               THEN ulc.text || ' [×' || ulc.times_in_form || ']' ELSE ulc.text END,
                          chr(10) ORDER BY ulc.first_line_no) FILTER (WHERE ulc.removed_by IS NULL) AS remarks_clean,
               count(*) FILTER (WHERE ulc.removed_by IS NULL) AS n_lines_clean,
               {rule_cols},
               list(struct_pack(line_no := ulc.first_line_no, text := ulc.text, times_in_form := ulc.times_in_form,
                                removed_by := ulc.removed_by) ORDER BY ulc.first_line_no) AS lines
        FROM ulc JOIN r USING (_row) GROUP BY ulc._row""")
    raw_counts = "SELECT _row, count(*) AS n_lines_raw FROM ln GROUP BY _row"
    zero = ", ".join(f"coalesce(a.n_removed_{k}, 0)::INTEGER AS n_removed_{k}" for k in RULES)
    con.execute(f"""
        CREATE TABLE out AS
        SELECT r.*,
               CASE WHEN r.has_remarks IS NOT NULL THEN coalesce(a.remarks_clean, '') END AS remarks_clean,
               CASE WHEN r.has_remarks IS NOT NULL THEN coalesce(a.n_lines_clean, 0) > 0 END AS has_remarks_clean,
               coalesce(c.n_lines_raw, 0)::INTEGER AS n_lines_raw,
               coalesce(a.n_lines_clean, 0)::INTEGER AS n_lines_clean,
               {zero},
               CASE WHEN r.has_remarks IS NOT NULL THEN coalesce(a.lines, []) END AS lines
        FROM r LEFT JOIN agg a USING (_row) LEFT JOIN ({raw_counts}) c USING (_row)
        ORDER BY r._row""")
    return con


CLEAN_COLUMNS = ["remarks_clean", "has_remarks_clean", "n_lines_raw", "n_lines_clean"] + [f"n_removed_{k}" for k in RULES]


def write_sqlite(con, src: Path, dst: Path, inputs: list):
    """Copy ec_remarks.sqlite (opened read-only) to dst and add the cleaning results to it."""
    ro = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True)
    db = sqlite3.connect(dst)
    try:
        ro.backup(db)
    finally:
        ro.close()
    try:
        db.execute("PRAGMA journal_mode = OFF")   # a crash leaves only the temp file, which is discarded
        db.execute("PRAGMA synchronous = OFF")
        # the two inputs must come from the same export
        a = db.execute("SELECT source_path, output FROM remarks ORDER BY source_path").fetchall()
        b = con.execute("SELECT source_path, output FROM out ORDER BY source_path").fetchall()
        if a != b:
            raise ValueError(f"{src.name} and ec_remarks.parquet differ in source_path or output; "
                             "use the two files written by the same export_db.py run")

        for col in CLEAN_COLUMNS:
            db.execute(f"ALTER TABLE remarks ADD COLUMN {col} {'TEXT' if col == 'remarks_clean' else 'INTEGER'}")
        db.executemany(f"UPDATE remarks SET {', '.join(f'{c} = ?' for c in CLEAN_COLUMNS)} WHERE source_path = ?",
                       con.execute(f"SELECT {', '.join(CLEAN_COLUMNS)}, source_path FROM out").fetchall())
        db.execute("CREATE INDEX remarks_has_remarks_clean ON remarks (has_remarks_clean)")

        db.execute("CREATE TABLE remark_lines (source_path TEXT, line_no INTEGER, text TEXT, times_in_form INTEGER, "
                   "removed_by TEXT)")
        db.executemany("INSERT INTO remark_lines VALUES (?, ?, ?, ?, ?)", con.execute("""
            SELECT r.source_path, u.first_line_no, u.text, u.times_in_form, u.removed_by
            FROM ulc u JOIN r USING (_row) ORDER BY r._row, u.first_line_no""").fetchall())
        db.execute("CREATE INDEX remark_lines_source_path ON remark_lines (source_path)")
        db.execute("CREATE INDEX remark_lines_removed_by ON remark_lines (removed_by)")

        db.execute("CREATE VIRTUAL TABLE remarks_clean_fts USING fts5(source_path UNINDEXED, remarks_clean, "
                   "tokenize = 'porter unicode61')")
        db.execute("INSERT INTO remarks_clean_fts SELECT source_path, remarks_clean FROM remarks WHERE remarks_clean <> ''")

        commit, modified = script_commit()
        cleaned = datetime.now(timezone.utc).isoformat(timespec="seconds")
        db.execute("CREATE TABLE clean_provenance (path TEXT, size INTEGER, sha256 TEXT, cleaned_utc TEXT, "
                   "script_commit TEXT, script_modified INTEGER)")
        db.executemany("INSERT INTO clean_provenance VALUES (?, ?, ?, ?, ?, ?)",
                       [(d["path"], d["size"], d["sha256"], cleaned, commit, modified) for d in inputs])
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    main()
