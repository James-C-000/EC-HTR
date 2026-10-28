#!/usr/bin/env python3
"""
Compare production.py's output on the 150 ground-truth scans with benchmark v9 (qwen3.8-27b, 4 MiP).

Per file, matched by filename:
  sent_wh        must match: the same pixels reached the model
  prompt_tokens  must match: the same prompt and image tokens reached the model
  output         should match on nearly every file; greedy vLLM output can still change with batch
                 composition. "detection" says whether both agree on remarks vs NO_REMARKS.
Exit code 0 when every v9 file has an ok production row whose sent_wh and prompt_tokens match.

  python Production/compare_v9.py runs/gt150.jsonl results/qwen3.8-27b_4MiP.jsonl --csv runs/compare_v9.csv
"""

import argparse
import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Benchmark"))   # v9's benchmark.py
from benchmark import says_none  # noqa: E402
from production import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Production run vs benchmark v9 on the ground-truth files")
    ap.add_argument("production", type=Path, help="production.py JSONL for the 150 ground-truth scans")
    ap.add_argument("v9", type=Path, help="v9's qwen3.8-27b_4MiP.jsonl")
    ap.add_argument("--csv", type=Path, help="Write the per-file comparison here")
    args = ap.parse_args()

    v9 = {r["filename"]: r for _, r in read_jsonl(args.v9)}
    latest = {r["source_path"]: r for _, r in read_jsonl(args.production)}
    prod = {}
    for r in latest.values():
        if r["filename"] in prod:
            print(f"WARNING: {r['filename']} appears under two paths; using {r['source_path']}")
        prod[r["filename"]] = r
    extra = sorted(set(prod) - set(v9))
    if extra:
        print(f"Ignoring {len(extra)} production files that are not in v9, e.g. {extra[0]}")

    rows = []
    for fn in sorted(v9):
        a, p = v9[fn], prod.get(fn)
        row = {"filename": fn, "status": p["status"] if p else "missing", "v9_sent_wh": a["sent_wh"],
               "v9_prompt_tokens": a["prompt_tokens"], "v9_output": a["output"]}
        if p and p["status"] == "ok":
            row.update(sent_wh=p["sent_wh"] == a["sent_wh"], prompt_tokens=p["prompt_tokens"] == a["prompt_tokens"],
                       output=p["output"] == a["output"],
                       detection=says_none(p["output"]) == says_none(a["output"]),
                       prod_sent_wh=p["sent_wh"], prod_prompt_tokens=p["prompt_tokens"], prod_output=p["output"])
        rows.append(row)

    def mark(r, key):
        return "-" if key not in r else "match" if r[key] else "DIFF"

    print(f"\n{'filename':<34} {'status':<16} {'sent_wh':<8} {'prompt_tokens':<14} {'output':<8} detection")
    for r in rows:
        print(f"{r['filename']:<34} {r['status']:<16} {mark(r, 'sent_wh'):<8} {mark(r, 'prompt_tokens'):<14} "
              f"{mark(r, 'output'):<8} {mark(r, 'detection')}")

    for r in rows:
        if r["status"] != "ok" or all(r[k] for k in ("sent_wh", "prompt_tokens", "output")):
            continue
        print(f"\n--- {r['filename']}")
        if not r["sent_wh"]:
            print(f"sent_wh: v9 {r['v9_sent_wh']}, production {r['prod_sent_wh']}")
        if not r["prompt_tokens"]:
            print(f"prompt_tokens: v9 {r['v9_prompt_tokens']}, production {r['prod_prompt_tokens']}")
        if not r["output"]:
            diff = difflib.unified_diff(r["v9_output"].splitlines(), r["prod_output"].splitlines(),
                                        "v9", "production", lineterm="", n=1)
            print("\n".join(list(diff)[2:]))

    ok = [r for r in rows if r["status"] == "ok"]
    n = len(rows)
    print(f"\n{len(ok)}/{n} v9 files have an ok production row")
    for key in ("sent_wh", "prompt_tokens", "output", "detection"):
        print(f"  {key:<14} {sum(r[key] for r in ok)}/{len(ok)} match")

    if args.csv:
        cols = ["filename", "status", "sent_wh", "prompt_tokens", "output", "detection", "v9_sent_wh", "prod_sent_wh",
                "v9_prompt_tokens", "prod_prompt_tokens", "v9_output", "prod_output"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"per-file comparison -> {args.csv}")

    exact = len(ok) == n and all(r["sent_wh"] and r["prompt_tokens"] for r in ok)
    print("PASS: sent_wh and prompt_tokens match on every file" if exact else
          "FAIL: a file is missing, failed, or differs in sent_wh or prompt_tokens")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
