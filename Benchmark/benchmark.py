#!/usr/bin/env python3
"""
EC Weather Remarks - VLM benchmark v9 (single H200).

One invocation = one model, all of its resolution rungs, one model load.

Fixes from v8 (details in the v8 audit):
  1. Mis-read positives are no longer dropped from recall/precision. Scoring is split into
     (a) file-level detection (sensitivity/specificity with Wilson 95% CIs) and
     (b) word-level transcription accuracy on files that have remarks. Word alignment is
     order-preserving and one-to-one, ignores line breaks, and over-extraction lowers precision.
  2. "NO_REMARKS" is recognised with surrounding punctuation/whitespace, not as an exact string.
  3. Class-mixing accuracy% removed.
  4. Resize by pixel budget only (no aspect-ratio-dependent max_dim cap), sides floored to
     multiples of 32 px, never upscaled. Sent size and prompt token count are logged per file.
  5. Greedy decoding with a fixed seed (v8: temperature 0.1, no seed).
  6. Requests are submitted in batches and GPU wall-clock is timed, so cost comes from
     throughput (v8 timed one request at a time).
  7. finish_reason logged, so truncated/looping outputs are visible.
  8. Station lookup accepts the same header variants as the organizer script (2022 first).

New models: Qwen3.8-27B (thinking disabled) and a Chandra OCR 2 recall probe.
"""

import argparse
import csv
import difflib
import html
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

CORPUS_FILES = 571_712  # pre-1940 files in the production corpus
MIB = 1024 * 1024       # rungs are in MiP: 2 MiP = 2,097,152 px, as in February
GRID = 32               # patch 16 x spatial merge 2 for Qwen3-VL, Qwen3.8 and Chandra 2

EXTRACT, OCR_PROBE = "extract", "ocr_probe"
MODELS = {
    "qwen3vl-8b":  {"id": "Qwen/Qwen3-VL-8B-Instruct",  "rungs": [2, 4, 8, 16]},
    "qwen3vl-32b": {"id": "Qwen/Qwen3-VL-32B-Instruct", "rungs": [2, 4, 8, 16]},
    "qwen3.8-27b": {"id": "Qwen/Qwen3.8-27B",           "rungs": [2, 4, 8, 16],
                    "chat_template_kwargs": {"enable_thinking": False}},
    # Chandra is trained only on fixed full-page OCR prompts, so it runs in its own "ocr" mode
    # at its own default pixel budget (3072 x 2048 = 6 MiP) and is scored on word recall only:
    # a go/no-go on a future OCR-then-filter pipeline, not a like-for-like competitor.
    "chandra2":    {"id": "datalab-to/chandra-ocr-2",   "rungs": [6], "mode": OCR_PROBE,
                    "max_tokens": 12384},
}

# Chandra's "ocr" prompt, copied verbatim from datalab-to/chandra (chandra/prompts.py).
_TAGS = ["math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th", "div",
         "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span", "img", "hr",
         "tbody", "small", "caption", "strong", "thead", "big", "code", "chem"]
_ATTRS = ["class", "colspan", "rowspan", "display", "checked", "type", "border", "value", "style",
          "href", "alt", "align", "data-bbox", "data-label"]
CHANDRA_OCR_PROMPT = f"""OCR this image to HTML.

Only use these tags {_TAGS}, and these attributes {_ATTRS}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural."""


# --------------------------------------------------------------------------- inputs

def load_stations(*paths):
    """Climate ID -> (name, province). Earlier paths win (pass 2022 first, like the organizer)."""
    def first(row, *keys):
        return next((row[k] for k in keys if row.get(k)), "")
    stations = {}
    for path in paths:
        if not path.exists():
            print(f"WARNING: station inventory not found: {path}")
            continue
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cid = first(row, "climate_id", "Climate ID", "CLIMATE ID").strip()
                if cid and cid not in stations:
                    stations[cid] = (first(row, "Name", "Station Name (Current)", "Station Name", "NAME").strip(),
                                     first(row, "Province", "PROVINCE").strip())
    return stations


def build_prompt(filename, stations):
    """February prompt, unchanged, so model comparisons aren't confounded with prompt changes."""
    parts = Path(filename).stem.split("_")
    name, province = stations.get(parts[1] if len(parts) > 1 else "", ("Unknown Station", "Unknown Province"))
    year = parts[2] if len(parts) > 2 else "Unknown"
    month = parts[3] if len(parts) > 3 else "Unknown"
    return f"""You are analyzing a historical weather observation form from {name}, {province}, Canada from {year}-{month}.

This form is part of a research project studying how Canadians understood and communicated about weather and nature, as seen in what they drew attention to. Your task is to extract handwritten qualitative remarks — observations that go beyond simple one-word weather states to describe, characterize, or comment on conditions.

EXTRACT:
- Phenological observations (seasonal changes in plants, animals, ice, or water)
- Extreme or notable weather events and their effects
- Atmospheric phenomena beyond ordinary weather
- Subjective or experiential descriptions that reveal how the observer perceived conditions
- Remarks written in French

DO NOT extract:
- Single-word weather states (e.g., "cloudy", "fair", "clear")
- Time entries (e.g., "AM", "PM", "all day")
- Numbers, temperatures, wind speeds, precipitation amounts
- Printed text, column headers, form boilerplate, or administrative stamps

When in doubt about whether a remark is qualitative, extract it.

If there are no qualitative remarks, respond with exactly: NO_REMARKS
If there are remarks, transcribe them exactly as written, one per line. Preserve original spelling, grammar, and language. Do not add explanations, categories, or formatting."""


def load_resized(path, budget_px):
    """Scale to at most budget_px pixels, sides floored to multiples of GRID, never upscaled.
    The models' own processors leave such an image as-is, so sent size = what the model sees."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(1.0, math.sqrt(budget_px / (w * h)))
    nw, nh = max(GRID, int(w * s) // GRID * GRID), max(GRID, int(h * s) // GRID * GRID)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img, (w, h)


# --------------------------------------------------------------------------- scoring

NONE_RE = re.compile(r"\W*no[\s_-]*remarks\W*", re.IGNORECASE)


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def says_none(text):
    return not text.strip() or NONE_RE.fullmatch(text.strip()) is not None


def words(text):
    """Lower-cased letter-only tokens, accents folded, HTML stripped. Digits and punctuation
    are ignored: numbers aren't target content and GT uses day-number prefixes ("7: brume")."""
    text = html.unescape(re.sub(r"<[^>]*>", " ", text)).lower()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.findall(r"[^\W\d_]+", text)


def matched_words(gt, out):
    """Words aligned in order, each used at most once (line breaks don't matter)."""
    blocks = difflib.SequenceMatcher(None, gt, out, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks)


def score(output, ground_truth, mode):
    gt_pos, out_pos = not says_none(ground_truth), not says_none(output)
    r = {"gt_has_remarks": gt_pos, "model_has_remarks": out_pos}
    if mode == EXTRACT:
        r["detection"] = {(True, True): "TP", (True, False): "FN",
                          (False, True): "FP", (False, False): "TN"}[(gt_pos, out_pos)]
    if gt_pos:
        g, o = words(ground_truth), (words(output) if out_pos else [])
        m = matched_words(g, o)
        r["word_recall"] = m / len(g) if g else None
        r["word_precision"] = m / len(o) if o else None   # undefined when nothing was output
        r["word_f1"] = 2 * m / (len(g) + len(o)) if g else None   # 0 for a miss
    return r


SUMMARY_FIELDS = ["config", "model_id", "mode", "budget_px", "files", "positives",
                  "sensitivity", "sensitivity_ci", "specificity", "specificity_ci", "TP", "FN", "FP", "TN",
                  "word_recall", "word_precision", "word_f1", "truncated",
                  "sent_mp_median", "sent_mp_min", "sent_mp_max", "prompt_tokens_median",
                  "gpu_seconds", "sec_per_file", "corpus_days", "corpus_usd", "usd_per_hour", "calls",
                  "vllm", "utc"]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)]


def summarize(recs, mode, gpu_seconds, usd_per_hour):
    pos = [r for r in recs if r["gt_has_remarks"]]

    def mean(key):
        vals = [r[key] for r in pos if r.get(key) is not None]
        return round(100 * statistics.mean(vals), 1) if vals else None

    s = {"files": len(recs), "positives": len(pos), "word_recall": mean("word_recall"),
         "word_precision": mean("word_precision") if mode == EXTRACT else None,
         "word_f1": mean("word_f1") if mode == EXTRACT else None}
    if mode == EXTRACT:
        c = Counter(r["detection"] for r in recs)
        tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
        s.update(TP=tp, FN=fn, FP=fp, TN=tn,
                 sensitivity=round(100 * tp / (tp + fn), 1) if tp + fn else None,
                 sensitivity_ci=wilson(tp, tp + fn),
                 specificity=round(100 * tn / (tn + fp), 1) if tn + fp else None,
                 specificity_ci=wilson(tn, tn + fp))
    sec_per_file = gpu_seconds / len(recs)
    mp = [r["sent_wh"][0] * r["sent_wh"][1] / 1e6 for r in recs]
    s.update(truncated=sum(r["finish_reason"] == "length" for r in recs),
             sent_mp_median=round(statistics.median(mp), 2),
             sent_mp_min=round(min(mp), 2), sent_mp_max=round(max(mp), 2),
             prompt_tokens_median=int(statistics.median(r["prompt_tokens"] for r in recs)),
             gpu_seconds=round(gpu_seconds, 1), sec_per_file=round(sec_per_file, 3),
             corpus_days=round(CORPUS_FILES * sec_per_file / 86400, 1),
             corpus_usd=round(CORPUS_FILES * sec_per_file / 3600 * usd_per_hour),
             usd_per_hour=usd_per_hour)
    return s


# --------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser(description="EC Weather Remarks VLM benchmark (v9)")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--rungs", type=float, nargs="+", help="Pixel budgets in MiP (default: the model's list)")
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--inventory-2014", type=Path, required=True)
    ap.add_argument("--inventory-2022", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--batch-megapixels", type=float, default=1000,
                    help="Total pixels (millions) per llm.chat call. Default puts all 150 files in one call at "
                         "<=6 MiP and splits 8/16 MiP into 2-3 calls. Lower it if host RAM runs out.")
    ap.add_argument("--usd-per-hour", type=float, default=3.50, help="Your actual H200 rate, for projections")
    ap.add_argument("--limit", type=int, help="Only the first N files (smoke test)")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    mode = cfg.get("mode", EXTRACT)
    with open(args.ground_truth, encoding="utf-8-sig") as f:
        gt = {r["filename"]: r["verified_extraction"] for r in csv.DictReader(f)}
    files = sorted(f for f in gt if (args.image_dir / f).exists())
    if len(files) < len(gt):
        print(f"WARNING: {len(gt) - len(files)} ground-truth files missing from {args.image_dir}")
    files = files[:args.limit] if args.limit else files
    if not files:
        raise SystemExit("No test images found.")
    stations = load_stations(args.inventory_2022, args.inventory_2014)
    unmatched = [f for f in files if f.split("_")[1] not in stations]
    print(f"{len(files)} files, {len(stations)} stations loaded, {len(unmatched)} files without a station match")

    from vllm import LLM, SamplingParams, __version__ as vllm_version  # late import so --help works anywhere

    llm = LLM(model=cfg["id"], dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1, "video": 0}, mm_processor_cache_gb=0, seed=0)
    # 512 tokens is >2.5x the longest ground truth; a greedy repetition loop is cut off there
    # instead of at 1024, which halves the time a looping request holds up the end of a batch.
    params = SamplingParams(temperature=0.0, max_tokens=cfg.get("max_tokens", 512), seed=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "summary.csv"

    for rung in args.rungs or cfg["rungs"]:
        tag = f"{args.model}_{rung:g}MiP"
        budget = int(rung * MIB)
        print(f"\n=== {tag} ({cfg['id']}, budget {budget:,} px, mode {mode}) ===")
        recs, gpu_seconds = [], 0.0
        # Each llm.chat call ends only when its slowest request does, so fewer, larger calls keep
        # one looping output from idling the GPU several times per rung.
        per_call = max(1, int(args.batch_megapixels * 1e6 // budget))
        for i in range(0, len(files), per_call):
            chunk, convs, meta = files[i:i + per_call], [], []
            for fn in chunk:  # resizing happens before the timer: production pre-resizes locally
                img, native = load_resized(args.image_dir / fn, budget)
                prompt = CHANDRA_OCR_PROMPT if mode == OCR_PROBE else build_prompt(fn, stations)
                convs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": img},
                                                           {"type": "text", "text": prompt}]}])
                meta.append((fn, native, img.size))
            t0 = time.perf_counter()
            outs = llm.chat(convs, params, chat_template_kwargs=cfg.get("chat_template_kwargs"))
            gpu_seconds += time.perf_counter() - t0
            for (fn, native, sent), out in zip(meta, outs):
                comp = out.outputs[0]
                text = clean(comp.text)
                recs.append({"filename": fn, "native_wh": list(native), "sent_wh": list(sent),
                             "prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(comp.token_ids),
                             "finish_reason": comp.finish_reason, "output": text,
                             "ground_truth": gt[fn], **score(text, gt[fn], mode)})

        with open(args.output_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = {"config": tag, "model_id": cfg["id"], "mode": mode, "budget_px": budget,
             **summarize(recs, mode, gpu_seconds, args.usd_per_hour),
             "calls": math.ceil(len(files) / per_call),
             "vllm": vllm_version, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        new_file = not summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="")
            if new_file:
                w.writeheader()
            w.writerow(s)
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
