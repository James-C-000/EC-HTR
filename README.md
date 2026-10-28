# EC Historical Weather Remarks Extraction Pipeline

A Python pipeline for extracting qualitative weather remarks from Environment Canada's (EC) pre-1940 historical weather observation forms using Vision Language Models (VLMs) running on rented cloud GPUs.

## Project Overview

### Goal

Extract handwritten qualitative remarks from pre-1940 weather observation forms, focusing on:
- Phenological observations (first robin in spring, ice break-up, harvest timing)
- Extreme weather events (storms, floods, fires, droughts)
- General weather descriptions that reveal how Canadians understood and communicated about weather and nature

### Dataset

The source dataset consists of approximately 1 million digitized scans (PNG/TIF) of monthly weather observation forms from Canadian (and some US) weather stations spanning the 1870s through the 1960s. The organizer counted 571,712 pre-1940 files. 567,917 of them are in the organized folders and were processed; the other 3,795 are second copies of organized files left in `RAW_DATA`, and the 2,901 of those that are real alternative scans were processed as well.

### Approach

The pipeline uses full-page VLM inference to extract qualitative remarks directly from scans. The VLM handles both the vision task (reading handwritten text) and the logic task (identifying which text constitutes qualitative remarks worth extracting). Inference runs on rented GPU hardware via [vast.ai](https://vast.ai), with images sent individually from a local machine to the remote server. This avoids uploading the full 6 TB dataset to the cloud and respects data handling requirements, as no permanent copy of the scans is stored remotely.

The production run used **Qwen3.8-27B at a 4-megapixel budget** on one rented H200, chosen with the v9 benchmark (Stage 2).

Earlier approaches that were explored and ultimately abandoned include region-of-interest cropping (too many form variants), extract-everything-then-filter post-processing (unnecessary with sufficiently powerful models), and local-only inference (hardware constraints made it infeasible at scale).

---

## Project Structure

```
EC-HTR/
├── README.md                          # This file
├── .gitignore                         # Keeps results, scans and inventories out of the repo
│
├── Organization/                      # Stage 1: file organization and drive audits
│   ├── ec_organizer_enhanced.py       # Sorts scans by province/territory and year
│   ├── ec_verify_organization.py      # Spot-checks file placement
│   ├── ec_audit_unmatched_ids.py      # Lists scans filed by the Climate ID first-digit fallback
│   ├── ec_check_integrity.py          # Pre-H200 check: counts, names, full decode of every scan
│   └── ec_leftover_audit.py           # Compares pre-1940 files left in RAW_DATA with their Organized namesakes
│
├── Benchmark/                         # Stage 2: VLM benchmark (v9)
│   ├── README.md                      # vast.ai setup and run guide
│   ├── benchmark.py                   # v9 VLM benchmark (runs on the rented H200)
│   └── ground_truth.csv               # Human-verified remarks for the 150 test scans
│
├── Production/                        # Stages 3-4: production run and exports
│   ├── production.py                  # Production run client (runs locally, talks to vLLM on the H200)
│   ├── tunnel.bat                     # SSH tunnel to the instance's vLLM server (fill in host and ports)
│   ├── gt150_paths.txt                # The 150 test scans on the drive, for the production check
│   ├── compare_v9.py                  # Checks production.py against v9 on those 150 scans
│   ├── extras_paths.txt               # The 2,901 alternative scans in RAW_DATA
│   ├── export_xlsx.py                 # Production JSONL -> Excel deliverable
│   └── export_db.py                   # Production JSONL -> Parquet + SQLite datasets
│
└── Post-Processing/                   # Stage 5: cleaning the remarks
    └── clean_remarks.py               # Conservative line-level cleaning -> ec_remarks_clean .parquet / .sqlite
```

**Not in the repo:**
- The scans. The 150 test scans are named in `ground_truth.csv` and `gt150_paths.txt`.
- The ECCC station inventories (2014 and 2022 editions), which are not redistributed here. The scripts expect them as `inventory_2014.csv` and `inventory_2022.csv`, and the production commands below read them from `EC_Inventories/`.
- Results: run JSONLs and logs, exports, cleaned datasets, and audit reports. `.gitignore` excludes them.

---

## Stage 1: Organization

The organization stage sorts the raw dataset of ~1 million scans into a structured directory hierarchy by province/territory, using each file's Climate ID to determine its geographic origin.

### How It Works

Each scan filename follows the convention `9904_CLIMATEID_YYYY_MM[_A].png`, where the 7-digit Climate ID encodes the province (first digit), climatological district (digits 2-3), and station (digits 4-7). The organizer cross-references each Climate ID against two EC station inventory CSVs (2022 first, then 2014) to resolve the province/territory, then moves the file into the corresponding directory under `PROCESSED_DATA/Organized`. Files dated after 1939 are separated into an "irrelevant" directory since they fall outside the project's scope.

Non-standard Climate IDs (e.g., `610ML02`, `706CFQ3`) are looked up directly in the inventory CSVs. Any files that cannot be resolved are flagged for manual review.

### Results

| Metric | Count |
|--------|-------|
| Total files found | 996,602 |
| Files organized | 988,852 |
| Pre-1940 files counted | 571,712 |
| Pre-1940 files in Organized | 567,917 |
| Post-1939 files (excluded) | 424,890 |
| Non-standard IDs resolved | 3,536 |
| Files needing manual QC | 0 |
| Errors | 0 |

The organizer counts a file before moving it and skips the move when the destination name already exists. That accounts for the 7,750 files left in `RAW_DATA`: 3,795 pre-1940 (hence 571,712 − 3,795 = 567,917) and 3,955 post-1939. See the leftover audit below.

**Pre-1940 files by province/territory (as counted by the organizer):**

| Province / Territory | Files |
|----------------------|-------|
| Ontario | 165,391 |
| British Columbia | 124,577 |
| Alberta | 66,435 |
| Saskatchewan | 59,467 |
| Quebec | 48,651 |
| Manitoba | 38,139 |
| Nova Scotia | 25,631 |
| New Brunswick | 16,292 |
| Newfoundland & Labrador | 12,140 |
| Northwest Territories | 5,713 |
| Prince Edward Island | 3,649 |
| Nunavut | 2,838 |
| Yukon | 2,789 |

### Usage

`ec_organizer_enhanced.py` and `ec_verify_organization.py` take no command-line arguments. Set the paths in the configuration block at the top of each script's `main()` (`SOURCE_DIR`, `OUTPUT_DIR`, `STATION_CSV_2014`, `STATION_CSV_2022`, and `DRY_RUN` for the organizer; `ORGANIZED_DIR`, the two CSVs and `SAMPLE_SIZE` for the verifier), then run:

```bash
python Organization/ec_organizer_enhanced.py      # writes organization_stats.csv/.json to OUTPUT_DIR and file_organization.log here
python Organization/ec_verify_organization.py     # writes verification_results.json here
```

### Province Fallback Audit

For a 7-digit Climate ID in neither inventory, the organizer falls back to a first-digit province mapping that is off by one (it uses 2 = Alberta ... 8 = Territories; EC uses 1 = BC, 2 = YT/NT/NU, 3 = AB, ...). `ec_audit_unmatched_ids.py` is a read-only check that lists every Organized scan whose Climate ID is in neither inventory, with the province the corrected scheme gives it.

```bash
python Organization/ec_audit_unmatched_ids.py "E:\PROCESSED_DATA" \
    --inventory-2014 EC_Inventories/inventory_2014.csv --inventory-2022 EC_Inventories/inventory_2022.csv \
    --out province_audit/unmatched_ids.csv > province_audit/province_audit_report.txt
```

On the drive it found **0 of 567,917** files with an unmatched ID (the inventories hold 8,781 IDs from 2022 and 4,128 more from 2014, 12,909 in all), so the bug misfiled nothing.

### Integrity Check (before the H200 run)

`ec_check_integrity.py` is a read-only check of `PROCESSED_DATA` on the external drive. It:

- Reconciles the file count of each folder with `organization_stats.json`. The organizer counted 7,750 same-name files that it then skipped. Those files are still in `RAW_DATA`, so each one is matched to its twin and byte-compared with it.
- Checks every file name and era.
- Reads, SHA-256 hashes and fully decodes every Organized scan with Pillow, the same way the inference pipeline opens it.

```bash
pip install pillow
python Organization/ec_check_integrity.py "E:\PROCESSED_DATA" --limit 100 --out-dir integrity_smoke   # a few minutes
python Organization/ec_check_integrity.py "E:\PROCESSED_DATA" --out-dir integrity_full              # about 4-6 hours
```

It writes `manifest.csv` (one row per scan with its hash and status), `issues.csv`, `raw_leftovers.csv` and `summary.json` to `--out-dir`. It exits with code 1 if any check fails. If a run is interrupted (Ctrl+C or a disconnected drive), it continues where it stopped with `--resume`.

### Leftover Audit (RAW_DATA alternative scans)

`ec_leftover_audit.py` is a read-only check that SHA-256 hashes every pre-1940 image still in `RAW_DATA` and compares it with its namesake in Organized. It also reads each copy's image header to report formats and pixel sizes. It was written and run in a Claude Code session on 22 Oct 2026 and recovered from that session's transcript; the code is unchanged.

```bash
python -I Organization/ec_leftover_audit.py E:/RAW_DATA E:/PROCESSED_DATA leftover_audit \
    EC_Inventories/inventory_2022.csv EC_Inventories/inventory_2014.csv Organization > leftover_audit/leftover_report.txt
```

It writes `leftovers_all.csv` and `leftovers_differ_or_missing.csv`. Results on the drive:

| Batch | Identical | Differ |
|---|---|---|
| B007_1_Data20210201-20210331 (NOT QUALITY CONTROLLED) | 12 | 266 |
| PNG191203_B003 (QUALITY CONTROLLED) | 0 | 2,094 |
| PNG_191203_B002 | 0 | 64 |
| PNG_210315_B005_ OWI | 8 | 290 |
| PNG_210315_B006_ WW2_WW1 | 0 | 142 |
| PNG_220431_B011_ECCC-OWI | 3 | 45 |
| PNG_230317_B0012 | 252 | 619 |
| **Total (all have a namesake)** | **275** | **3,520** |

The 619 that differ in `PNG_230317_B0012` are corrupt RAW copies: they have the same byte count as their Organized namesake but are not valid images. The other **2,901** are real alternative scans of forms that are in Organized, encoded or sized differently. `Production/extras_paths.txt` lists them: the `RAW_DATA` path of every `leftovers_differ_or_missing.csv` row outside `PNG_230317_B0012`, sorted. They were processed as a separate production run (Stage 3).

---

## Stage 2: Benchmarking (v9)

The benchmark compares VLM configurations against 150 stratified scans with human-verified ground truth (66 with remarks, 84 without), to choose the model and resolution for the production run. It runs on one rented H200. See `Benchmark/README.md` for the setup and run guide, metric definitions and caveats.

### Why Cloud GPUs

Local hardware (RTX 4080, 12 GB VRAM) proved insufficient for this task. The largest model that could run locally was Qwen3-VL-8B at 4-bit quantization, which produced inadequate accuracy and still could not meet the 5 second/file target needed to process the full dataset within a reasonable timeframe. Smaller models (Qwen2.5-VL-3B) improved speed but further degraded quality. Specialized vision encoder-decoders (Florence-2-Large) performed even worse. After exhausting local options, GPU rental was approved for both benchmarking and the eventual production run.

### History

The February 2026 benchmark (v8) tested Qwen3-VL-4B/8B/32B at four pixel budgets on RTX A6000 and H200 GPUs. Its scoring, resolution handling, sampling and timing were later found to be faulty, so v9 replaced it with the same extraction prompt. Earlier testing had already eliminated Qwen2.5-VL, DeepSeek-VL2 and Florence-2-Large.

### v9 Results

Each model runs at pixel budgets ("rungs") of 2, 4, 8 and 16 MiP (1 MiP = 1,048,576 pixels); scans are scaled down to fit, never up. Decoding is greedy with a fixed seed. Chandra OCR 2 cannot take the extraction prompt, so it runs as a full-page OCR recall probe only.

| Config | Sensitivity % | Specificity % | Word recall % | Word precision % | Word F1 % | Truncated | s/file | Corpus USD* |
|---|---|---|---|---|---|---|---|---|
| qwen3vl-8b 2 MiP | 97.0 | 69.0 | 69.3 | 47.5 | 51.1 | 11 | 0.119 | 93 |
| qwen3vl-8b 4 MiP | 97.0 | 72.6 | 72.7 | 53.0 | 56.5 | 10 | 0.246 | 191 |
| qwen3vl-8b 8 MiP | 97.0 | 71.4 | 77.9 | 53.9 | 58.1 | 12 | 0.567 | 440 |
| qwen3vl-8b 16 MiP | 97.0 | 71.4 | 79.2 | 55.0 | 59.3 | 11 | 0.902 | 701 |
| qwen3vl-32b 2 MiP | 90.9 | 83.3 | 65.8 | 44.7 | 46.4 | 8 | 0.352 | 273 |
| qwen3vl-32b 4 MiP | 93.9 | 82.1 | 73.6 | 50.9 | 53.5 | 7 | 0.646 | 501 |
| qwen3vl-32b 8 MiP | 93.9 | 84.5 | 77.0 | 55.2 | 57.7 | 2 | 1.299 | 1,009 |
| qwen3vl-32b 16 MiP | 93.9 | 82.1 | 76.3 | 53.8 | 56.5 | 4 | 1.885 | 1,464 |
| qwen3.8-27b 2 MiP | 98.5 | 58.3 | 75.8 | 50.3 | 56.2 | 7 | 0.292 | 227 |
| **qwen3.8-27b 4 MiP** | **100.0** | **59.5** | **81.0** | **53.9** | **60.7** | **9** | **0.529** | **411** |
| qwen3.8-27b 8 MiP | 100.0 | 57.1 | 81.7 | 53.8 | 60.4 | 7 | 1.048 | 814 |
| qwen3.8-27b 16 MiP | 100.0 | 61.9 | 80.1 | 53.3 | 60.2 | 2 | 1.450 | 1,126 |
| chandra2 6 MiP (OCR probe) | – | – | 68.1 | – | – | 13 | 1.183 | 918 |

\* Projection to 571,712 files on one H200 at $4.89/hour.

**Production choice: Qwen3.8-27B at 4 MiP.** It found every file with remarks (sensitivity 100%) and had the highest word F1 (60.7%); higher budgets did not improve on it and cost two to three times as much. Its trade-off is the lowest specificity of the Qwen models (59.5%): more forms without remarks get some output.

---

## Stage 3: Production Run

`Production/production.py` runs on the local Windows machine. It reads each scan from the external drive, resizes and PNG-encodes it exactly as benchmark v9 does, and sends it to a vLLM OpenAI-compatible server on the rented H200 through an SSH tunnel (`Production/tunnel.bat`). Nothing but vLLM runs on the H200, and no scan is stored there. The output JSONL is also the checkpoint: the script can be stopped at any time and restarted with the same arguments, and it retries timeouts and connection errors automatically. The module docstring has the full status table, the `vllm serve` command and the retry rules. `production.py` and `compare_v9.py` import v9's image and prompt code from `Benchmark/benchmark.py`, so the two folders must stay side by side.

The October 2026 runs were made before these files moved out of `Benchmark/` (28 Oct 2026), so their logs show `Benchmark/` paths.

```bash
# On the H200: start vLLM with v9's engine settings (see production.py's docstring for the full command)
# Locally: fill in Production/tunnel.bat and leave it running, then
python Production/production.py --file-list Production/gt150_paths.txt \
    --inventory-2014 EC_Inventories/inventory_2014.csv --inventory-2022 EC_Inventories/inventory_2022.csv \
    --out runs/gt150.jsonl --backup-dir C:/ec_backup
python Production/compare_v9.py runs/gt150.jsonl results/qwen3.8-27b_4MiP.jsonl --csv runs/compare_v9.csv

python Production/production.py --file-list Production/extras_paths.txt \
    --inventory-2014 EC_Inventories/inventory_2014.csv --inventory-2022 EC_Inventories/inventory_2022.csv \
    --out runs/extras.jsonl --backup-dir C:/ec_backup

python Production/production.py --root "E:\PROCESSED_DATA\Organized" \
    --inventory-2014 EC_Inventories/inventory_2014.csv --inventory-2022 EC_Inventories/inventory_2022.csv \
    --out runs/main.jsonl --backup-dir C:/ec_backup
```

### Runs (October 2026)

| Run | Input | When | Result |
|---|---|---|---|
| `gt150` | The 150 benchmark scans on the drive | 22 Oct | 150 ok |
| `extras` | `extras_paths.txt`: 2,901 alternative scans in `RAW_DATA` | 22 Oct, 30 min | 2,901 ok |
| `main` | All 567,917 scans in `PROCESSED_DATA/Organized` | 22–27 Oct, in two sessions | 567,901 ok, 16 decode_error |

- **Check against v9 (`compare_v9.py` on `gt150`):** 149 of 150 files reached the model with the same pixels and prompt tokens as in v9. The exception, `9904_8502800_1939_01_A.png`, is a different scan: the v9 test image is the 1904×2822 copy from `RAW_DATA`, while production read the 4569×4210 copy in Organized. 134 outputs were identical and 148 of 150 agreed on remarks vs `NO_REMARKS`; greedy vLLM output can still shift with batch composition.
- **Main run:** about 1.7–1.85 files/s, close to the 0.529 s/file the benchmark measured. Its JSONL holds 574,793 lines: one per file plus 6,876 transient errors (timeouts, connection errors, HTTP 5xx), each retried until it succeeded. Two server or tunnel outages (31 s and 5 min) paused sending until vLLM answered again.
- **decode_error:** 16 Organized scans that will not open or decode. They are kept in the exports with their error.
- **finish_reason = length:** 31,158 main and 200 extras outputs hit the 512-token cap. They are mostly repetition loops; their output is kept unedited.

---

## Stage 4: Exports

Both export scripts only read the JSONLs. They keep one row per `source_path` from its latest record, sorted by `source_path`, so failed files stay visible instead of going missing. Pass `main.jsonl` first and `extras.jsonl` second.

```bash
python Production/export_xlsx.py runs/main.jsonl runs/extras.jsonl --xlsx runs/all.xlsx
python Production/export_db.py runs/main.jsonl runs/extras.jsonl --out-dir runs
```

**`export_xlsx.py`** writes the Excel deliverable: a Results sheet with 570,818 rows (567,917 main + 2,901 extras) and a Summary sheet. Text is cleaned for Excel: illegal characters are stripped, cells are capped at 32,767 characters, and text that starts with `=` is stored as text. The columns are the JSONL fields plus:

| Column | Meaning |
|---|---|
| `province_normalized` | The raw `province` (a 2022 inventory name, or a 2014 abbreviation for the 44,575 rows whose station is only in the 2014 inventory) as its Organized folder name, e.g. `Newfoundland and Labrador` |
| `run` | `main` or `extras`: the JSONL the row came from |
| `scans_of_form` | Rows sharing this filename: 2 or 3 for the 2,889 forms with an alternative scan (5,790 rows), 1 otherwise |
| `alternate_scan` | True for rows from `extras` |

**`export_db.py`** writes the same rows for analysis, with the raw JSONL text (no Excel cleaning):

- `ec_remarks.parquet` and the SQLite table `remarks`: the Excel rows, with `native_wh`/`sent_wh` split into integer width and height columns, `climate_id` as text, integer `year`/`month`, `has_remarks` null for failed files, and `utc` as a timestamp (ISO 8601 text in SQLite).
- `ec_attempts.parquet` and the table `attempts`: every readable JSONL line, including the retried timeouts, with `line_no` and `byte_offset`. This is the only record of which files timed out.
- `ec_remarks.sqlite` also has the table `provenance`, which records each input's size, SHA-256 and line count, plus the export time and git commit; `ec_remarks.provenance.json` holds the same information. The database is indexed for common filters and has a full-text index over `output`:

```sql
SELECT r.source_path, r.year, r.output
FROM remarks_fts f JOIN remarks r USING (source_path)
WHERE remarks_fts MATCH 'aurora' AND r.finish_reason = 'stop';
```

The full-text index uses the porter stemmer, so `aurora` also matches `auroras`. The stemmer reduces `auroral` to `auror`, so search `auror*` to include it. Unlike `LIKE '%aurora%'`, a word search does not match inside other words.

### Results

| | Rows |
|---|---|
| Files (main + extras) | 570,818 |
| ok, with remarks | 334,505 |
| ok, `NO_REMARKS` | 236,297 |
| decode_error | 16 |
| ok, `finish_reason = length` | 31,358 |

`has_remarks` is True for most `finish_reason = length` rows, including loops of routine weather (e.g. "Fair & warm") that the project scope excludes. Filter on `finish_reason` when counting remarks.

---

## Stage 5: Post-Processing

The raw output is noisy: two thirds of all output lines are in the 31,358 token-capped repetition loops, and much of the rest is routine weather (states like "Rain" or "Cloudy", wind directions, monthly statistics) that the project scope excludes. `Post-Processing/clean_remarks.py` cleans it line by line. It is **conservative and non-destructive**: `output` and every other input column stay exactly as produced, the cleaned text goes in new columns, and every removed line is listed with the rule that removed it. It needs `duckdb` and `pandas`.

It reads `ec_remarks.parquet` and `ec_remarks.sqlite` (from the same `export_db.py` run) from `Production/`, opening both read-only, and writes its outputs to `Post-Processing/`. Use `--production-dir` and `--out-dir` for other folders.

```bash
pip install duckdb pandas
python Post-Processing/clean_remarks.py                                  # Production/ -> Post-Processing/
python Post-Processing/clean_remarks.py --production-dir runs --out-dir runs
```

Both outputs are new files with the cleaning columns added; the inputs are never modified. They are written under temporary names and renamed once both are complete.

- `ec_remarks_clean.parquet`: every row and column of `ec_remarks.parquet`, plus the cleaning columns.
- `ec_remarks_clean.sqlite`: a copy of `ec_remarks.sqlite` (`remarks`, `attempts`, `provenance`, `remarks_fts`, all unchanged) with the cleaning columns added to `remarks`, plus:
  - `remark_lines`: the `lines` column as a table, one row per line (2,235,190 rows), joined to `remarks` on `source_path`
  - `remarks_clean_fts`: a full-text index over `remarks_clean`, next to the existing `remarks_fts` over `output`
  - `clean_provenance`: each input's path, size and SHA-256, the cleaning time and the git commit of the script

| Cleaning column | Meaning |
|---|---|
| `remarks_clean` | The kept lines in their original order, one per distinct line. In forms that finished normally, a line that occurred N > 1 times ends with ` [×N]`. Empty when nothing was kept |
| `has_remarks_clean` | True if any line was kept |
| `n_lines_raw`, `n_lines_clean` | Non-empty lines in `output` (0 for `NO_REMARKS` rows) and lines kept |
| `n_removed_<rule>` | Distinct lines removed by each rule |
| `lines` | Every distinct line of the form, kept or removed, in order: `line_no`, `text`, `times_in_form`, `removed_by` (null = kept). Empty for `NO_REMARKS` rows. In SQLite, the `remark_lines` table |

`remarks_clean`, `has_remarks_clean` and `lines` are null for the 16 failed files, so a failed file never reads as having no remarks.

```sql
-- Forms whose cleaned remarks mention aurora, with the original output alongside
SELECT r.source_path, r.year, r.month, r.remarks_clean, r.output
FROM remarks_clean_fts f JOIN remarks r USING (source_path)
WHERE remarks_clean_fts MATCH 'aurora';

-- What the cleaning removed from one form, and why
SELECT line_no, text, removed_by FROM remark_lines
WHERE source_path = ? AND removed_by IS NOT NULL ORDER BY line_no;
```

A word search can find fewer forms in `remarks_clean_fts` than in `remarks_fts`: "aurora" matches 16,938 forms in the cleaned text and 18,343 in the raw output. The 1,405 difference is forms whose only aurora lines were zero-count table entries such as "Aurora 0".

Lines are compared after lowercasing and stripping accents and punctuation. Repeats within a form are kept once. Then each distinct line is removed only if it matches one of these rules:

| Rule | Removes | Example | Lines removed |
|---|---|---|---|
| `plain_weather` | Fewer than 8 words, all plain weather, time, direction or function words (some French included) | "Rain", "Fine warm", "Light rain at night" | 771,380 |
| `reading` | At least one number with only statistics, compass or month words, or single letters alone | "Max = 97.0 on 18th", "NE 8" | 381,775 |
| `zero_count` | A printed summary-table entry with a zero count | "Aurora 0", "Gales 0" | 5,922 |
| `null_entry` | Only "No", "None", "Nil", "Nothing" | "None" | 5,186 |
| `cut_off_repeat` | The cut-off last line of a capped output that begins a kept line in the same form | "Qu" after "Quel Cool" | 3,365 |
| `symbols` | No letters or digits | ditto marks | 3,156 |

Everything else is kept, including intensifiers (very, heavy, severe), event words (storm, thunder, frost, hail, gale), expressive words (gloomy, sultry), months and seasons, misspellings, plain-vocabulary narratives of 8+ words, and, for now, lines with hot, cold, dry, drift/drifting, blowing or fall (possible heat, cold, drought and blowing-snow events). No line with a phenology or phenomenon keyword (robin, geese, ice, thaw, aurora, smoke, frost, storm and so on) is removed by the vocabulary rules. The filter errs towards keeping: lines such as "Trace", "Sunday" or "Total" are kept because they carry no number, and can be filtered later.

| | Before | After |
|---|---|---|
| Forms with remarks (main + extras) | 334,505 | 251,381 |
| Forms with remarks (main) | 332,739 | 250,203 |
| Output lines (non-empty) → distinct lines kept | 7,787,744 | 1,064,406 |
| Characters | 74.1M | 22.8M |

Kept lines are still mostly routine; the cleaning only removes lines that are clearly out of scope. To make it stricter later, edit the word lists in the script and re-run it; the `lines` column (`remark_lines` in SQLite) shows exactly which lines any change affects.

---

## Known Challenges

- **Character substitution:** VLMs may misread handwriting (e.g., "Sleet" as "Sheet", "Windy" as "Blindy")
- **Pre-1900 handwriting:** Faded ink and archaic cursive styles are the most consistent source of false negatives across all models
- **French-language forms:** All models tested currently struggle with French content; the 48,651 Quebec files may have lower accuracy
- **Form variation:** Remarks don't appear in consistent locations across decades of form changes, and observers sometimes ignored form structure entirely
- **Ink bleed-through:** Text from the reverse side of pages sometimes appears on scans
- **False positives from adjacent columns:** Full-page extraction occasionally reads content from neighboring form columns (e.g., "General State of Weather"); post-processing filters can catch most of these
- **Repetition loops:** Greedy decoding sometimes repeats a line until the 512-token cap (`finish_reason = length`, about 5.5% of production outputs)
