# EC Historical Weather Remarks Extraction Pipeline

A Python pipeline for extracting qualitative weather remarks from Environment Canada's (EC) pre-1940 historical weather observation forms using Vision Language Models (VLMs) running on rented cloud GPUs.

## Project Overview

### Goal

Extract handwritten qualitative remarks from pre-1940 weather observation forms, focusing on:
- Phenological observations (first robin in spring, ice break-up, harvest timing)
- Extreme weather events (storms, floods, fires, droughts)
- General weather descriptions that reveal how Canadians understood and communicated about weather and nature

### Dataset

The source dataset consists of approximately 1 million digitized scans (PNG/TIF) of monthly weather observation forms from Canadian (and some US) weather stations spanning the 1870s through the 1960s. Of these, 571,712 pre-1940 files are the target for processing.

### Approach

The pipeline uses full-page VLM inference to extract qualitative remarks directly from scans. The VLM handles both the vision task (reading handwritten text) and the logic task (identifying which text constitutes qualitative remarks worth extracting). Inference runs on rented GPU hardware via [vast.ai](https://vast.ai), with images sent individually from a local machine to the remote server. This avoids uploading the full 6 TB dataset to the cloud and respects data handling requirements, as no permanent copy of the scans is stored remotely.

Earlier approaches that were explored and ultimately abandoned include region-of-interest cropping (too many form variants), extract-everything-then-filter post-processing (unnecessary with sufficiently powerful models), and local-only inference (hardware constraints made it infeasible at scale).

---

## Project Structure

```
EC-HTR/
├── README.md                          # This file
│
├── Organization/                      # Stage 1: File organization tools
│   ├── ec_organizer_enhanced.py       # Sorts scans by province/territory and year
│   ├── ec_verify_organization.py      # Verifies correct file placement
│   ├── EC-Inventory-2014.csv          # Station inventory (18,707 stations)
│   ├── EC-Inventory-2022.csv          # Station inventory (8,781 stations)
│   ├── organization_stats.csv         # Organization results
│   ├── file_organization.log          # Processing log
│   └── verification_results.json      # Verification results
│
└── Benchmark/                         # Stage 2: VLM benchmarking package
    ├── benchmark.py                   # Main VLM benchmarking script (runs on vast.ai)
    ├── ground_truth.csv               # Human-verified extractions for 150 test images
    ├── EC-Inventory-2022.csv          # Station inventory for lookups
    ├── EC-Inventory-2014.csv          # Station inventory for lookups
    ├── README.md                      # Step-by-step vast.ai setup guide
    └── test_images/                   # 150 stratified sample images for benchmarking
```

---

## Stage 1: Organization

The organization stage sorts the raw dataset of ~1 million scans into a structured directory hierarchy by province/territory, using each file's Climate ID to determine its geographic origin.

### How It Works

Each scan filename follows the convention `9904_CLIMATEID_YYYY_MM[_A].png`, where the 7-digit Climate ID encodes the province (first digit), climatological district (digits 2-3), and station (digits 4-7). The organizer cross-references each Climate ID against two EC station inventory CSVs (2014 and 2022) to resolve the province/territory, then moves the file into the corresponding directory. Files dated after 1939 are separated into an "irrelevant" directory since they fall outside the project's scope.

Non-standard Climate IDs (e.g., `610ML02`, `706CFQ3`) are looked up directly in the inventory CSVs. Any files that cannot be resolved are flagged for manual review.

### Results

| Metric | Count |
|--------|-------|
| Total files found | 996,602 |
| Files organized | 988,852 |
| Pre-1940 files (target set) | 571,712 |
| Post-1939 files (excluded) | 424,890 |
| Non-standard IDs resolved | 3,536 |
| Files needing manual QC | 0 |
| Errors | 0 |

**Pre-1940 files by province/territory:**

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

```bash
python Organization/ec_organizer_enhanced.py \
    --input-dir /path/to/raw/scans \
    --output-dir /path/to/organized \
    --csv-2014 Organization/EC-Inventory-2014.csv \
    --csv-2022 Organization/EC-Inventory-2022.csv

python Organization/ec_verify_organization.py \
    --organized-dir /path/to/organized \
    --csv-2014 Organization/EC-Inventory-2014.csv \
    --csv-2022 Organization/EC-Inventory-2022.csv
```

---

## Stage 2: Benchmarking

The benchmarking stage evaluates VLM configurations to determine the best model, resolution, and GPU combination for processing the full dataset. This stage runs on rented GPU hardware from vast.ai.

### Why Cloud GPUs

Local hardware (RTX 4080, 12 GB VRAM) proved insufficient for this task. The largest model that could run locally was Qwen3-VL-8B at 4-bit quantization, which produced inadequate accuracy and still could not meet the 5 second/file target needed to process the full dataset within a reasonable timeframe. Smaller models (Qwen2.5-VL-3B) improved speed but further degraded quality. Specialized vision encoder-decoders (Florence-2-Large) performed even worse. After exhausting local options, GPU rental was approved for both benchmarking and the eventual production run.

### Models Tested

Three models from the Qwen3-VL family were selected, all running at FP16 precision on the vLLM inference framework. Earlier testing eliminated Qwen2.5-VL, DeepSeek-VL2, and Florence-2-Large due to inferior performance or excessive suggestibility.

| Model | Parameters | GPU | Hourly Cost (USD) |
|-------|-----------|-----|--------------------|
| Qwen3-VL-4B-Instruct | 4B | NVIDIA RTX A6000 (48 GB) | $0.60 |
| Qwen3-VL-8B-Instruct | 8B | NVIDIA RTX A6000 (48 GB) | $0.60 |
| Qwen3-VL-32B-Instruct | 32B | NVIDIA H200 SXM (141 GB) | $2.50 |

Each model was tested at four maximum pixel budgets (16M, 8M, 4M, 2M megapixels) for a total of 12 configurations. The test dataset consists of 150 stratified random samples with manually verified ground truth.

### Usage

See `Benchmark/README.md` for step-by-step instructions on renting a GPU from vast.ai and running benchmarks. In brief:

```bash
# On the rented VM
python benchmark.py \
    --model 9 \
    --image-dir ./test_images \
    --ground-truth ./ground_truth.csv \
    --inventory-2022 ./EC-Inventory-2022.csv \
    --inventory-2014 ./EC-Inventory-2014.csv \
```

---

## Known Challenges

- **Character substitution:** VLMs may misread handwriting (e.g., "Sleet" as "Sheet", "Windy" as "Blindy")
- **Pre-1900 handwriting:** Faded ink and archaic cursive styles are the most consistent source of false negatives across all models
- **French-language forms:** All models tested currently struggle with French content; the 48,651 Quebec files may have lower accuracy
- **Form variation:** Remarks don't appear in consistent locations across decades of form changes, and observers sometimes ignored form structure entirely
- **Ink bleed-through:** Text from the reverse side of pages sometimes appears on scans
- **False positives from adjacent columns:** Full-page extraction occasionally reads content from neighboring form columns (e.g., "General State of Weather"); post-processing filters can catch most of these
