# EC Historical Weather Remarks Extraction Pipeline

A Python pipeline for extracting qualitative weather remarks from Environment Canada's (EC) historical weather observation forms using Vision Language Models (VLMs).

## Project Overview

### Goal

Extract handwritten qualitative remarks from pre-1940 weather observation forms, focusing on:
- Phenological observations (first robin in spring, ice break-up, harvest timing)
- Extreme weather events (storms, floods, fires, droughts)
- General weather descriptions that reveal how Canadians understood and communicated about weather and nature

### Approach

The pipeline uses a Vision Language Model (VLM) to perform handwritten text recognition (HTR) on full-page scans, followed by post-processing filters to isolate qualitative remarks from quantitative data and form boilerplate.

---

## System Requirements

### Hardware

**My Configuration:**
- GPU: NVIDIA RTX 4080 (12GB VRAM)
- RAM: 32GB
- CPU: AMD Ryzen 9 7945HX
- Storage: 6TB+ for full dataset

**Recommended for production:**
- GPU: 24GB+ VRAM (RTX 4090, A100, etc.)
- RAM: 32GB+
- Fast SSD storage for scan files

### Software

- Python 3.13
- CUDA 11.8+ (for GPU acceleration)
- Ollama (for local VLM inference)

### Python Dependencies

```bash
pip install pandas openpyxl Pillow requests
```

For VLM backends (install as needed):
```bash
# Ollama backend (recommended)
# Install Ollama separately: https://ollama.ai

# Optional: vLLM backend
pip install vllm

# Optional: Transformers backend  
pip install transformers torch accelerate qwen-vl-utils
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EC HTR EXTRACTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Scan Files │────▶│ vlm_processor    │────▶│  results.jsonl  │
│  (PNG/TIF)  │     │       .py        │     │  (raw VLM text) │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                            │                         │
                            ▼                         ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │ vlm_backends.py  │     │ postprocess_    │
                    │ (Ollama/vLLM/    │     │ remarks.py      │
                    │  Transformers)   │     └────────┬────────┘
                    └──────────────────┘              │
                                                      ▼
┌─────────────────┐                         ┌─────────────────┐
│ Station CSVs    │                         │ extracted_      │
│ (2014 & 2022    │                         │ remarks.xlsx    │
│  inventories)   │                         │ (final output)  │
└─────────────────┘                         └─────────────────┘
```

### Components

| File | Purpose |
|------|---------|
| `vlm_processor.py` | Main processing script - parses filenames, looks up stations, calls VLM |
| `vlm_backends.py` | VLM abstraction layer - supports Ollama, vLLM, Transformers backends |
| `postprocess_remarks.py` | Filters raw VLM output to extract qualitative remarks |
| `test_vlm_processor.py` | Validation utilities for testing and debugging |

---

## Data Format

### Input: Scan Filenames

Scans follow the naming convention:
```
9904_CLIMATEID_YYYY_MM[_A].png
```

| Component | Description | Example |
|-----------|-------------|---------|
| `9904` | Form code (historical monthly observation form) | `9904` |
| `CLIMATEID` | 7-digit station identifier | `1010774` |
| `YYYY` | Year | `1932` |
| `MM` | Month (01-12) | `11` |
| `_A` | Optional suffix for backside of form | `_A` |

**Example:** `9904_1010774_1932_11_A.png` = November 1932, Station 1010774, backside

### Climate ID Structure

The Climate ID is a 7-digit number assigned by the Meteorological Service of Canada:
- **Digit 1:** Province code
- **Digits 2-3:** Climatological district within province
- **Digits 4-7:** Unique station identifier

### Station Inventory CSVs

Two CSV files provide station metadata:

**2014 Inventory** (`EC_Historical_Weather_Station_inventory_2014_01.csv`):
- 18,707 stations
- Contains: Climate ID, Station Name, Province, Coordinates, Date ranges

**2022 Inventory** (`EC_Historical_Weather_Station_inventory_2022.csv`):
- 8,781 stations
- Contains: Similar fields with some variations

The processor merges both sources, preferring 2022 data when available.

---

## Installation

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start the service
sudo systemctl start ollama

# Pull the model
ollama pull qwen2.5-vl:7b
```

### 2. Clone/Download Scripts

Place all Python scripts in your working directory:
```
project/
├── vlm_processor.py
├── vlm_backends.py
├── postprocess_remarks.py
├── test_ec_processor.py
├── EC_Historical_Weather_Station_inventory_2014_01.csv
├── EC_Historical_Weather_Station_inventory_2022.csv
└── scans/           # Your scan files
    ├── 9904_1010774_1932_11.png
    └── ...
```

### 3. Install Python Dependencies

```bash
pip install pandas openpyxl Pillow requests
```

---

## Usage

### Stage 1: VLM Text Extraction

Extract all handwritten text from scans:

```bash
python ec_vlm_processor.py \
    --input-dir ./scans \
    --output ./results.jsonl \
    --csv-2014 ./EC_Historical_Weather_Station_inventory_2014_01.csv \
    --csv-2022 ./EC_Historical_Weather_Station_inventory_2022.csv \
    --model qwen3-vl:8b-instruct \
    --backend ollama
```

**Key Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | Required | Directory containing scan files |
| `--output` | `results.jsonl` | Output file for VLM results |
| `--csv-2014` | Required | Path to 2014 station inventory |
| `--csv-2022` | Required | Path to 2022 station inventory |
| `--model` | `qwen3-vl:8b-instruct` | VLM model name |
| `--backend` | `ollama` | VLM backend (ollama/vllm/transformers) |
| `--batch-size` | `10` | Files per checkpoint |
| `--skip-backside` | `True` | Skip files ending in `_A` |
| `--cutoff-year` | `1940` | Skip files after this year |

**Checkpointing:**

The processor automatically saves progress every `--batch-size` files. If interrupted, simply re-run the same command to resume from where it stopped.

### Stage 2: Post-Processing

Filter raw VLM output to extract qualitative remarks:

```bash
python postprocess_remarks.py \
    ./results.jsonl \
    --output ./extracted_remarks.xlsx
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `input` | Required | Input JSONL file from Stage 1 |
| `--output` | `extracted_remarks.xlsx` | Output Excel file |
| `--verbose` | False | Print sample extractions |

---

## Output Format

### Stage 1 Output: `results.jsonl`

JSON Lines format with one record per file:

```json
{
  "filepath": "/path/to/9904_1010774_1932_11.png",
  "filename": "9904_1010774_1932_11.png",
  "climate_id": "1010774",
  "station_name": "VICTORIA INTL A",
  "province": "BC",
  "location": "British Columbia",
  "year": 1932,
  "month": 11,
  "extracted_text": "Rain all day; Heavy frost overnight; First snow of season",
  "processing_time": 4.23,
  "status": "success",
  "timestamp": "2025-11-25T14:30:00"
}
```

### Stage 2 Output: `extracted_remarks.xlsx`

Excel workbook with three sheets:

**Sheet 1: Extracted Remarks**
| Filename | Year | Month | Station | Location | Remarks |
|----------|------|-------|---------|----------|---------|
| 9904_1010774_1932_11.png | 1932 | 11 | VICTORIA INTL A | British Columbia | Rain all day; Heavy frost overnight |

**Sheet 2: Summary**
| Metric | Value |
|--------|-------|
| Total files processed | 100 |
| Files with qualitative remarks | 74 |
| Files without remarks | 26 |
| Total individual remarks extracted | 677 |
| Extraction rate | 74.0% |

**Sheet 3: All Files**

Complete list of all processed files with metadata and status.

---

## Post-Processing Filters

The post-processor removes the following categories of text:

### Filtered Out

| Category | Examples |
|----------|----------|
| **Form boilerplate** | "Station:", "Province:", "Observer:", "For the month of" |
| **Column headers** | "Remarks", "Wind", "Temperature", "Precipitation" |
| **Quantitative data** | Numbers, measurements, coordinates |
| **Time entries** | "8:00 a.m.", "Noon to Midnight" |
| **Statistical terms** | "Means", "Sums", "Total", "Diff. from Normal" |
| **Month/day labels** | "January", "Mon.", "Day 1:" |
| **Generic weather terms** | "Fine", "Fair", "Good", "Mild" (in isolation) |
| **VLM artifacts** | Repeated text, gibberish patterns |

### Preserved

| Category | Examples |
|----------|----------|
| **Phenological observations** | "First robin of spring", "Ice break-up", "Commenced haying" |
| **Extreme weather** | "Blizzard", "Gale", "Forest fire", "Thunderstorm" |
| **Descriptive remarks** | "Very wet miserable", "Fog & gusty gale", "Heavy frost" |
| **Compound observations** | "Clear & cold", "Cloudy - Snowfall", "Rain all day" |

---

## Performance Considerations

### Processing Speed

| Model | Hardware | Speed | Notes |
|-------|----------|-------|-------|
| qwen2.5-vl:7b | RTX 4080 12GB | 5-15 sec/file | Recommended for testing |
| qwen2.5-vl:3b | RTX 4080 12GB | 2-5 sec/file | Faster but less accurate |
| qwen3-vl:8b | 24GB+ VRAM | 3-8 sec/file | Best accuracy |

### Memory Management

- Images are processed one at a time to minimize VRAM usage
- Checkpointing prevents loss of progress
- Consider splitting large directories into batches

---

## Known Limitations

### VLM Accuracy Issues

1. **Character substitution:** "Blindy" instead of "Windy", "Sheet" instead of "Sleet"
2. **Repetition loops:** VLM sometimes outputs the same word hundreds of times
3. **Difficult cursive:** Some handwriting from the 1880s-1900s is very challenging
4. **Ink bleed-through:** Text from the back of pages sometimes appears

### Filter Limitations

1. **Over-filtering:** Some legitimate remarks may be removed if they match boilerplate patterns
2. **Under-filtering:** Some form structure still passes through filters
3. **Language:** Filters tuned for English and French; other languages may need additions

---

## Future Improvements

1. **Model upgrade:** Test with larger VLMs when hardware allows
2. **Confidence scoring:** Add uncertainty estimates to extractions
3. **Active learning:** Flag low-confidence extractions for human review
4. **Batch processing:** Implement parallel processing for cloud deployment
