# EC Weather Remarks — VLM Benchmark (v9)

Benchmarks vision-language models for extracting qualitative remarks from historical ECCC weather forms,
against the 150-file ground truth (66 with remarks, 84 without). Everything runs on one rented H200.

v9 replaces the February (v8) harness, whose scoring, resolution handling, sampling and timing were faulty.
See the v8 audit for details. The extraction prompt is unchanged from February, so the models are compared on the same prompt.

## Test matrix

| `--model` | Hugging Face ID | Rungs (MiP) | Notes |
|---|---|---|---|
| `qwen3vl-8b` | Qwen/Qwen3-VL-8B-Instruct | 2, 4, 8, 16 | |
| `qwen3vl-32b` | Qwen/Qwen3-VL-32B-Instruct | 2, 4, 8, 16 | February's recommended model |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B | 2, 4, 8, 16 | Thinking disabled (`enable_thinking: False`) |
| `chandra2` | datalab-to/chandra-ocr-2 | 6 | Recall probe only (see below) |

One invocation loads one model once and runs all of its rungs. Rungs are pixel budgets (2 MiP = 2,097,152 px).
Images are scaled down to fit the budget, never up, so a scan smaller than the budget is sent at native size. The actual size sent is logged per file.

**Chandra 2** is trained only on fixed full-page OCR prompts, so it can't take our extraction prompt.
It runs in its own `ocr` mode at its default budget (~6 MiP) and reports **word recall only**: how much of the
ground-truth remark text appears, in order, in its full-page transcript. That is a ceiling on its reading ability. Compare it
with the Qwen models' word recall to decide whether an OCR-then-filter pipeline would be worth building. Its cost figures cover the
OCR pass only; a filter pass would add to them.

## Package layout

```
ec_benchmark/
├── benchmark.py
├── ground_truth.csv
├── inventory_2014.csv
├── inventory_2022.csv
└── test_images/          # the 150 scans named in ground_truth.csv
```

```bash
tar -czvf ec_benchmark.tar.gz ec_benchmark/
```

## 1. Rent the instance (vast.ai)

- 1× H200 (SXM or NVL)
- Disk ≥ 250 GB (the four models total ~150 GB, plus pip packages and results)
- Max CUDA ≥ 12.9 (vLLM 0.29 wheels are built against CUDA 12.9)
- Host RAM ≥ 64 GB is comfortable at the 16 MiP rung
- Any recent PyTorch/CUDA template image

## 2. Set up (on the instance)

```bash
scp -P <PORT> ec_benchmark.tar.gz root@<IP>:/root/        # from your machine
ssh -p <PORT> root@<IP>
tmux new -s bench                                          # survives SSH drops; reattach: tmux a -t bench
cd /root && tar -xzf ec_benchmark.tar.gz && cd ec_benchmark
pip install vllm==0.29.0                                   # pulls torch and transformers>=5.10
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"

# Start all model downloads in the background while you smoke-test
for m in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct Qwen/Qwen3.8-27B datalab-to/chandra-ocr-2; do
  nohup hf download $m > dl_$(basename $m).log 2>&1 &
done
```

## 3. Smoke test (~5 min)

```bash
ARGS="--image-dir test_images --ground-truth ground_truth.csv --inventory-2014 inventory_2014.csv --inventory-2022 inventory_2022.csv --usd-per-hour <your H200 rate>"
python benchmark.py --model qwen3vl-8b --rungs 2 --limit 8 --output-dir smoke $ARGS
```

The smoke test only checks that the pipeline works. Its accuracy and cost figures mean nothing: the first 8 files
are all positives, and with 8 files a single slow request dominates the timing. Check that:
- The startup line reports **0 files without a station match** (or a number you expected).
- In `smoke/summary.csv`: `files` = 8, `sent_mp_max` ≤ 2.1, and `prompt_tokens_median` ≈ 2,300 (2,048 image tokens plus the prompt).
- The JSONL outputs are real transcriptions, not errors or empty strings.

A `truncated` output (`finish_reason: "length"`) is a model behaviour, not a script fault. Greedy decoding sometimes
falls into a repetition loop on a hard scan. The full run reports how often it happens per config.

## 4. Full run

Leaving out `--limit` and `--rungs` runs all 150 files at each model's default rungs (see the test matrix).
`ARGS` is repeated here because a shell variable is lost when you reconnect or open a new tmux window.

```bash
cd /root/ec_benchmark
ARGS="--image-dir test_images --ground-truth ground_truth.csv --inventory-2014 inventory_2014.csv --inventory-2022 inventory_2022.csv --usd-per-hour <your H200 rate>"
for m in qwen3vl-8b qwen3vl-32b qwen3.8-27b chandra2; do
  python benchmark.py --model $m $ARGS 2>&1 | tee log_$m.txt
done
```

Each rung's results are written as soon as it finishes, so a crash only loses the rung in progress. To redo one:
`python benchmark.py --model qwen3vl-32b --rungs 16 $ARGS`. That appends another row to `summary.csv`; delete the stale one.

## 5. Download results and stop billing

```bash
scp -P <PORT> -r root@<IP>:/root/ec_benchmark/results ./     # from your machine
```

Then **destroy** the instance in the vast.ai dashboard. A stopped instance still bills for its disk.

## Outputs

`results/<model>_<rung>MiP.jsonl` has one line per file: native and sent size, prompt/output token counts,
`finish_reason`, the model output, the ground truth, and the per-file scores.

`results/summary.csv` has one row per model × rung:

| Column | Meaning |
|---|---|
| `sensitivity` [`_ci`] | Files with remarks that the model flagged (didn't answer NO_REMARKS), with Wilson 95% CI |
| `specificity` [`_ci`] | Files without remarks that the model correctly answered NO_REMARKS |
| `TP` `FN` `FP` `TN` | File-level detection counts |
| `word_recall` | Mean over the 66 files with remarks: share of ground-truth words the model reproduced, in order. Misses count as 0 |
| `word_precision` | Mean over files where the model output something: share of output words that are in the ground truth. Over-extraction lowers it |
| `word_f1` | Per-file harmonic mean of the two, averaged over the 66 files. Misses count as 0 |
| `truncated` | Outputs that hit `max_tokens` (512 for extraction): almost always repetition loops |
| `sent_mp_*` | Megapixels actually sent to the model |
| `sec_per_file` | GPU wall-clock ÷ files, measured with batched requests |
| `corpus_days`, `corpus_usd` | Projection to 571,712 files on one H200 at `--usd-per-hour` |
| `calls` | Number of `llm.chat` calls the rung was split into (see Caveats) |

Words are compared lower-cased and accent-folded, with digits and punctuation ignored. Line breaks don't affect the score.
File-level precision is not reported: it depends on how common remarks are, and this test set is 44% positive by
construction. Use sensitivity and specificity, and project precision once the corpus rate is estimated.

## Caveats

- Each `llm.chat` call ends only when its slowest request finishes. One looping output (up to 512 tokens) can keep
  the GPU mostly idle for several seconds at the end of a call. `--batch-megapixels` (default 1000) keeps this to one
  call per rung at ≤6 MiP and 2–3 calls at 8/16 MiP. A production server keeps requests continuously in flight,
  so a loop there only occupies one slot. Treat `sec_per_file` as a slightly conservative estimate, most of all
  for configs with a high `truncated` count.
- Decoding is greedy with a fixed seed. vLLM results can still differ in the last digit between batch sizes, so compare configs on
  their confidence intervals, not on one-file differences.
- The production run is only as fast as scans can reach the GPU. Check that your upload bandwidth can sustain
  (file size ÷ `sec_per_file`) at the chosen rung.
- Chandra's own client retries repetition loops at a higher temperature. This probe doesn't, and flags them in `truncated` instead.

## Troubleshooting

- **Host RAM runs out (process killed at 8/16 MiP):** rerun that rung with `--batch-megapixels 400`. Check
  headroom first with `free -g`; the default needs roughly 30 GB free.
- **CUDA OOM while loading or at 16 MiP:** lower `gpu_memory_utilization` (0.90) in the `LLM(...)` call.
- **A model fails to load:** check the `transformers` version printed in setup (Qwen3.8 needs ≥ 5.8), and look at vLLM's model support issues on GitHub.
