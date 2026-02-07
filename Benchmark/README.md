# EC Weather Remarks - VLM Benchmark Package

This package benchmarks Vision Language Models for extracting qualitative weather remarks from historical ECCC weather observation forms.

## Quick Start

1. **Prepare benchmark package locally**
2. **Rent a GPU on vast.ai**
3. **Upload files and run benchmarks**
4. **Download results and generate report**

---

## Step 1: Prepare Benchmark Package (Local)

Create a directory with all required files:

```
ec_benchmark/
├── benchmark.py           # Main benchmark script
├── ground_truth.csv      # Your verified ground truth
├── inventory_2022.csv    # Climate ID lookup (2022)
├── inventory_2014.csv    # Climate ID lookup (2014)
└── test_images/               # Your test images
    ├── 9904_2100500_1958_05_A.png
    ├── 9904_2100630_1960_12.png
    └── ...
```

Create a tarball for upload:
```bash
tar -czvf ec_benchmark.tar.gz ec_benchmark/
```

---

## Step 2: Rent a GPU on vast.ai

### 2.1 Create Account
1. Go to https://vast.ai and create an account
2. Add credits

### 2.2 Choose a GPU Instance

**For RTX 6000 Ada (~$0.60/hour):**
- Good for: 32B models at FP8, lower parameter models at FP16
- Search filters: `gpu_name = RTX 6000 Ada`, `disk_space >= 100GB`

**For H200 SXM (~$2.50/hour):**
- Good for: 32B models at FP16 precision, higher at FP8 (or lower) precision
- Search filters: `gpu_name = H200`, `disk_space >= 200GB`

### 2.3 Launch Instance
1. Select an instance with:
   - At least 100GB disk space
   - Docker image: CUDA
   
2. Click "Rent" and wait for instance to start

3. Note the SSH connection string (looks like: `ssh -p 12345 root@123.45.67.89`)

---

## Step 3: Upload and Run Benchmarks

### 3.1 Connect via SSH
```bash
ssh -p <PORT> root@<IP_ADDRESS>
```

### 3.2 Upload Benchmark Package
From a **new local terminal**:
```bash
scp -P <PORT> ec_benchmark.tar.gz root@<IP_ADDRESS>:/root/
```

### 3.3 Setup Environment (on VM)
```bash
cd /root
tar -xzvf ec_benchmark.tar.gz
cd ec_benchmark
```

### 3.4 Run Benchmarks

**Run individual models**
```bash
# Example: Qwen3-VL-8B at 16M resolution
python benchmark.py \
    --model 5 \
    --image-dir ./images \
    --ground-truth ./ground_truth.csv \
    --inventory-2022 ./inventory_2022.csv \
    --inventory-2014 ./inventory_2014.csv \
```

### 3.5 Monitor Progress
- Each file takes ~1-10 seconds depending on model
- 100 files × 10 seconds = ~17 minutes per model
- Watch for errors in the output

---

## Step 4: Download Results and Generate Report

### 4.1 Download Results (from local terminal)
```bash
scp -P <PORT> -r root@<IP_ADDRESS>:/root/ec_benchmark/results ./
```

### 4.2 Destroy the VM
**IMPORTANT:** Stop billing by destroying the instance in the vast.ai dashboard!

---

## Test Matrix

| # | Model | Max Resolution | GPU |
|---|-------|----------------|-----|
| 1 | Qwen/Qwen3-VL-4B-Instruct | 16M | RTX A6000 |
| 2 | Qwen/Qwen3-VL-4B-Instruct | 8M | RTX A6000 |
| 3 | Qwen/Qwen3-VL-4B-Instruct | 4M | RTX A6000 |
| 4 | Qwen/Qwen3-VL-4B-Instruct | 2M | RTX A6000 |
| 5 | Qwen/Qwen3-VL-8B-Instruct | 16M | RTX A6000 |
| 6 | Qwen/Qwen3-VL-8B-Instruct | 8M | RTX A6000 |
| 7 | Qwen/Qwen3-VL-8B-Instruct | 4M | RTX A6000 |
| 8 | Qwen/Qwen3-VL-8B-Instruct | 2M | RTX A6000 |
| 9 | Qwen/Qwen3-VL-32B-Instruct | 16M | H200 |
| 10 | Qwen/Qwen3-VL-32B-Instruct | 8M | H200 |
| 11 | Qwen/Qwen3-VL-32B-Instruct | 4M | H200 |
| 12 | Qwen/Qwen3-VL-32B-Instruct | 2M | H200 |

---

## Output Files

After running benchmarks, you'll have:

```
results/
├── Qwen3-VL-4B-16M_detailed_results.jsonl
├── Qwen3-VL-8B-16M_detailed_results.jsonl
├── Qwen3-VL-32B-16M_detailed_results.jsonl
├── benchmark_summary.json
└── ...
```

---

## Troubleshooting

### "CUDA out of memory"
- Try FP8 quantization instead of FP16
- Reduce `max_num_seqs` in MODEL_CONFIGS
- Use a GPU with more VRAM

### Model download is slow
- vast.ai instances have fast internet; model download should be ~5-10 min for large models
- Consider instances with more bandwidth

### SSH connection refused
- Wait a few minutes for instance to fully start
- Check if instance is "Running" in dashboard
- Use the Jupyter Notebook (installed by default with most instances)

### vLLM errors
- Some model architectures may not be fully supported
- Check vLLM GitHub issues and Wiki for your specific model

---

## Cost Control Tips

1. **Test one model first** before running the full suite
2. **Use spot/interruptible instances** if available (cheaper but may be preempted)
3. **Destroy instances immediately** after downloading results
4. **Monitor costs** in vast.ai dashboard
