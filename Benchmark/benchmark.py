#!/usr/bin/env python3
"""
EC Weather Remarks - VLM Benchmark Script v8
Addresses critical issues from v7 code review.

Fixes from v7:
1. Qwen3-VL: Added thinking block stripping from output
2. Stop tokens: Made model-specific (ChatML for Qwen)
3. Image format: Created model-family-specific image passing formats
4. Patch size: Now model-specific (32 for Qwen3-VL)

References:
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
- Qwen3 thinking mode: https://qwen.readthedocs.io/en/latest/deployment/vllm.html
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from PIL import Image

# Check dependencies at startup
def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        missing.append("vllm")
    
    if missing:
        print(f"ERROR: Missing required packages: {missing}")
        sys.exit(1)

check_dependencies()

from vllm import LLM, SamplingParams


class ModelFamily(Enum):
    """Model families with different requirements."""
    QWEN3_VL = "qwen3-vl"


@dataclass
class ModelConfig:
    """Configuration for a VLM model with family-specific settings."""
    name: str
    model_id: str
    family: ModelFamily
    dtype: str = "auto"
    quantization: Optional[str] = None
    gpu: str = "RTX A6000"
    max_model_len: int = 32768
    max_pixels: Optional[int] = None  # Per-model max pixels, None uses CLI default
    trust_remote_code: bool = False
    extra_vllm_args: dict = field(default_factory=dict)
    
    @property
    def patch_size(self) -> int:
        """Return the correct patch size for this model family."""
        if self.family == ModelFamily.QWEN3_VL:
            return 32
        return 28  # Default fallback
    
    @property
    def stop_tokens(self) -> List[str]:
        """Return the correct stop tokens for this model family."""
        if self.family == ModelFamily.QWEN3_VL:
            # ChatML format stop tokens
            return ["<|im_end|>", "<|endoftext|>"]
        return ["<|im_end|>", "<|endoftext|>"]  # Default to ChatML
    
    @property
    def has_thinking_mode(self) -> bool:
        """Check if this model has thinking mode that needs stripping."""
        return self.family == ModelFamily.QWEN3_VL


# Model configurations - Testing Matrix
# Each model size is tested at 4 resolutions (16M, 8M, 4M, 2M megapixels)
MODEL_CONFIGS = {
    # Qwen3-VL-4B (RTX A6000)
    1: ModelConfig(
        name="Qwen3-VL-4B-16M",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=16_777_216,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    2: ModelConfig(
        name="Qwen3-VL-4B-8M",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=8_388_608,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    3: ModelConfig(
        name="Qwen3-VL-4B-4M",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=4_194_304,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    4: ModelConfig(
        name="Qwen3-VL-4B-2M",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=2_097_152,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    # Qwen3-VL-8B (RTX A6000)
    5: ModelConfig(
        name="Qwen3-VL-8B-16M",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=16_777_216,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    6: ModelConfig(
        name="Qwen3-VL-8B-8M",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=8_388_608,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    7: ModelConfig(
        name="Qwen3-VL-8B-4M",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=4_194_304,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    8: ModelConfig(
        name="Qwen3-VL-8B-2M",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="RTX A6000",
        max_pixels=2_097_152,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    # Qwen3-VL-32B (H200)
    9: ModelConfig(
        name="Qwen3-VL-32B-16M",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="H200",
        max_pixels=16_777_216,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    10: ModelConfig(
        name="Qwen3-VL-32B-8M",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="H200",
        max_pixels=8_388_608,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    11: ModelConfig(
        name="Qwen3-VL-32B-4M",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="H200",
        max_pixels=4_194_304,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
    12: ModelConfig(
        name="Qwen3-VL-32B-2M",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        family=ModelFamily.QWEN3_VL,
        dtype="bfloat16",
        gpu="H200",
        max_pixels=2_097_152,
        extra_vllm_args={"limit_mm_per_prompt": {"image": 1, "video": 0}},
    ),
}


# Image resizing configuration
DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024  # ~1300 image tokens for Qwen


def resize_image_for_vlm(image_path: Path, max_pixels: int, max_dim: int, 
                          patch_size: int = 28) -> Image.Image:
    """
    Resize image to reduce token count while preserving aspect ratio.

    For Qwen3-VL, image tokens ≈ (width * height) / (32 * 32)

    Args:
        image_path: Path to the image file
        max_pixels: Maximum total pixels allowed
        max_dim: Maximum dimension (width or height)
        patch_size: Model-specific patch size (32 for Qwen3-VL)
    """
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (handles RGBA, grayscale, palette, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
    
    original_size = img.size
    width, height = img.size
    
    # Check if resize is needed
    current_pixels = width * height
    
    if current_pixels <= max_pixels and width <= max_dim and height <= max_dim:
        return img
    
    # Calculate scale factor
    scale_by_pixels = (max_pixels / current_pixels) ** 0.5
    scale_by_dim = min(max_dim / width, max_dim / height)
    scale = min(scale_by_pixels, scale_by_dim)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Ensure dimensions are multiples of patch_size (model-specific)
    new_width = (new_width // patch_size) * patch_size
    new_height = (new_height // patch_size) * patch_size
    
    # Minimum size
    min_size = patch_size * 8  # At least 8 patches per dimension
    new_width = max(new_width, min_size)
    new_height = max(new_height, min_size)
    
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    print(f"    Resized: {original_size} -> ({new_width}, {new_height}) "
          f"[{current_pixels:,} -> {new_width*new_height:,} pixels, "
          f"patch_size={patch_size}]")
    
    return img_resized


def strip_thinking_blocks(text: str) -> str:
    """
    Strip <think>...</think> blocks from Qwen3-VL output.
    
    Qwen3 models have thinking mode enabled by default, which generates
    reasoning in <think> tags before the final response. For benchmark
    accuracy, we only want the final response.
    
    Args:
        text: Raw model output that may contain thinking blocks
        
    Returns:
        Text with thinking blocks removed
    """
    # Pattern matches <think>...</think> including newlines
    # Uses DOTALL flag to match across multiple lines
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Also handle potential variations
    pattern2 = r'<\|think\|>.*?<\|/think\|>'
    cleaned = re.sub(pattern2, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any resulting double newlines or leading/trailing whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def build_qwen_messages(img: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    """
    Build chat messages for Qwen3-VL models.
    
    Uses the image_pil format supported by vLLM for direct PIL image input.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_pil",
                    "image_pil": img,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def load_ground_truth(csv_path: Path) -> dict:
    """Load ground truth data from CSV."""
    ground_truth = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            if filename:
                ground_truth[filename] = {
                    'verified_extraction': row.get('verified_extraction', ''),
                    'climate_id': row.get('climate_id', ''),
                }
    return ground_truth


def load_station_inventories(inv_2014_path: Path, inv_2022_path: Path) -> dict:
    """Load station inventory data for location context."""
    stations = {}
    
    # Load 2014 inventory
    if inv_2014_path.exists():
        with open(inv_2014_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                climate_id = row.get('Climate ID', '').strip()
                if climate_id:
                    stations[climate_id] = {
                        'name': row.get('Station Name (Current)', ''),
                        'province': row.get('Province', ''),
                    }
    
    # Load 2022 inventory (may have updated info)
    if inv_2022_path.exists():
        with open(inv_2022_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                climate_id = row.get('climate_id', '').strip()
                if climate_id:
                    stations[climate_id] = {
                        'name': row.get('Name', ''),
                        'province': row.get('Province', ''),
                    }
    
    return stations


def extract_climate_id(filename: str) -> str:
    """Extract climate ID from filename like '9904_2100500_1958_05_A.png'."""
    parts = filename.replace('.png', '').replace('.jpg', '').split('_')
    if len(parts) >= 2:
        return parts[1]
    return ''


def build_prompt(station_name: str, province: str, year: str, month: str) -> str:
    """Build the extraction prompt with location context."""
    return f"""You are analyzing a historical weather observation form from {station_name}, {province}, Canada from {year}-{month}.

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

def is_no_remarks(text: str) -> bool:
    """
    Check if text indicates "no remarks".
    Uses EXACT match only - not substring matching.
    """
    text_upper = text.upper().strip()
    
    # Empty or whitespace only
    if not text_upper:
        return True
    
    # Exact matches only (not substring!)
    no_remarks_indicators = {"NO_REMARKS", "NO REMARKS", "NONE", "N/A", "NA", "NIL"}
    if text_upper in no_remarks_indicators:
        return True
    
    return False


def calculate_accuracy(model_output: str, ground_truth: str) -> tuple:
    """
    Calculate if model output matches ground truth.
    Returns (is_match: bool, match_type: str)
    
    Match types:
    - exact: Strings match exactly (case-insensitive, whitespace-normalized)
    - true_negative: Both correctly indicate no remarks
    - partial_XX%: At least 50% of ground truth lines have matching content
    - false_negative: Model said NO_REMARKS but ground truth has content (MISS)
    - false_positive: Model has content but ground truth says NO_REMARKS (HALLUCINATION)
    - no_match_XX%: Both have content but insufficient overlap
    """
    model_clean = model_output.strip()
    gt_clean = ground_truth.strip()
    
    # Determine if each indicates "no remarks" FIRST
    # This must happen before exact match to correctly classify true negatives.
    # Otherwise, both saying "NO_REMARKS" matches exactly and gets counted as
    # a true positive in the summary metrics, inflating precision/recall/F1.
    model_no_remarks = is_no_remarks(model_clean)
    gt_no_remarks = is_no_remarks(gt_clean)
    
    # Both indicate no remarks - TRUE NEGATIVE
    if model_no_remarks and gt_no_remarks:
        return True, "true_negative"
    
    # Model says no remarks but ground truth has content - FALSE NEGATIVE (miss)
    if model_no_remarks and not gt_no_remarks:
        return False, "false_negative"
    
    # Model has output but ground truth says no remarks - FALSE POSITIVE (hallucination)
    if not model_no_remarks and gt_no_remarks:
        return False, "false_positive"
    
    # Both have content - check exact match first
    # Normalize for comparison
    model_normalized = ' '.join(model_clean.upper().split())
    gt_normalized = ' '.join(gt_clean.upper().split())
    
    if model_normalized == gt_normalized:
        return True, "exact"
    
    # Both have content - check partial match using word overlap
    model_lines = [l.strip() for l in model_clean.split('\n') if l.strip()]
    gt_lines = [l.strip() for l in gt_clean.split('\n') if l.strip()]
    
    if not gt_lines:
        # Edge case: gt appears empty after splitting but wasn't caught above
        return False, "empty_gt"
    
    # Count how many ground truth lines have a matching model line
    # A match requires at least 50% word overlap
    matches = 0
    for gt_line in gt_lines:
        gt_words = set(gt_line.upper().split())
        if not gt_words:
            continue
            
        best_overlap = 0
        for model_line in model_lines:
            model_words = set(model_line.upper().split())
            if model_words:
                # Overlap = intersection / ground truth words
                overlap = len(gt_words & model_words) / len(gt_words)
                best_overlap = max(best_overlap, overlap)
        
        if best_overlap >= 0.5:  # At least 50% of GT words found
            matches += 1
    
    match_ratio = matches / len(gt_lines) if gt_lines else 0
    
    # At least 50% of ground truth lines must match
    if match_ratio >= 0.5:
        return True, f"partial_{int(match_ratio*100)}%"
    
    return False, f"no_match_{int(match_ratio*100)}%"


def run_benchmark(config: ModelConfig, ground_truth: dict, stations: dict,
                  image_dir: Path, output_dir: Path, max_pixels: int, max_dim: int) -> dict:
    """Run benchmark for a single model configuration."""
    # CLI --max-pixels always takes precedence over per-model defaults.
    # Per-model max_pixels is only used if no CLI value was provided (i.e., using the
    # script's DEFAULT_MAX_IMAGE_PIXELS). This prevents silent overrides when the user
    # explicitly specifies a resolution.
    if max_pixels != DEFAULT_MAX_IMAGE_PIXELS:
        # User explicitly set --max-pixels on CLI, always use it
        effective_max_pixels = max_pixels
        if config.max_pixels is not None and config.max_pixels != max_pixels:
            print(f"  NOTE: CLI --max-pixels ({max_pixels:,}) overrides "
                  f"per-model default ({config.max_pixels:,}) for {config.name}")
    else:
        # No explicit CLI value, use per-model default if available
        effective_max_pixels = config.max_pixels if config.max_pixels is not None else max_pixels
    effective_max_dim = int(math.ceil(math.sqrt(effective_max_pixels * 1.5)))

    print(f"\n{'='*80}")
    print(f"Testing model: {config.name}")
    print(f"Model ID: {config.model_id}")
    print(f"Model Family: {config.family.value}")
    print(f"DType: {config.dtype}")
    print(f"Quantization: {config.quantization or 'None'}")
    print(f"Max Model Len: {config.max_model_len}")
    print(f"GPU: {config.gpu}")
    print(f"Patch Size: {config.patch_size}")
    print(f"Stop Tokens: {config.stop_tokens}")
    print(f"Has Thinking Mode: {config.has_thinking_mode}")
    print(f"Max Image Pixels: {effective_max_pixels:,}")
    print(f"Max Image Dimension: {effective_max_dim}")
    print(f"{'='*80}\n")
    
    # Build vLLM arguments
    vllm_args = {
        "model": config.model_id,
        "dtype": config.dtype,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": 0.95,
        "disable_log_stats": True,
        "trust_remote_code": config.trust_remote_code,
    }

    # Add limit_mm_per_prompt only if not already in extra_vllm_args
    if "limit_mm_per_prompt" not in config.extra_vllm_args:
        vllm_args["limit_mm_per_prompt"] = {"image": 1}

    if config.quantization:
        vllm_args["quantization"] = config.quantization

    # Apply extra model-specific args
    vllm_args.update(config.extra_vllm_args)
    
    # Initialize vLLM
    print("Initializing vLLM...")
    print(f"vLLM args: {json.dumps({k: str(v) for k, v in vllm_args.items()}, indent=2)}")
    try:
        llm = LLM(**vllm_args)
        print("vLLM initialized successfully\n")
    except Exception as e:
        print(f"ERROR: Failed to initialize vLLM: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "config": config.name}
    
    # Use model-specific stop tokens
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        stop=config.stop_tokens,
    )
    
    results = []
    total_time = 0
    successful = 0
    errors = 0
    
    # Detailed accuracy counters
    match_counts = {
        "exact": 0,
        "true_negative": 0,
        "partial": 0,
        "false_negative": 0,
        "false_positive": 0,
        "no_match": 0,
    }
    
    # Process each test image
    test_files = sorted([f for f in ground_truth.keys() if (image_dir / f).exists()])
    
    if not test_files:
        print(f"ERROR: No test files found in {image_dir}")
        return {"error": "No test files found", "config": config.name}
    
    print(f"Found {len(test_files)} test files\n")
    
    for idx, filename in enumerate(test_files, 1):
        print(f"Processing [{idx}/{len(test_files)}]: {filename}")
        
        image_path = image_dir / filename
        gt_data = ground_truth[filename]
        
        # Get location context
        climate_id = extract_climate_id(filename)
        station_info = stations.get(climate_id, {})
        station_name = station_info.get('name', 'Unknown Station')
        province = station_info.get('province', 'Unknown Province')
        
        # Extract year and month from filename
        parts = filename.replace('.png', '').replace('.jpg', '').split('_')
        year = parts[2] if len(parts) > 2 else 'Unknown'
        month = parts[3] if len(parts) > 3 else 'Unknown'
        
        # Build prompt
        prompt = build_prompt(station_name, province, year, month)
        
        try:
            # Resize image using model-specific patch size and pixel limits
            img = resize_image_for_vlm(
                image_path,
                max_pixels=effective_max_pixels,
                max_dim=effective_max_dim,
                patch_size=config.patch_size
            )
            
            # Build chat messages and run inference
            messages = build_qwen_messages(img, prompt)

            start_time = time.time()
            outputs = llm.chat(messages, sampling_params=sampling_params)
            elapsed = time.time() - start_time
            
            if outputs and outputs[0].outputs:
                model_output = outputs[0].outputs[0].text.strip()
                
                # Strip thinking blocks for Qwen3-VL
                if config.has_thinking_mode:
                    original_output = model_output
                    model_output = strip_thinking_blocks(model_output)
                    if model_output != original_output:
                        print(f"    Stripped thinking blocks from output")
                
                total_time += elapsed
                successful += 1
                
                # Calculate accuracy
                is_match, match_type = calculate_accuracy(
                    model_output, gt_data['verified_extraction']
                )
                
                # Update match counts
                if match_type == "exact":
                    match_counts["exact"] += 1
                elif match_type == "true_negative":
                    match_counts["true_negative"] += 1
                elif match_type.startswith("partial"):
                    match_counts["partial"] += 1
                elif match_type == "false_negative":
                    match_counts["false_negative"] += 1
                elif match_type == "false_positive":
                    match_counts["false_positive"] += 1
                else:  # no_match_XX%
                    match_counts["no_match"] += 1
                
                results.append({
                    "filename": filename,
                    "climate_id": climate_id,
                    "station_name": station_name,
                    "province": province,
                    "model_output": model_output,
                    "ground_truth": gt_data['verified_extraction'],
                    "is_match": is_match,
                    "match_type": match_type,
                    "time_seconds": elapsed,
                    "error": None,
                })
                
                status = "✓" if is_match else "✗"
                print(f"  {status} Processed in {elapsed:.2f}s ({match_type})")
            else:
                errors += 1
                results.append({
                    "filename": filename,
                    "error": "No output generated",
                })
                print(f"  ERROR: No output generated")
                
        except Exception as e:
            errors += 1
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            results.append({
                "filename": filename,
                "error": error_msg,
            })
            print(f"  ERROR: {error_msg}")
    
    # Save detailed results
    output_file = output_dir / f"{config.name.replace(' ', '_')}_detailed_results.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nSaved detailed results to {output_file}")
    
    # Calculate summary statistics
    matches = sum(1 for r in results if r.get('is_match', False))
    accuracy = (matches / successful * 100) if successful > 0 else 0
    avg_time = (total_time / successful) if successful > 0 else 0
    
    # Calculate more detailed metrics
    true_positives = match_counts["exact"] + match_counts["partial"]
    true_negatives = match_counts["true_negative"]
    false_positives = match_counts["false_positive"]
    false_negatives = match_counts["false_negative"]
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    summary = {
        "model_name": config.name,
        "model_id": config.model_id,
        "model_family": config.family.value,
        "gpu": config.gpu,
        "patch_size": config.patch_size,
        "max_pixels": effective_max_pixels,
        "max_dim": effective_max_dim,
        "total_files": len(test_files),
        "successful": successful,
        "errors": errors,
        "matches": matches,
        "accuracy_percent": round(accuracy, 2),
        "avg_time_seconds": round(avg_time, 2),
        "total_time_seconds": round(total_time, 2),
        "match_breakdown": {
            "exact_matches": match_counts["exact"],
            "partial_matches": match_counts["partial"],
            "true_negatives": match_counts["true_negative"],
            "false_negatives": match_counts["false_negative"],
            "false_positives": match_counts["false_positive"],
            "no_match": match_counts["no_match"],
        },
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }
    
    # Print detailed breakdown
    print(f"\n{'='*60}")
    print("DETAILED ACCURACY BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Exact matches:        {match_counts['exact']:>5}")
    print(f"  Partial matches:      {match_counts['partial']:>5}")
    print(f"  True negatives:       {match_counts['true_negative']:>5}")
    print(f"  False negatives:      {match_counts['false_negative']:>5} (model missed remarks)")
    print(f"  False positives:      {match_counts['false_positive']:>5} (model hallucinated)")
    print(f"  No match:             {match_counts['no_match']:>5}")
    print(f"{'='*60}")
    print(f"  Precision: {precision*100:.1f}%  Recall: {recall*100:.1f}%  F1: {f1*100:.1f}%")
    print(f"{'='*60}")
    
    # Clean up
    del llm
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM models for EC Weather Remarks extraction")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground_truth.csv")
    parser.add_argument("--inventory-2014", type=Path, required=True, help="Path to 2014 station inventory CSV")
    parser.add_argument("--inventory-2022", type=Path, required=True, help="Path to 2022 station inventory CSV")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing test images")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"), help="Output directory")
    parser.add_argument("--models", type=int, nargs="+", default=[1], 
                        help=f"Model IDs to test. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS,
                        help=f"Maximum image pixels (default: {DEFAULT_MAX_IMAGE_PIXELS})")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Maximum image dimension. Default: auto-calculated from max-pixels")
    args = parser.parse_args()
    
    # Calculate max_dim if not specified
    if args.max_dim is None:
        args.max_dim = int(math.ceil(math.sqrt(args.max_pixels * 1.5)))
    
    print(f"\nImage resize settings: max_pixels={args.max_pixels:,}, max_dim={args.max_dim}\n")
    
    # Validate paths
    if not args.ground_truth.exists():
        print(f"ERROR: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    if not args.image_dir.exists():
        print(f"ERROR: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} test cases")
    
    print("Loading station inventories...")
    stations = load_station_inventories(args.inventory_2014, args.inventory_2022)
    print(f"Loaded {len(stations)} station records")
    
    # Run benchmarks
    all_results = []
    for model_id in args.models:
        if model_id not in MODEL_CONFIGS:
            print(f"WARNING: Unknown model ID {model_id}, skipping")
            continue
        
        config = MODEL_CONFIGS[model_id]
        result = run_benchmark(config, ground_truth, stations, args.image_dir, args.output_dir,
                               max_pixels=args.max_pixels, max_dim=args.max_dim)
        all_results.append(result)
    
    # Save summary
    summary_file = args.output_dir / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark complete! Summary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "=" * 140)
    print("BENCHMARK SUMMARY")
    print("=" * 140)
    print(f"{'Model':<30} {'Family':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg Time':>10} {'GPU':<20}")
    print("-" * 140)
    for r in all_results:
        if "error" in r and r.get("error"):
            print(f"{r.get('config', 'Unknown'):<30} {'-':<15} {'ERROR':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':<20}")
        else:
            print(f"{r['model_name']:<30} {r['model_family']:<15} {r['accuracy_percent']:>9.1f}% {r['precision']:>9.1f}% "
                  f"{r['recall']:>9.1f}% {r['f1_score']:>9.1f}% {r['avg_time_seconds']:>9.2f}s {r['gpu']:<20}")
    print("=" * 140)
    print("\nMatch type legend:")
    print("  - exact: Model output exactly matches ground truth")
    print("  - partial_XX%: At least 50% of ground truth lines found in model output")
    print("  - true_negative: Both model and ground truth correctly indicate no remarks")
    print("  - false_negative: Model said NO_REMARKS but ground truth has content (MISS)")
    print("  - false_positive: Model produced content but ground truth says NO_REMARKS (HALLUCINATION)")
    print("  - no_match_XX%: Both have content but insufficient overlap")


if __name__ == "__main__":
    main()
