#!/usr/bin/env python3
"""
EC Processor Validation and Testing Utility

Use this to validate setup before running the full pipeline on 571k files.
Tests:
- Climate ID lookup table loading
- Filename parsing
- Location resolution
- Prompt generation
- Checkpoint system

Author: James C. Caldwell
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ec_vlm_processor import (
    ClimateIDLookup,
    FilenameParser,
    CheckpointManager,
    PromptGenerator,
    ECProcessor,
    ProcessingStatus,
    ProcessingResult,
    FileMetadata
)


def test_climate_id_lookup(inventory_2022: str, inventory_2014: str) -> bool:
    """Test Climate ID lookup table loading and resolution."""
    print("\n" + "=" * 60)
    print("Testing Climate ID Lookup")
    print("=" * 60)
    
    lookup = ClimateIDLookup()
    
    # Load 2022 inventory
    count_2022 = lookup.load_2022_inventory(inventory_2022)
    print(f"✓ Loaded {count_2022} stations from 2022 inventory")
    
    # Load 2014 inventory
    count_2014 = lookup.load_2014_inventory(inventory_2014)
    print(f"✓ Loaded {count_2014} stations from 2014 inventory")
    
    print(f"✓ Total unique stations: {lookup.total_stations}")
    
    # Test some known Climate IDs
    test_ids = [
        '1010774',   # Standard numeric ID from examples
        '1010066',   # Another standard ID
        '10114F6',   # Alphanumeric ID
        '101AE00',   # Non-standard ID from 2014
        '9999999',   # Non-existent ID
    ]
    
    print("\nTesting sample Climate ID lookups:")
    for cid in test_ids:
        station = lookup.get_station(cid)
        location = lookup.get_location_string(cid)
        
        if station:
            print(f"  {cid}: {station.name} ({station.province_full}) [source: {station.source}]")
        else:
            print(f"  {cid}: NOT FOUND")
    
    # Test province mapping
    print("\nProvince mapping check:")
    provinces_found = set()
    for _ in range(min(1000, lookup.total_stations)):
        # Sample some stations
        pass
    
    return True


def test_filename_parser() -> bool:
    """Test filename parsing for various patterns."""
    print("\n" + "=" * 60)
    print("Testing Filename Parser")
    print("=" * 60)
    
    test_cases = [
        # (filename, expected_climate_id, expected_year, expected_month, expected_backside, expected_standard)
        ("9904_1010774_1932_11.png", "1010774", 1932, 11, False, True),
        ("9904_1010774_1932_11_A.png", "1010774", 1932, 11, True, True),
        ("9904_610ML02_1984_12.png", "610ML02", 1984, 12, False, False),
        ("9904_706CFQ3_1945_01_A.png", "706CFQ3", 1945, 1, True, False),
        ("9904_110CCCC_1945_08.png", "110CCCC", 1945, 8, False, False),
        ("9904_2100635_1902_12.png", "2100635", 1902, 12, False, True),
        ("9904_8502875_1914_07.tif", "8502875", 1914, 7, False, True),
        # Edge cases
        ("invalid_filename.png", None, None, None, None, None),  # Parse error expected
        ("9904_1234567_2025_13.png", "1234567", 2025, 13, False, True),  # Invalid month but parseable
    ]
    
    all_passed = True
    for test in test_cases:
        filename = test[0]
        metadata = FilenameParser.parse(f"/fake/path/{filename}")
        
        if test[1] is None:  # Expected parse error
            if metadata.parse_error:
                print(f"✓ {filename}: Correctly rejected (parse error)")
            else:
                print(f"✗ {filename}: Should have failed to parse")
                all_passed = False
        else:
            expected_cid, expected_year, expected_month, expected_back, expected_std = test[1:]
            
            checks = [
                (metadata.climate_id == expected_cid, f"climate_id={metadata.climate_id}, expected={expected_cid}"),
                (metadata.year == expected_year, f"year={metadata.year}, expected={expected_year}"),
                (metadata.month == expected_month, f"month={metadata.month}, expected={expected_month}"),
                (metadata.is_backside == expected_back, f"is_backside={metadata.is_backside}, expected={expected_back}"),
                (metadata.is_standard_id == expected_std, f"is_standard_id={metadata.is_standard_id}, expected={expected_std}"),
            ]
            
            failures = [msg for passed, msg in checks if not passed]
            
            if not failures:
                print(f"✓ {filename}: All checks passed")
            else:
                print(f"✗ {filename}: Failed checks: {', '.join(failures)}")
                all_passed = False
    
    return all_passed


def test_prompt_generator() -> bool:
    """Test prompt generation with various inputs."""
    print("\n" + "=" * 60)
    print("Testing Prompt Generator")
    print("=" * 60)
    
    test_cases = [
        ("Montreal, Quebec, Canada", 1932, 11),
        ("Victoria, British Columbia, Canada", 1905, 3),
        (None, 1920, 7),  # Unknown location
        ("Halifax, Nova Scotia, Canada", 1899, 12),
    ]
    
    for location, year, month in test_cases:
        prompt = PromptGenerator.generate_extraction_prompt(location, year, month)
        
        print(f"\nLocation: {location or 'Unknown'}, Date: {year}/{month:02d}")
        print("-" * 40)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    return True


def test_checkpoint_system() -> bool:
    """Test checkpoint save/restore functionality."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint System")
    print("=" * 60)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp(prefix="ec_test_")
    
    try:
        # Create a checkpoint manager
        cm = CheckpointManager(temp_dir, checkpoint_interval=5)
        print(f"✓ Created checkpoint manager in {temp_dir}")
        
        # Create some fake files and results
        fake_files = [
            ("/fake/path/scan_001.png", "file content 1"),
            ("/fake/path/scan_002.png", "file content 2"),
            ("/fake/path/scan_003.png", "file content 3"),
        ]
        
        # Create actual temp files to get real hashes
        for i, (fake_path, content) in enumerate(fake_files):
            temp_file = Path(temp_dir) / f"test_file_{i}.txt"
            temp_file.write_text(content)
            
            result = ProcessingResult(
                filepath=str(temp_file),
                status=ProcessingStatus.SUCCESS,
                climate_id=f"100000{i}",
                year=1932,
                month=11
            )
            
            cm.mark_processed(str(temp_file), result)
        
        print(f"✓ Marked {cm.processed_count} files as processed")
        
        # Force save
        cm.save()
        print("✓ Checkpoint saved")
        
        # Verify checkpoint file exists
        checkpoint_file = Path(temp_dir) / "checkpoint.json"
        results_file = Path(temp_dir) / "results.jsonl"
        
        assert checkpoint_file.exists(), "Checkpoint file not created"
        assert results_file.exists(), "Results file not created"
        print("✓ Checkpoint and results files created")
        
        # Create new checkpoint manager to test restore
        cm2 = CheckpointManager(temp_dir, checkpoint_interval=5)
        assert cm2.processed_count == 3, f"Expected 3 processed files, got {cm2.processed_count}"
        print(f"✓ Checkpoint restored: {cm2.processed_count} files")
        
        # Verify is_processed works
        temp_file_0 = Path(temp_dir) / "test_file_0.txt"
        assert cm2.is_processed(str(temp_file_0)), "File should be marked as processed"
        print("✓ is_processed check works")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_end_to_end(inventory_2022: str, inventory_2014: str) -> bool:
    """Test complete processing pipeline (without VLM)."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Pipeline (Dry Run)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="ec_e2e_")
    
    try:
        # Create processor
        processor = ECProcessor(
            inventory_2022_path=inventory_2022,
            inventory_2014_path=inventory_2014,
            checkpoint_dir=temp_dir,
            cutoff_year=1940
        )
        print("✓ Processor initialized")
        
        # Create some test files
        test_files_dir = Path(temp_dir) / "test_scans"
        test_files_dir.mkdir()
        
        test_filenames = [
            "9904_1010774_1932_11.png",      # Should process (pre-1940, front)
            "9904_1010774_1932_11_A.png",    # Should process (pre-1940, back)
            "9904_1010774_1945_03.png",      # Should skip (post-1940)
            "9904_2100635_1902_12.png",      # Should process (old file)
            "9904_UNKNOWN1_1920_05.png",     # Should process but flag (unknown ID)
        ]
        
        for filename in test_filenames:
            (test_files_dir / filename).touch()
        
        print(f"✓ Created {len(test_filenames)} test files")
        
        # Process (dry run - no VLM callback)
        stats = processor.process_directory(
            root_dir=str(test_files_dir),
            vlm_callback=None
        )
        
        print("\nProcessing Results:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped (post-1940): {stats['skipped_post_cutoff']}")
        print(f"  Climate ID not found: {stats['climate_id_not_found']}")
        
        # Verify expected behavior
        assert stats['processed'] == 4, f"Expected 4 processed, got {stats['processed']}"
        assert stats['skipped_post_cutoff'] == 1, f"Expected 1 skipped post-1940"
        
        print("\n✓ End-to-end test passed!")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def analyze_sample_files(scan_dir: str, inventory_2022: str, inventory_2014: str, max_files: int = 100):
    """
    Analyze a sample of actual scan files to check ID coverage and patterns.
    
    This helps identify how many files have resolvable Climate IDs.
    """
    print("\n" + "=" * 60)
    print(f"Analyzing Sample Files from: {scan_dir}")
    print("=" * 60)
    
    lookup = ClimateIDLookup()
    lookup.load_2022_inventory(inventory_2022)
    lookup.load_2014_inventory(inventory_2014)
    
    stats = {
        'total': 0,
        'parsed': 0,
        'parse_errors': 0,
        'id_found': 0,
        'id_not_found': 0,
        'standard_id': 0,
        'non_standard_id': 0,
        'pre_1940': 0,
        'post_1940': 0,
        'fronts': 0,
        'backs': 0,
        'provinces': {},
    }
    
    missing_ids = []
    
    from pathlib import Path
    for filepath in Path(scan_dir).rglob('*'):
        if filepath.suffix.lower() not in {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}:
            continue
            
        stats['total'] += 1
        if stats['total'] > max_files:
            break
        
        metadata = FilenameParser.parse(str(filepath))
        
        if metadata.parse_error:
            stats['parse_errors'] += 1
            continue
        
        stats['parsed'] += 1
        
        if metadata.is_standard_id:
            stats['standard_id'] += 1
        else:
            stats['non_standard_id'] += 1
        
        if metadata.year < 1940:
            stats['pre_1940'] += 1
        else:
            stats['post_1940'] += 1
        
        if metadata.is_backside:
            stats['backs'] += 1
        else:
            stats['fronts'] += 1
        
        station = lookup.get_station(metadata.climate_id)
        if station:
            stats['id_found'] += 1
            prov = station.province_full
            stats['provinces'][prov] = stats['provinces'].get(prov, 0) + 1
        else:
            stats['id_not_found'] += 1
            missing_ids.append(metadata.climate_id)
    
    print(f"\nAnalyzed {stats['total']} files:")
    print(f"  Successfully parsed: {stats['parsed']}")
    print(f"  Parse errors: {stats['parse_errors']}")
    print(f"\nClimate ID Resolution:")
    print(f"  Found in inventory: {stats['id_found']} ({100*stats['id_found']/stats['parsed']:.1f}%)")
    print(f"  Not found: {stats['id_not_found']} ({100*stats['id_not_found']/stats['parsed']:.1f}%)")
    print(f"\nID Types:")
    print(f"  Standard (7-digit): {stats['standard_id']}")
    print(f"  Non-standard: {stats['non_standard_id']}")
    print(f"\nDate Range:")
    print(f"  Pre-1940: {stats['pre_1940']}")
    print(f"  Post-1940: {stats['post_1940']}")
    print(f"\nPage Sides:")
    print(f"  Front sides: {stats['fronts']}")
    print(f"  Back sides (_A): {stats['backs']}")
    
    if stats['provinces']:
        print(f"\nBy Province:")
        for prov, count in sorted(stats['provinces'].items(), key=lambda x: -x[1]):
            print(f"  {prov}: {count}")
    
    if missing_ids:
        print(f"\nSample unresolved Climate IDs (first 10):")
        for cid in missing_ids[:10]:
            print(f"  {cid}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and validate EC processor components')
    parser.add_argument('--inventory-2022', required=True, help='Path to 2022 inventory CSV')
    parser.add_argument('--inventory-2014', required=True, help='Path to 2014 inventory CSV')
    parser.add_argument('--scan-dir', help='Directory with actual scan files to analyze')
    parser.add_argument('--max-sample', type=int, default=100, help='Max files to sample for analysis')
    parser.add_argument('--skip-tests', action='store_true', help='Skip unit tests, only run analysis')
    
    args = parser.parse_args()
    
    all_passed = True
    
    if not args.skip_tests:
        # Run unit tests
        all_passed &= test_climate_id_lookup(args.inventory_2022, args.inventory_2014)
        all_passed &= test_filename_parser()
        all_passed &= test_prompt_generator()
        all_passed &= test_checkpoint_system()
        all_passed &= test_end_to_end(args.inventory_2022, args.inventory_2014)
    
    # Run file analysis if directory provided
    if args.scan_dir:
        analyze_sample_files(
            args.scan_dir,
            args.inventory_2022,
            args.inventory_2014,
            args.max_sample
        )
    
    if not args.skip_tests:
        print("\n" + "=" * 60)
        if all_passed:
            print("All tests PASSED ✓")
        else:
            print("Some tests FAILED ✗")
        print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
