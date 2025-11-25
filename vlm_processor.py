#!/usr/bin/env python3
"""
EC Historical Weather Observation Scanner - VLM Processing Pipeline

This script processes scanned weather observation forms from Environment 
Canada's historical collection. It extracts qualitative remarks using a 
Vision Language Model (VLM) with location-aware prompts.

Features:
- Loads Climate ID lookup tables into memory for fast resolution
- Parses filenames to extract metadata (Climate ID, year, month)
- Generates tailored prompts with human-readable location context
- Implements robust checkpointing for crash recovery
- Flags non-standard Climate IDs for manual review

Author: James C. Caldwell
"""

import csv
import json
import os
import re
import shutil
import sys
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator
from enum import Enum
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ec_processor.log')
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status codes for file processing results."""
    SUCCESS = "success"
    SKIPPED_POST_1940 = "skipped_post_1940"
    SKIPPED_BACKSIDE = "skipped_backside"
    CLIMATE_ID_NOT_FOUND = "climate_id_not_found"
    NON_STANDARD_ID = "non_standard_climate_id"
    PARSE_ERROR = "parse_error"
    VLM_ERROR = "vlm_error"
    ALREADY_PROCESSED = "already_processed"


@dataclass
class StationInfo:
    """Weather station information from inventory."""
    climate_id: str
    name: str
    province: str
    province_full: str  # Full province name for prompts
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    first_year: Optional[int] = None
    last_year: Optional[int] = None
    source: str = "unknown"  # Which CSV the data came from


@dataclass
class FileMetadata:
    """Parsed metadata from a scan filename."""
    filepath: str
    filename: str
    form_code: str  # Usually "9904"
    climate_id: str
    year: int
    month: int
    is_backside: bool
    is_standard_id: bool
    parse_error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    filepath: str
    status: ProcessingStatus
    climate_id: Optional[str] = None
    station_name: Optional[str] = None
    location: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    extracted_text: Optional[str] = None
    remarks: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ClimateIDLookup:
    """
    In-memory lookup table for Climate ID to station information.
    
    Loads both 2014 and 2022 inventory CSVs into memory for fast lookups.
    Prioritizes 2022 data but falls back to 2014 for historical stations.
    """
    
    # Province code to full name mapping
    PROVINCE_MAP = {
        'BC': 'British Columbia',
        'AB': 'Alberta', 
        'SK': 'Saskatchewan',
        'MB': 'Manitoba',
        'ON': 'Ontario',
        'QC': 'Quebec',
        'NB': 'New Brunswick',
        'NS': 'Nova Scotia',
        'PE': 'Prince Edward Island',
        'NL': 'Newfoundland and Labrador',
        'YT': 'Yukon',
        'NT': 'Northwest Territories',
        'NU': 'Nunavut',
        # Also handle full names from 2022 CSV
        'BRITISH COLUMBIA': 'British Columbia',
        'ALBERTA': 'Alberta',
        'SASKATCHEWAN': 'Saskatchewan',
        'MANITOBA': 'Manitoba',
        'ONTARIO': 'Ontario',
        'QUEBEC': 'Quebec',
        'NEW BRUNSWICK': 'New Brunswick',
        'NOVA SCOTIA': 'Nova Scotia',
        'PRINCE EDWARD ISLAND': 'Prince Edward Island',
        'NEWFOUNDLAND AND LABRADOR': 'Newfoundland and Labrador',
        'NEWFOUNDLAND': 'Newfoundland and Labrador',
        'YUKON': 'Yukon',
        'NORTHWEST TERRITORIES': 'Northwest Territories',
        'NUNAVUT': 'Nunavut',
    }
    
    def __init__(self):
        self._lookup: Dict[str, StationInfo] = {}
        self._load_count_2014 = 0
        self._load_count_2022 = 0
        
    def load_2022_inventory(self, filepath: str) -> int:
        """
        Load the 2022 inventory CSV.
        
        Expected columns: fid,Name,Province,climate_id,station_id,wmo_id,tc_id,
                         latitude,Longitude,...,first_year,last_year,...
        """
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                climate_id = row.get('climate_id', '').strip()
                if not climate_id:
                    continue
                    
                # Normalize climate_id (uppercase)
                climate_id = climate_id.upper()
                
                province_raw = row.get('Province', '').strip()
                province_full = self.PROVINCE_MAP.get(province_raw.upper(), province_raw)
                
                # Parse numeric fields safely
                lat = self._safe_float(row.get('latitude'))
                lon = self._safe_float(row.get('Longitude'))
                first_year = self._safe_int(row.get('first_year'))
                last_year = self._safe_int(row.get('last_year'))
                
                station = StationInfo(
                    climate_id=climate_id,
                    name=row.get('Name', '').strip(),
                    province=province_raw,
                    province_full=province_full,
                    latitude=lat,
                    longitude=lon,
                    first_year=first_year,
                    last_year=last_year,
                    source='2022'
                )
                
                # 2022 data takes priority
                self._lookup[climate_id] = station
                count += 1
                
        self._load_count_2022 = count
        logger.info(f"Loaded {count} stations from 2022 inventory")
        return count
    
    def load_2014_inventory(self, filepath: str) -> int:
        """
        Load the 2014 inventory CSV.
        
        Expected columns: Internal Database Identifier,Climate ID,Station Name (Current),
                         Province,...
        """
        count = 0
        new_entries = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                climate_id = row.get('Climate ID', '').strip()
                if not climate_id:
                    continue
                    
                # Normalize climate_id (uppercase, strip whitespace)
                climate_id = climate_id.upper().strip()
                
                # Skip if already have 2022 data (2022 takes priority)
                if climate_id in self._lookup:
                    count += 1
                    continue
                
                province_raw = row.get('Province', '').strip()
                province_full = self.PROVINCE_MAP.get(province_raw.upper(), province_raw)
                
                # Parse coordinates
                lat = self._safe_float(row.get('LAT. (decimal degrees)'))
                lon = self._safe_float(row.get('LONG. (decimal degrees)'))
                
                station = StationInfo(
                    climate_id=climate_id,
                    name=row.get('Station Name (Current)', '').strip(),
                    province=province_raw,
                    province_full=province_full,
                    latitude=lat,
                    longitude=lon,
                    source='2014'
                )
                
                self._lookup[climate_id] = station
                count += 1
                new_entries += 1
                
        self._load_count_2014 = count
        logger.info(f"Loaded {count} stations from 2014 inventory ({new_entries} new entries)")
        return count
    
    def get_station(self, climate_id: str) -> Optional[StationInfo]:
        """Look up station info by Climate ID."""
        return self._lookup.get(climate_id.upper().strip())
    
    def get_location_string(self, climate_id: str) -> Optional[str]:
        """
        Get a human-readable location string for VLM prompts.
        
        Returns format like: "Station Name, Province, Canada"
        """
        station = self.get_station(climate_id)
        if not station:
            return None
            
        parts = []
        if station.name:
            parts.append(station.name.title())
        if station.province_full:
            parts.append(station.province_full)
        parts.append("Canada")
        
        return ", ".join(parts)
    
    @property
    def total_stations(self) -> int:
        return len(self._lookup)
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely parse a float value."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely parse an integer value."""
        if value is None or value == '':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


class FilenameParser:
    """
    Parser for EC scan filenames.
    
    Standard format: 9904_CLIMATEID_YEAR_MONTH[_A].png
    Examples:
        - 9904_1010774_1932_11.png (front side)
        - 9904_1010774_1932_11_A.png (back side)
        - 9904_610ML02_1984_12.png (non-standard ID)
    """
    
    # Standard Climate ID pattern: 7 alphanumeric characters
    # Per EC documentation, should be 7 digits, but I've seen alphanumeric
    STANDARD_ID_PATTERN = re.compile(r'^[0-9]{7}$')
    
    # Filename pattern: FORMCODE_CLIMATEID_YEAR_MONTH[_A].ext
    FILENAME_PATTERN = re.compile(
        r'^(\d{4})_([A-Za-z0-9]+)_(\d{4})_(\d{2})(_A)?\.(?:png|tif|tiff|jpg|jpeg)$',
        re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, filepath: str) -> FileMetadata:
        """Parse a filename and extract metadata."""
        filename = os.path.basename(filepath)
        
        match = cls.FILENAME_PATTERN.match(filename)
        if not match:
            return FileMetadata(
                filepath=filepath,
                filename=filename,
                form_code="",
                climate_id="",
                year=0,
                month=0,
                is_backside=False,
                is_standard_id=False,
                parse_error=f"Filename does not match expected pattern: {filename}"
            )
        
        form_code = match.group(1)
        climate_id = match.group(2).upper()
        year = int(match.group(3))
        month = int(match.group(4))
        is_backside = match.group(5) is not None
        
        # Check if this is a standard 7-digit Climate ID
        is_standard = bool(cls.STANDARD_ID_PATTERN.match(climate_id))
        
        return FileMetadata(
            filepath=filepath,
            filename=filename,
            form_code=form_code,
            climate_id=climate_id,
            year=year,
            month=month,
            is_backside=is_backside,
            is_standard_id=is_standard
        )


class CheckpointManager:
    """
    Manages checkpointing for crash recovery.
    
    Features:
    - Atomic checkpoint writes (write to temp file, then rename)
    - Tracks processed files by hash to handle file moves
    - Periodic auto-save
    - Safe resume from last checkpoint
    """
    
    def __init__(self, checkpoint_dir: str, checkpoint_interval: int = 100):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_interval: Number of files between auto-saves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.checkpoint_temp = self.checkpoint_dir / "checkpoint.json.tmp"
        self.processed_log = self.checkpoint_dir / "processed_files.log"
        self.results_file = self.checkpoint_dir / "results.jsonl"
        self.results_temp = self.checkpoint_dir / "results.jsonl.tmp"
        
        self.checkpoint_interval = checkpoint_interval
        self._processed_hashes: set = set()
        self._processed_count = 0
        self._results_buffer: List[ProcessingResult] = []
        self._last_checkpoint_time = datetime.now()
        
        # Load existing checkpoint if available
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint state from disk."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self._processed_hashes = set(data.get('processed_hashes', []))
                    self._processed_count = data.get('processed_count', 0)
                    logger.info(f"Resumed from checkpoint: {self._processed_count} files already processed")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
                self._processed_hashes = set()
                self._processed_count = 0
    
    def _save_checkpoint(self):
        """Save checkpoint state to disk atomically."""
        checkpoint_data = {
            'processed_hashes': list(self._processed_hashes),
            'processed_count': self._processed_count,
            'last_save': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Write to temp file first
        with open(self.checkpoint_temp, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Atomic rename
        shutil.move(str(self.checkpoint_temp), str(self.checkpoint_file))
        
        # Also flush results buffer
        self._flush_results()
        
        self._last_checkpoint_time = datetime.now()
        logger.debug(f"Checkpoint saved: {self._processed_count} files")
    
    def _flush_results(self):
        """Flush results buffer to disk."""
        if not self._results_buffer:
            return
            
        # Append results to file atomically
        with open(self.results_file, 'a') as f:
            for result in self._results_buffer:
                # Convert to dict and handle enum serialization
                result_dict = asdict(result)
                result_dict['status'] = result.status.value  # Convert enum to string
                f.write(json.dumps(result_dict) + '\n')
        
        self._results_buffer = []
    
    def get_file_hash(self, filepath: str) -> str:
        """Generate a unique hash for a file (using path + size + mtime)."""
        stat = os.stat(filepath)
        hash_input = f"{filepath}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def is_processed(self, filepath: str) -> bool:
        """Check if a file has already been processed."""
        file_hash = self.get_file_hash(filepath)
        return file_hash in self._processed_hashes
    
    def mark_processed(self, filepath: str, result: ProcessingResult):
        """Mark a file as processed and store its result."""
        file_hash = self.get_file_hash(filepath)
        self._processed_hashes.add(file_hash)
        self._processed_count += 1
        self._results_buffer.append(result)
        
        # Auto-save checkpoint periodically
        if self._processed_count % self.checkpoint_interval == 0:
            self._save_checkpoint()
            logger.info(f"Progress: {self._processed_count} files processed (checkpoint saved)")
    
    def save(self):
        """Force save checkpoint (call at end of processing)."""
        self._save_checkpoint()
    
    @property
    def processed_count(self) -> int:
        return self._processed_count


class PromptGenerator:
    """
    Generates tailored prompts for the VLM based on file metadata.
    
    Uses zero-shot prompting (per project notes: Qwen3-VL is very suggestible,
    so specific examples cause hallucinations).
    """
    
    # Month names for natural language prompts
    MONTHS = [
        '', 'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    @classmethod
    def generate_extraction_prompt(
        cls,
        location: Optional[str],
        year: int,
        month: int,
        include_context: bool = True
    ) -> str:
        """
        Generate a prompt for text extraction from a weather observation form.
        
        Args:
            location: Human-readable location string (e.g., "Montreal, Quebec, Canada")
            year: Year of the observation
            month: Month of the observation (1-12)
            include_context: Whether to include temporal/location context
        """
        month_name = cls.MONTHS[month] if 1 <= month <= 12 else f"Month {month}"
        
        # Build context string if location is known
        context_parts = []
        if include_context:
            if location:
                context_parts.append(f"This is a weather observation form from {location}")
            context_parts.append(f"recorded in {month_name} {year}")
        
        context_str = ", ".join(context_parts) + "." if context_parts else ""
        
        # Zero-shot prompt - avoid specific examples per project notes
        prompt = f"""Extract all handwritten text from this historical weather observation form. {context_str}

Instructions:
1. Focus on handwritten entries, especially those in the "Remarks" or "General Observations" sections
2. Transcribe the handwritten text as accurately as possible
3. Preserve the original wording and spelling
4. Ignore printed form headers, column labels, and pre-printed text
5. Ignore administrative stamps and form numbers
6. If text is crossed out or illegible, indicate this with [crossed out] or [illegible]
7. Ignore any symbolic notation or shorthand codes

Return only the transcribed handwritten text, separated by newlines if there are multiple entries."""

        return prompt
    
    @classmethod
    def generate_remarks_prompt(
        cls,
        location: Optional[str],
        year: int,
        month: int
    ) -> str:
        """
        Generate a prompt specifically for extracting qualitative remarks.
        
        This is an alternative prompt focused specifically on the types of 
        remarks we are interested in (phenological info, weather events, etc.)
        """
        month_name = cls.MONTHS[month] if 1 <= month <= 12 else f"Month {month}"
        
        location_str = f"from {location} " if location else ""
        
        prompt = f"""This is a historical Canadian weather observation form {location_str}from {month_name} {year}.

Extract any handwritten qualitative remarks or observations about:
- Weather conditions and events (storms, unusual weather, etc.)
- Natural phenomena (first frost, ice breakup, first robin, etc.)
- Environmental observations

Transcribe only the handwritten notes and remarks. Ignore:
- Numerical measurements and data
- Printed form text and labels
- Administrative stamps
- Symbolic codes

Return the transcribed remarks, or "No remarks found" if there are none."""

        return prompt


class ECProcessor:
    """
    Main processor for EC weather observation scans.
    
    Coordinates file discovery, parsing, VLM inference, and result storage.
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
    
    def __init__(
        self,
        inventory_2022_path: str,
        inventory_2014_path: str,
        checkpoint_dir: str,
        cutoff_year: int = 1940
    ):
        """
        Initialize the processor.
        
        Args:
            inventory_2022_path: Path to 2022 inventory CSV
            inventory_2014_path: Path to 2014 inventory CSV
            checkpoint_dir: Directory for checkpoints and results
            cutoff_year: Only process files before this year (exclusive)
        """
        self.cutoff_year = cutoff_year
        
        # Initialize components
        logger.info("Loading Climate ID lookup tables...")
        self.lookup = ClimateIDLookup()
        self.lookup.load_2022_inventory(inventory_2022_path)
        self.lookup.load_2014_inventory(inventory_2014_path)
        logger.info(f"Total stations loaded: {self.lookup.total_stations}")
        
        self.checkpoint = CheckpointManager(checkpoint_dir)
        self.prompt_generator = PromptGenerator()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'skipped_post_cutoff': 0,
            'skipped_already_done': 0,
            'climate_id_not_found': 0,
            'non_standard_ids': 0,
            'parse_errors': 0,
            'vlm_errors': 0,
        }
    
    def discover_files(self, root_dir: str) -> Generator[str, None, None]:
        """
        Recursively discover all scan files in a directory.
        
        Yields filepaths sorted by (year, month) for chronological processing.
        """
        files_with_dates = []
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(dirpath, filename)
                    
                    # Parse to get date for sorting
                    metadata = FilenameParser.parse(filepath)
                    if metadata.parse_error is None:
                        files_with_dates.append((metadata.year, metadata.month, filepath))
                    else:
                        # Still include unparseable files, sort them last
                        files_with_dates.append((9999, 99, filepath))
        
        # Sort by year, month
        files_with_dates.sort()
        
        for _, _, filepath in files_with_dates:
            yield filepath
    
    def should_process(self, metadata: FileMetadata) -> Tuple[bool, ProcessingStatus]:
        """
        Determine if a file should be processed based on metadata.
        
        Returns:
            Tuple of (should_process, reason_if_not)
        """
        if metadata.parse_error:
            return False, ProcessingStatus.PARSE_ERROR
        
        if metadata.year >= self.cutoff_year:
            return False, ProcessingStatus.SKIPPED_POST_1940
        
        return True, ProcessingStatus.SUCCESS
    
    def process_file(
        self,
        filepath: str,
        vlm_callback=None  # Callback function for VLM inference
    ) -> ProcessingResult:
        """
        Process a single file.
        
        Args:
            filepath: Path to the scan file
            vlm_callback: Optional callback function that takes (image_path, prompt) 
                         and returns extracted text. If None, only metadata is extracted.
        
        Returns:
            ProcessingResult with extraction results
        """
        start_time = datetime.now()
        
        # Check if already processed
        if self.checkpoint.is_processed(filepath):
            self.stats['skipped_already_done'] += 1
            return ProcessingResult(
                filepath=filepath,
                status=ProcessingStatus.ALREADY_PROCESSED
            )
        
        # Parse filename
        metadata = FilenameParser.parse(filepath)
        
        # Check if we should process this file
        should_process, reason = self.should_process(metadata)
        if not should_process:
            if reason == ProcessingStatus.SKIPPED_POST_1940:
                self.stats['skipped_post_cutoff'] += 1
            elif reason == ProcessingStatus.PARSE_ERROR:
                self.stats['parse_errors'] += 1
            
            return ProcessingResult(
                filepath=filepath,
                status=reason,
                error_message=metadata.parse_error
            )
        
        # Look up station info
        station = self.lookup.get_station(metadata.climate_id)
        location = self.lookup.get_location_string(metadata.climate_id)
        
        if station is None:
            self.stats['climate_id_not_found'] += 1
            if not metadata.is_standard_id:
                self.stats['non_standard_ids'] += 1
                status = ProcessingStatus.NON_STANDARD_ID
            else:
                status = ProcessingStatus.CLIMATE_ID_NOT_FOUND
            
            # Still process, but with limited context
            logger.debug(f"Climate ID not found: {metadata.climate_id} in {filepath}")
        else:
            status = ProcessingStatus.SUCCESS
        
        # Generate prompt
        prompt = self.prompt_generator.generate_extraction_prompt(
            location=location,
            year=metadata.year,
            month=metadata.month
        )
        
        # Run VLM inference if callback provided
        extracted_text = None
        if vlm_callback:
            try:
                extracted_text = vlm_callback(filepath, prompt)
            except Exception as e:
                self.stats['vlm_errors'] += 1
                logger.error(f"VLM error on {filepath}: {e}")
                return ProcessingResult(
                    filepath=filepath,
                    status=ProcessingStatus.VLM_ERROR,
                    climate_id=metadata.climate_id,
                    year=metadata.year,
                    month=metadata.month,
                    error_message=str(e)
                )
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        self.stats['processed'] += 1
        
        return ProcessingResult(
            filepath=filepath,
            status=status,
            climate_id=metadata.climate_id,
            station_name=station.name if station else None,
            location=location,
            year=metadata.year,
            month=metadata.month,
            extracted_text=extracted_text,
            processing_time_ms=processing_time
        )
    
    def process_directory(
        self,
        root_dir: str,
        vlm_callback=None,
        max_files: Optional[int] = None
    ) -> Dict:
        """
        Process all files in a directory.
        
        Args:
            root_dir: Root directory containing scans
            vlm_callback: Optional VLM inference callback
            max_files: Maximum number of files to process (for testing)
        
        Returns:
            Statistics dictionary
        """
        import time
        
        logger.info(f"Starting processing of {root_dir}")
        logger.info(f"Cutoff year: {self.cutoff_year}")
        if vlm_callback is None:
            logger.warning("No VLM callback provided - extracted_text will be None")
        
        file_count = 0
        processing_times = []
        start_time = time.time()
        
        try:
            for filepath in self.discover_files(root_dir):
                self.stats['total_files'] += 1
                
                file_start = time.time()
                result = self.process_file(filepath, vlm_callback)
                file_elapsed = time.time() - file_start
                
                # Mark as processed (unless already done)
                if result.status != ProcessingStatus.ALREADY_PROCESSED:
                    self.checkpoint.mark_processed(filepath, result)
                    file_count += 1
                    
                    # Track timing for VLM-processed files
                    if result.status == ProcessingStatus.SUCCESS and vlm_callback:
                        processing_times.append(file_elapsed)
                        
                        # Log progress with timing every 10 files
                        if len(processing_times) % 10 == 0:
                            avg_time = sum(processing_times) / len(processing_times)
                            remaining = max_files - file_count if max_files else "unknown"
                            if isinstance(remaining, int):
                                eta_seconds = remaining * avg_time
                                eta_str = f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.0f} sec"
                            else:
                                eta_str = "calculating..."
                            logger.info(f"Processed {len(processing_times)} files | Avg: {avg_time:.2f}s/file | ETA: {eta_str}")
                
                # Check max files limit
                if max_files and file_count >= max_files:
                    logger.info(f"Reached max files limit ({max_files})")
                    break
                    
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        finally:
            # Always save checkpoint on exit
            self.checkpoint.save()
            
            total_time = time.time() - start_time
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                logger.info(f"Final: Processed {file_count} new files in {total_time:.1f}s (avg {avg_time:.2f}s/file)")
            else:
                logger.info(f"Final checkpoint saved. Processed {file_count} new files in {total_time:.1f}s")
        
        return self.stats
    
    def generate_report(self) -> str:
        """Generate a summary report of processing."""
        report = [
            "=" * 60,
            "EC Processing Report",
            "=" * 60,
            f"Total files discovered: {self.stats['total_files']}",
            f"Files processed: {self.stats['processed']}",
            f"Skipped (post-{self.cutoff_year}): {self.stats['skipped_post_cutoff']}",
            f"Skipped (already done): {self.stats['skipped_already_done']}",
            f"Climate ID not found: {self.stats['climate_id_not_found']}",
            f"Non-standard IDs: {self.stats['non_standard_ids']}",
            f"Parse errors: {self.stats['parse_errors']}",
            f"VLM errors: {self.stats['vlm_errors']}",
            "=" * 60,
        ]
        return "\n".join(report)


def create_ollama_vlm_callback(model_name: str = "qwen2.5-vl:7b", host: str = "http://localhost:11434"):
    """
    Create a VLM callback using Ollama.
    
    Args:
        model_name: Ollama model to use (e.g., 'qwen2.5-vl:7b', 'qwen3-vl:8b')
        host: Ollama server URL
    
    Returns:
        Callback function for VLM inference
    """
    try:
        import ollama
    except ImportError:
        logger.error("Ollama package not installed. Run: pip install ollama")
        raise ImportError("ollama package required. Install with: pip install ollama")
    
    client = ollama.Client(host=host)
    
    # Verify model is available
    try:
        models = client.list()
        model_names = [m['name'] for m in models.get('models', [])]
        base_model = model_name.split(':')[0]
        if not any(base_model in name for name in model_names):
            available = ', '.join(model_names) if model_names else 'none'
            logger.warning(f"Model '{model_name}' not found. Available models: {available}")
            logger.warning(f"Pull the model with: ollama pull {model_name}")
    except Exception as e:
        logger.warning(f"Could not verify model availability: {e}")
    
    def callback(image_path: str, prompt: str) -> str:
        """Extract text from image using Ollama VLM."""
        import time
        start_time = time.time()
        
        try:
            response = client.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }],
                options={
                    'temperature': 0.1,  # Low temperature for factual extraction
                    'num_predict': 1024,  # Max tokens
                }
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Ollama inference took {elapsed:.2f}s for {os.path.basename(image_path)}")
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama inference failed for {image_path}: {e}")
            raise
    
    return callback


def create_dummy_vlm_callback():
    """
    Create a dummy VLM callback for testing (no actual inference).
    
    Use --dummy flag to enable this instead of real VLM.
    """
    logger.warning("Using DUMMY VLM callback - no actual text extraction will occur!")
    
    def callback(image_path: str, prompt: str) -> str:
        return f"[DUMMY - no VLM called] {os.path.basename(image_path)}"
    
    return callback


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process EC historical weather observation scans'
    )
    parser.add_argument(
        'input_dir',
        help='Root directory containing scan files'
    )
    parser.add_argument(
        '--inventory-2022',
        default='EC_Historical_Weather_Station_inventory_2022.csv',
        help='Path to 2022 inventory CSV'
    )
    parser.add_argument(
        '--inventory-2014',
        default='EC_Historical_Weather_Station_inventory_2014_01.csv',
        help='Path to 2014 inventory CSV'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='./checkpoints',
        help='Directory for checkpoints and results'
    )
    parser.add_argument(
        '--cutoff-year',
        type=int,
        default=1940,
        help='Only process files before this year (default: 1940)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only parse and validate files, skip VLM entirely (no extraction)'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Use dummy VLM callback (for testing pipeline without GPU)'
    )
    parser.add_argument(
        '--model',
        default='qwen2.5-vl:7b',
        help='Ollama model to use (default: qwen2.5-vl:7b)'
    )
    parser.add_argument(
        '--ollama-host',
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = ECProcessor(
        #inventory_2022_path=args.inventory_2022,
        #inventory_2014_path=args.inventory_2014,
        inventory_2022_path="/home/james/PycharmProjects/EC-HTR/EC-Inventory-2022.csv",
        inventory_2014_path="/home/james/PycharmProjects/EC-HTR/EC-Inventory-2014.csv",
        checkpoint_dir=args.checkpoint_dir,
        cutoff_year=args.cutoff_year
    )
    
    # Set up VLM callback
    if args.dry_run:
        vlm_callback = None
        logger.info("Dry run mode: skipping VLM inference entirely")
    elif args.dummy:
        vlm_callback = create_dummy_vlm_callback()
    else:
        logger.info(f"Initializing Ollama VLM with model: {args.model}")
        vlm_callback = create_ollama_vlm_callback(
            model_name=args.model,
            host=args.ollama_host
        )
    
    # Process files
    stats = processor.process_directory(
        root_dir=args.input_dir,
        vlm_callback=vlm_callback,
        max_files=args.max_files
    )
    
    # Print report
    print(processor.generate_report())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
