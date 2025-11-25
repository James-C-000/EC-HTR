#!/usr/bin/env python3
"""
EC Historical Weather Data Organization Script - Enhanced Version
Sorts weather observation files by province/territory/state based on Climate ID and year.
Handles both standard 7-digit numeric IDs and non-standard alphanumeric IDs found in CSV files.
Keeps both front and back (_A) files together for later analysis.
"""

import os
import re
import shutil
import logging
from pathlib import Path
from datetime import datetime
import csv
import json
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_organization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ECFileOrganizer:
    """Organizes EC weather observation files by province and year."""
    
    def __init__(self, source_dir: str, output_dir: str, station_csv_2014: str = None, 
                 station_csv_2022: str = None, dry_run: bool = False):
        """
        Initialize the organizer.
        
        Args:
            source_dir: Root directory containing the EC data
            output_dir: Directory where organized files will be stored
            station_csv_2014: Path to 2014 station inventory CSV
            station_csv_2022: Path to 2022 station inventory CSV
            dry_run: If True, only simulate file moves without actually moving
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.station_csv_2014 = station_csv_2014
        self.station_csv_2022 = station_csv_2022
        self.dry_run = dry_run
        
        # Climate ID to Province mapping (first digit for standard 7-digit IDs)
        self.province_mapping = {
            '1': 'British_Columbia',
            '2': 'Alberta',
            '3': 'Saskatchewan',
            '4': 'Manitoba',
            '5': 'Ontario',
            '6': 'Quebec',
            '7': 'Atlantic',  # NB, NS, PEI, NFLD - will use station data to disambiguate
            '8': 'Territories'  # NWT, Nunavut, Yukon - will use station data to disambiguate
        }
        
        # Statistics tracking
        all_provinces = list(self.province_mapping.values()) + [
            'New_Brunswick', 'Nova_Scotia', 'Prince_Edward_Island', 
            'Newfoundland_and_Labrador', 'Northwest_Territories', 'Nunavut', 'Yukon',
            'South_Dakota', 'Montana', 'North_Dakota', 'Washington', 'Other'
        ]
        self.stats = {
            'total_files': 0,
            'moved_files': 0,
            'post_1939': 0,
            'pre_1940': 0,
            'manual_qc': 0,
            'non_standard_ids_resolved': 0,
            'skipped_existing': 0,
            'errors': 0,
            'by_province': {prov: 0 for prov in all_provinces}
        }
        
        # Keep track of problem files
        self.problem_files = []
        
        # Keep track of successfully resolved non-standard IDs
        self.resolved_non_standard = []
        
        # Load station data if available
        self.station_data = self.load_station_data()
        
    def _get_csv_field(self, row: dict, field_names: list) -> str:
        """Get field value from CSV row, trying multiple possible column names."""
        for name in field_names:
            if name in row:
                val = row.get(name)
                if val is not None:
                    return str(val)
        return ''
    
    def load_station_data(self) -> Dict:
        """Load station metadata from CSV files."""
        station_dict = {}
        
        # Province name normalization mapping (all uppercase for comparison)
        province_normalize = {
            # 2014 format (abbreviated)
            'BC': 'British_Columbia',
            'ALTA': 'Alberta',
            'SASK': 'Saskatchewan',
            'MAN': 'Manitoba',
            'ONT': 'Ontario',
            'QUE': 'Quebec',
            'NB': 'New_Brunswick',
            'NS': 'Nova_Scotia',
            'PEI': 'Prince_Edward_Island',
            'NFLD': 'Newfoundland_and_Labrador',
            'NWT': 'Northwest_Territories',
            'NU': 'Nunavut',
            'YT': 'Yukon',
            # US States (current and potential)
            'SD': 'South_Dakota',
            'MT': 'Montana',
            'ND': 'North_Dakota',
            'WA': 'Washington',
            # Other
            'OTHR': 'Other',
            # 2022 format (full names)
            'BRITISH COLUMBIA': 'British_Columbia',
            'ALBERTA': 'Alberta',
            'SASKATCHEWAN': 'Saskatchewan',
            'MANITOBA': 'Manitoba',
            'ONTARIO': 'Ontario',
            'QUEBEC': 'Quebec',
            'NEW BRUNSWICK': 'New_Brunswick',
            'NOVA SCOTIA': 'Nova_Scotia',
            'PRINCE EDWARD ISLAND': 'Prince_Edward_Island',
            'NEWFOUNDLAND': 'Newfoundland_and_Labrador',
            'NEWFOUNDLAND AND LABRADOR': 'Newfoundland_and_Labrador',
            'NORTHWEST TERRITORIES': 'Northwest_Territories',
            'NUNAVUT': 'Nunavut',
            'YUKON': 'Yukon',
            'YUKON TERRITORY': 'Yukon'
        }
        
        # Try loading 2022 data first (more recent)
        if self.station_csv_2022:
            try:
                with open(self.station_csv_2022, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try multiple possible column names for climate_id
                        climate_id = self._get_csv_field(row, ['climate_id', 'Climate ID', 'CLIMATE ID']).strip()
                        if climate_id:  # Store ALL climate IDs
                            province_raw = self._get_csv_field(row, ['Province', 'PROVINCE']).strip()
                            # Uppercase for normalization lookup
                            province = province_normalize.get(province_raw.upper())
                            if not province:
                                logger.debug(f"Unknown province format in 2022 CSV: '{province_raw}' for station {climate_id}")
                                province = province_raw.replace(' ', '_')
                            station_dict[climate_id] = {
                                'name': self._get_csv_field(row, ['Name', 'Station Name', 'NAME']),
                                'province': province,
                                'source': '2022'
                            }
                logger.info(f"Loaded {len(station_dict)} stations from 2022 CSV")
            except Exception as e:
                logger.warning(f"Could not load 2022 station CSV: {e}")
        
        # Load 2014 data and fill in gaps
        if self.station_csv_2014:
            try:
                with open(self.station_csv_2014, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try multiple possible column names
                        climate_id = self._get_csv_field(row, ['Climate ID', 'climate_id', 'CLIMATE ID']).strip()
                        if climate_id and climate_id not in station_dict:  # Store ALL climate IDs
                            province_raw = self._get_csv_field(row, ['Province', 'PROVINCE']).strip()
                            # Uppercase for normalization lookup
                            province = province_normalize.get(province_raw.upper())
                            if not province:
                                logger.debug(f"Unknown province format in 2014 CSV: '{province_raw}' for station {climate_id}")
                                province = province_raw.replace(' ', '_')
                            station_dict[climate_id] = {
                                'name': self._get_csv_field(row, ['Station Name (Current)', 'Name', 'Station Name']),
                                'province': province,
                                'source': '2014'
                            }
                logger.info(f"Total stations in database: {len(station_dict)}")
                
                # Count how many are standard vs non-standard
                standard_count = sum(1 for id in station_dict.keys() 
                                   if len(id) == 7 and id.isdigit())
                non_standard_count = len(station_dict) - standard_count
                logger.info(f"  Standard 7-digit IDs: {standard_count}")
                logger.info(f"  Non-standard IDs: {non_standard_count}")
                
            except Exception as e:
                logger.warning(f"Could not load 2014 station CSV: {e}")
                
        return station_dict
    
    def parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], bool]:
        """
        Parse EC filename to extract components.
        
        Expected format: 9904_1010774_1932_11_A.png
        Where:
            - 9904: Form type
            - 1010774: Climate station ID (usually 7 digits, but can be alphanumeric)
            - 1932: Year
            - 11: Month
            - A-Z: Optional page indicator (multi-page documents)
            
        Returns:
            Tuple of (form_type, station_id, year, month, is_non_standard)
            Returns None values if parsing fails
        """
        # Remove extension
        name_parts = os.path.splitext(filename)[0]
        
        # Remove page suffix if present (can be _A through _Z for multi-page docs)
        if len(name_parts) > 2 and name_parts[-2] == '_' and name_parts[-1].isalpha() and len(name_parts[-1]) == 1:
            name_parts = name_parts[:-2]  # Remove '_X' suffix for parsing
        
        # Split by underscores
        parts = name_parts.split('_')
        
        # Check if we have the expected 4 parts: form_stationid_year_month
        if len(parts) != 4:
            return None, None, None, None, False
        
        try:
            form_type = parts[0]
            station_id = parts[1]
            year = int(parts[2])
            month = int(parts[3])
            
            # Validate month is 1-12
            if not (1 <= month <= 12):
                logger.debug(f"Invalid month {month} in filename: {filename}")
                return None, None, None, None, False
            
            # Optionally validate year (reasonable bounds for weather data)
            if not (1840 <= year <= 2025):
                logger.debug(f"Suspicious year {year} in filename: {filename}")
            
            # Check if station_id is standard (7-digit numeric) or non-standard
            is_standard = len(station_id) == 7 and station_id.isdigit()
            
            # Return the station_id regardless of format
            return form_type, station_id, year, month, not is_standard
            
        except (ValueError, IndexError):
            # Year or month couldn't be parsed as integers
            return None, None, None, None, False
    
    def get_province(self, station_id: str) -> Optional[str]:
        """
        Determine province from station ID.
        
        Args:
            station_id: Climate station ID (can be 7-digit numeric or alphanumeric)
            
        Returns:
            Province name or None if cannot determine
        """
        if not station_id:
            return None
            
        # First check station database for exact match (works for both standard and non-standard IDs)
        if station_id in self.station_data:
            province = self.station_data[station_id].get('province')
            if province:
                # Log if we resolved a non-standard ID
                if not (len(station_id) == 7 and station_id.isdigit()):
                    logger.debug(f"Resolved non-standard ID '{station_id}' to province '{province}'")
                return province
        
        # For standard 7-digit IDs, fall back to first digit mapping
        if len(station_id) == 7 and station_id.isdigit():
            first_digit = station_id[0]
            mapped_province = self.province_mapping.get(first_digit)
            
            # If we get Atlantic or Territories, try to be more specific from station data
            if mapped_province in ['Atlantic', 'Territories'] and station_id in self.station_data:
                specific_province = self.station_data[station_id].get('province')
                if specific_province and specific_province not in ['Atlantic', 'Territories']:
                    return specific_province
            
            return mapped_province
        
        # Could not determine province
        return None
    
    def create_output_structure(self):
        """Create the output directory structure."""
        directories = [
            self.output_dir / 'Organized',
            self.output_dir / 'Irrelevant_Post1939',
            self.output_dir / 'Manual_QC'
        ]
        
        # Add all province/territory/state directories
        all_locations = [
            'British_Columbia', 'Alberta', 'Saskatchewan', 'Manitoba', 'Ontario', 'Quebec',
            'New_Brunswick', 'Nova_Scotia', 'Prince_Edward_Island', 'Newfoundland_and_Labrador',
            'Northwest_Territories', 'Nunavut', 'Yukon', 'Atlantic', 'Territories',
            'South_Dakota', 'Montana', 'Other'
        ]
        
        for location in all_locations:
            directories.append(self.output_dir / 'Organized' / location)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created output directory structure in {self.output_dir}")
    
    def move_file(self, source_file: Path, destination_dir: Path) -> bool:
        """
        Move a file to the destination directory.
        
        Args:
            source_file: Path to source file
            destination_dir: Destination directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_file = destination_dir / source_file.name
            
            # Check if file already exists at destination
            if destination_file.exists():
                logger.debug(f"File already exists at destination: {destination_file}")
                self.stats['skipped_existing'] += 1
                return False
            
            # In dry run mode, just log what would happen
            if self.dry_run:
                logger.debug(f"DRY RUN: Would move {source_file} -> {destination_file}")
                return True
            
            # Move the file
            shutil.move(str(source_file), str(destination_file))
            return True
            
        except Exception as e:
            logger.error(f"Error moving {source_file}: {e}")
            self.stats['errors'] += 1
            self.problem_files.append({
                'file': str(source_file),
                'error': str(e)
            })
            return False
    
    def process_file(self, file_path: Path) -> str:
        """
        Process a single file and determine its destination.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Destination category ('province_name', 'post_1939', 'manual_qc')
        """
        filename = file_path.name
        
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
            return 'skip'
        
        # Parse filename
        form_type, station_id, year, month, is_non_standard = self.parse_filename(filename)
        
        # Check if parsing failed completely
        if not year or not station_id:
            self.stats['manual_qc'] += 1
            destination = self.output_dir / 'Manual_QC'
            if self.move_file(file_path, destination):
                self.stats['moved_files'] += 1
            logger.warning(f"Could not parse filename: {filename}")
            return 'manual_qc'
        
        # Check year
        if year > 1939:
            self.stats['post_1939'] += 1
            destination = self.output_dir / 'Irrelevant_Post1939'
            if self.move_file(file_path, destination):
                self.stats['moved_files'] += 1
            return 'post_1939'
        
        # Pre-1940 file - determine province
        province = self.get_province(station_id)
        
        if not province:
            # Could not determine province even with CSV lookup
            self.stats['manual_qc'] += 1
            destination = self.output_dir / 'Manual_QC'
            if self.move_file(file_path, destination):
                self.stats['moved_files'] += 1
            logger.warning(f"Could not determine province for station {station_id}: {filename}")
            return 'manual_qc'
        
        # Successfully determined province
        if is_non_standard:
            self.stats['non_standard_ids_resolved'] += 1
            self.resolved_non_standard.append({
                'filename': filename,
                'station_id': station_id,
                'province': province,
                'station_name': self.station_data.get(station_id, {}).get('name', 'Unknown')
            })
        
        # Move to appropriate province folder
        self.stats['pre_1940'] += 1
        if province in self.stats['by_province']:
            self.stats['by_province'][province] += 1
        else:
            logger.warning(f"Unexpected province '{province}' - adding to stats")
            self.stats['by_province'][province] = 1
            
        destination = self.output_dir / 'Organized' / province
        if self.move_file(file_path, destination):
            self.stats['moved_files'] += 1
        return province
    
    def process_directory(self, directory: Path):
        """
        Recursively process all files in a directory.
        
        Args:
            directory: Directory to process
        """
        # Get all files recursively
        all_files = []
        for ext in ['*.png', '*.PNG', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            all_files.extend(directory.rglob(ext))
        
        self.stats['total_files'] = len(all_files)
        logger.info(f"Found {self.stats['total_files']} image files to process")
        
        # Process each file with progress bar
        with tqdm(total=len(all_files), desc="Organizing files") as pbar:
            for file_path in all_files:
                # Skip XML directory files
                if 'XML' in file_path.parts:
                    pbar.update(1)
                    continue
                    
                self.process_file(file_path)
                pbar.update(1)
    
    def run(self):
        """Main execution method."""
        logger.info(f"Starting EC file organization")
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.dry_run:
            logger.info("MODE: DRY RUN - No files will actually be moved")
        
        # Create output structure
        self.create_output_structure()
        
        # Process all files
        self.process_directory(self.source_dir)
        
        # Save statistics and problem files
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save processing results to files."""
        # Save statistics as CSV
        stats_file = self.output_dir / 'organization_stats.csv'
        
        # Prepare data for CSV
        stats_rows = []
        
        # Add summary statistics
        stats_rows.append(['Summary Statistics', ''])
        stats_rows.append(['Total files found', self.stats['total_files']])
        stats_rows.append(['Files moved', self.stats['moved_files']])
        stats_rows.append(['Post-1939 files (irrelevant)', self.stats['post_1939']])
        stats_rows.append(['Pre-1940 files (organized)', self.stats['pre_1940']])
        stats_rows.append(['Non-standard IDs resolved', self.stats['non_standard_ids_resolved']])
        stats_rows.append(['Files needing manual QC', self.stats['manual_qc']])
        stats_rows.append(['Errors encountered', self.stats['errors']])
        stats_rows.append(['', ''])
        
        # Add breakdown by location
        stats_rows.append(['Files by Province/Territory/State', 'Count'])
        for province, count in sorted(self.stats['by_province'].items()):
            if count > 0:
                stats_rows.append([province, count])
        
        # Write CSV
        with open(stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(stats_rows)
        
        # Save resolved non-standard IDs
        if self.resolved_non_standard:
            resolved_file = self.output_dir / 'resolved_non_standard_ids.csv'
            with open(resolved_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'station_id', 'province', 'station_name'])
                writer.writeheader()
                writer.writerows(self.resolved_non_standard)
            logger.info(f"  - Resolved non-standard IDs: resolved_non_standard_ids.csv")
        
        # Save problem files as CSV if any exist
        if self.problem_files:
            problem_file = self.output_dir / 'problem_files.csv'
            with open(problem_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'Error'])
                for problem in self.problem_files:
                    writer.writerow([problem['file'], problem['error']])
            logger.info(f"  - Problem files: problem_files.csv")
        
        # Also keep a JSON backup for programmatic access
        stats_json = self.output_dir / 'organization_stats.json'
        with open(stats_json, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  - Main statistics: organization_stats.csv")
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*50)
        print("ORGANIZATION SUMMARY" + (" (DRY RUN)" if self.dry_run else ""))
        print("="*50)
        print(f"Total files found: {self.stats['total_files']:,}")
        print(f"Files {'would be' if self.dry_run else ''} moved: {self.stats['moved_files']:,}")
        print(f"Post-1939 files (irrelevant): {self.stats['post_1939']:,}")
        print(f"Pre-1940 files (organized): {self.stats['pre_1940']:,}")
        print(f"Non-standard IDs resolved via CSV: {self.stats['non_standard_ids_resolved']:,}")
        print(f"Files needing manual QC: {self.stats['manual_qc']:,}")
        print(f"Errors encountered: {self.stats['errors']:,}")
        
        print("\nFiles by Location:")
        
        # Canadian locations
        canadian_locations = [
            'British_Columbia', 'Alberta', 'Saskatchewan', 'Manitoba', 'Ontario', 'Quebec',
            'New_Brunswick', 'Nova_Scotia', 'Prince_Edward_Island', 'Newfoundland_and_Labrador',
            'Northwest_Territories', 'Nunavut', 'Yukon', 'Atlantic', 'Territories'
        ]
        has_canadian = False
        for location in canadian_locations:
            if location in self.stats['by_province'] and self.stats['by_province'][location] > 0:
                if not has_canadian:
                    print("  Canadian Provinces/Territories:")
                    has_canadian = True
                print(f"    {location}: {self.stats['by_province'][location]:,}")
        
        # US States
        us_states = ['Montana', 'South_Dakota']
        has_us = False
        for state in us_states:
            if state in self.stats['by_province'] and self.stats['by_province'][state] > 0:
                if not has_us:
                    print("  US States:")
                    has_us = True
                print(f"    {state}: {self.stats['by_province'][state]:,}")
        
        # Other
        if 'Other' in self.stats['by_province'] and self.stats['by_province']['Other'] > 0:
            print("  Other Locations:")
            print(f"    Other: {self.stats['by_province']['Other']:,}")
        
        if self.dry_run:
            print("\nNote: This was a DRY RUN - no files were actually moved")
        print("="*50)

def main():
    """Main entry point."""
    # Configuration
    SOURCE_DIR = "/run/media/james/ENV-CA-JC/RAW_DATA"  # EC drive path
    OUTPUT_DIR = "/run/media/james/ENV-CA-JC/PROCESSED_DATA"  # Where organized files will go
    
    STATION_CSV_2014 = "/home/james/PycharmProjects/EC-HTR/EC-Inventory-2014.csv"
    STATION_CSV_2022 = "/home/james/PycharmProjects/EC-HTR/EC-Inventory-2022.csv"
    
    # IMPORTANT: Set to False when ready to actually move files
    DRY_RUN = False  # Change to False for actual file organization
    
    # Create organizer and run
    organizer = ECFileOrganizer(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        station_csv_2014=STATION_CSV_2014,
        station_csv_2022=STATION_CSV_2022,
        dry_run=DRY_RUN
    )
    
    if DRY_RUN:
        logger.info("="*50)
        logger.info("RUNNING IN DRY RUN MODE - NO FILES WILL BE MOVED")
        logger.info("="*50)
    
    # Optional: Test on a single batch first
    # Uncomment below to test on just one directory:
    # test_dir = Path(SOURCE_DIR) / "PNG_B10_2_Data20210201-20210331" / "1_BC_5"
    # organizer.process_directory(test_dir)
    # return
    
    organizer.run()

if __name__ == "__main__":
    main()
