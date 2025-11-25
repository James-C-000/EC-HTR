#!/usr/bin/env python3
"""
ECCC File Organization Verification Script
Samples moved files and verifies they're in the correct province folders
by cross-referencing with the station CSV files.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple
import json

class OrganizationVerifier:
    def __init__(self, organized_dir: str, station_csv_2014: str, station_csv_2022: str):
        """
        Initialize verifier.
        
        Args:
            organized_dir: Path to PROCESSED_DATA directory with organized files
            station_csv_2014: Path to 2014 station inventory
            station_csv_2022: Path to 2022 station inventory
        """
        self.organized_dir = Path(organized_dir)
        self.station_csv_2014 = station_csv_2014
        self.station_csv_2022 = station_csv_2022
        
        # Province normalization for comparison
        self.folder_to_standard = {
            'British_Columbia': ['BRITISH COLUMBIA', 'BC'],
            'Alberta': ['ALBERTA', 'ALTA'],
            'Saskatchewan': ['SASKATCHEWAN', 'SASK'],
            'Manitoba': ['MANITOBA', 'MAN'],
            'Ontario': ['ONTARIO', 'ONT'],
            'Quebec': ['QUEBEC', 'QUE'],
            'New_Brunswick': ['NEW BRUNSWICK', 'NB'],
            'Nova_Scotia': ['NOVA SCOTIA', 'NS'],
            'Prince_Edward_Island': ['PRINCE EDWARD ISLAND', 'PEI'],
            'Newfoundland_and_Labrador': ['NEWFOUNDLAND', 'NEWFOUNDLAND AND LABRADOR', 'NFLD'],
            'Northwest_Territories': ['NORTHWEST TERRITORIES', 'NWT'],
            'Nunavut': ['NUNAVUT', 'NU'],
            'Yukon': ['YUKON', 'YUKON TERRITORY', 'YT'],
            'South_Dakota': ['SD'],
            'Montana': ['MT'],
            'North_Dakota': ['ND'],
            'Washington': ['WA'],
            'Other': ['OTHR']
        }
        
        # Load station data
        self.station_data = self.load_station_data()
        
    def load_station_data(self) -> Dict:
        """Load station data from CSV files."""
        station_dict = {}
        
        # Helper function for flexible field access
        def get_field(row, field_names):
            for name in field_names:
                if name in row and row[name]:
                    return str(row[name]).strip()
            return ''
        
        # Load 2022 CSV
        if self.station_csv_2022 and Path(self.station_csv_2022).exists():
            with open(self.station_csv_2022, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    climate_id = get_field(row, ['climate_id', 'Climate ID', 'CLIMATE ID'])
                    if climate_id:
                        province = get_field(row, ['Province', 'PROVINCE'])
                        station_dict[climate_id] = {
                            'province': province.upper() if province else '',
                            'name': get_field(row, ['Name', 'Station Name']),
                            'source': '2022'
                        }
        
        # Load 2014 CSV
        if self.station_csv_2014 and Path(self.station_csv_2014).exists():
            with open(self.station_csv_2014, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    climate_id = get_field(row, ['Climate ID', 'climate_id', 'CLIMATE ID'])
                    if climate_id and climate_id not in station_dict:
                        province = get_field(row, ['Province', 'PROVINCE'])
                        station_dict[climate_id] = {
                            'province': province.upper() if province else '',
                            'name': get_field(row, ['Station Name (Current)', 'Name']),
                            'source': '2014'
                        }
        
        print(f"Loaded {len(station_dict)} stations from CSV files")
        return station_dict
    
    def extract_station_id(self, filename: str) -> str:
        """Extract station ID from filename."""
        # Remove extension
        base = os.path.splitext(filename)[0]
        
        # Remove page suffix if present (_A, _B, etc.)
        if len(base) > 2 and base[-2] == '_' and base[-1].isalpha():
            base = base[:-2]
        
        # Split and get station ID (second component)
        parts = base.split('_')
        if len(parts) >= 4:
            return parts[1]
        return None
    
    def verify_file_location(self, file_path: Path) -> Tuple[bool, str]:
        """
        Verify if a file is in the correct folder based on its station ID.
        
        Returns:
            Tuple of (is_correct, message)
        """
        # Get the folder name (province)
        folder_name = file_path.parent.name
        
        # Extract station ID from filename
        station_id = self.extract_station_id(file_path.name)
        if not station_id:
            return False, f"Could not extract station ID from {file_path.name}"
        
        # Look up in station data
        if station_id not in self.station_data:
            # Check if it's a standard 7-digit ID that should be derivable from first digit
            if len(station_id) == 7 and station_id.isdigit():
                # Use first-digit mapping
                first_digit_mapping = {
                    '1': 'British_Columbia',
                    '2': 'Alberta',
                    '3': 'Saskatchewan',
                    '4': 'Manitoba',
                    '5': 'Ontario',
                    '6': 'Quebec',
                    '7': ['New_Brunswick', 'Nova_Scotia', 'Prince_Edward_Island', 
                          'Newfoundland_and_Labrador', 'Atlantic'],
                    '8': ['Northwest_Territories', 'Nunavut', 'Yukon', 'Territories']
                }
                expected = first_digit_mapping.get(station_id[0])
                if isinstance(expected, list):
                    if folder_name in expected:
                        return True, f"Station {station_id} correctly placed (by first digit rule)"
                    else:
                        return False, f"Station {station_id} in {folder_name} but first digit suggests {expected}"
                elif expected == folder_name:
                    return True, f"Station {station_id} correctly placed (by first digit rule)"
                else:
                    return False, f"Station {station_id} in {folder_name} but first digit suggests {expected}"
            else:
                return False, f"Station ID {station_id} not found in CSV files"
        
        # Check if CSV province matches folder
        csv_province = self.station_data[station_id]['province']
        
        # Check if the CSV province matches the folder name
        if folder_name in self.folder_to_standard:
            acceptable_values = self.folder_to_standard[folder_name]
            if csv_province in acceptable_values:
                return True, f"Station {station_id} correctly in {folder_name} (CSV: {csv_province})"
            else:
                return False, f"Station {station_id} in {folder_name} but CSV says {csv_province}"
        
        return False, f"Unknown folder name: {folder_name}"
    
    def sample_and_verify(self, sample_size: int = 100, seed: int = 42) -> Dict:
        """
        Sample files and verify their locations.
        
        Args:
            sample_size: Number of files to sample per province
            seed: Random seed for reproducible sampling
        """
        random.seed(seed)
        results = {
            'total_sampled': 0,
            'correct': 0,
            'incorrect': 0,
            'errors': [],
            'by_province': {}
        }
        
        # Get all province folders
        organized_path = self.organized_dir / 'Organized'
        if not organized_path.exists():
            print(f"Error: {organized_path} does not exist")
            return results
        
        province_folders = [d for d in organized_path.iterdir() if d.is_dir()]
        
        for province_folder in province_folders:
            province_name = province_folder.name
            print(f"\nVerifying {province_name}...")
            
            # Get all files in this province
            files = list(province_folder.glob('*.png')) + list(province_folder.glob('*.tif')) + \
                   list(province_folder.glob('*.tiff')) + list(province_folder.glob('*.jpg'))
            
            if not files:
                print(f"  No files found in {province_name}")
                continue
            
            # Sample files (or take all if fewer than sample_size)
            sample_count = min(sample_size, len(files))
            sampled_files = random.sample(files, sample_count)
            
            province_results = {
                'total': sample_count,
                'correct': 0,
                'incorrect': 0,
                'incorrect_files': []
            }
            
            for file_path in sampled_files:
                is_correct, message = self.verify_file_location(file_path)
                results['total_sampled'] += 1
                
                if is_correct:
                    results['correct'] += 1
                    province_results['correct'] += 1
                else:
                    results['incorrect'] += 1
                    province_results['incorrect'] += 1
                    error_info = {
                        'file': file_path.name,
                        'folder': province_name,
                        'message': message
                    }
                    results['errors'].append(error_info)
                    province_results['incorrect_files'].append(file_path.name)
            
            results['by_province'][province_name] = province_results
            
            # Print province summary
            accuracy = (province_results['correct'] / province_results['total'] * 100) if province_results['total'] > 0 else 0
            print(f"  Sampled: {province_results['total']} files")
            print(f"  Correct: {province_results['correct']} ({accuracy:.1f}%)")
            if province_results['incorrect'] > 0:
                print(f"  Incorrect: {province_results['incorrect']}")
                for fname in province_results['incorrect_files'][:5]:  # Show first 5
                    print(f"    - {fname}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print verification summary."""
        print("\n" + "="*50)
        print("VERIFICATION SUMMARY")
        print("="*50)
        print(f"Total files sampled: {results['total_sampled']}")
        print(f"Correctly placed: {results['correct']}")
        print(f"Incorrectly placed: {results['incorrect']}")
        
        if results['total_sampled'] > 0:
            accuracy = results['correct'] / results['total_sampled'] * 100
            print(f"Overall accuracy: {accuracy:.2f}%")
        
        if results['incorrect'] > 0:
            print(f"\nFirst 10 errors:")
            for error in results['errors'][:10]:
                print(f"  {error['file']} in {error['folder']}")
                print(f"    Issue: {error['message']}")
        
        # Save detailed results
        with open('verification_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\nDetailed results saved to verification_results.json")
        
        print("="*50)

def main():
    """Main entry point."""
    # Configuration
    ORGANIZED_DIR = "/run/media/james/ENV-CA-JC/PROCESSED_DATA"
    STATION_CSV_2014 = "/home/james/PycharmProjects/EC-HTR/EC-Inventory-2014.csv"
    STATION_CSV_2022 = "/home/james/PycharmProjects/EC-HTR/EC-Inventory-2022.csv"
    
    # How many files to sample per province
    SAMPLE_SIZE = 100  # Adjust as needed
    
    # Create verifier
    verifier = OrganizationVerifier(
        organized_dir=ORGANIZED_DIR,
        station_csv_2014=STATION_CSV_2014,
        station_csv_2022=STATION_CSV_2022
    )
    
    # Run verification
    print("Starting organization verification...")
    results = verifier.sample_and_verify(sample_size=SAMPLE_SIZE)
    
    # Print summary
    verifier.print_summary(results)

if __name__ == "__main__":
    main()
