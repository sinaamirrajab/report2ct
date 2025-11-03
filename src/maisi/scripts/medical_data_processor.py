"""
Medical Data Processor
=====================

A comprehensive module for processing JSON files containing medical imaging data.
Includes functions for reading JSON files, extracting metadata from .mha files,
and parsing medical reports to separate findings and impressions.

Requirements:
- SimpleITK (for .mha file processing)
- pathlib
- json
- re
- typing

Usage:
    from medical_data_processor import read_json_file, process_json_data_updated
    
    # Load and process data
    data = read_json_file("prompts.json")
    processed_data = process_json_data_updated(data)
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

def install_package(package: str) -> None:
    """
    Install a package using pip.
    
    Args:
        package (str): Name of the package to install
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def ensure_simpleitk():
    """Ensure SimpleITK is installed and import it."""
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError:
        print("SimpleITK not found. Installing...")
        install_package("SimpleITK")
        import SimpleITK as sitk
        print("SimpleITK installed successfully")
        return sitk

def read_json_file(json_path: str) -> List[Dict]:
    """
    Read JSON file and return the data as a list of dictionaries.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        List[Dict]: List of dictionaries containing the data
        
    Raises:
        FileNotFoundError: If the JSON file is not found
        json.JSONDecodeError: If the JSON file has invalid format
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} entries from {json_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def extract_mha_metadata_sitk(mha_path: str, verbose: bool = True) -> Dict:
    """
    Extract metadata from .mha file using SimpleITK.
    
    Args:
        mha_path (str): Path to the .mha file
        verbose (bool): Whether to print extraction details
        
    Returns:
        Dict: Dictionary containing metadata information with keys:
            - file_path: Original file path
            - size: Image dimensions (width, height, depth)
            - spacing: Physical spacing between pixels
            - origin: Physical coordinate of the first pixel
            - direction: Direction cosines
            - pixel_type: Pixel data type
            - number_of_components: Components per pixel
            - dimension: Image dimension (2D, 3D, etc.)
            - dicom_tags: Dictionary of DICOM metadata tags
    """
    try:
        # Ensure SimpleITK is available
        sitk = ensure_simpleitk()
        
        # Load the .mha file using SimpleITK
        image = sitk.ReadImage(mha_path)
        
        # Extract basic information
        metadata = {
            'file_path': mha_path,
            'size': image.GetSize(),  # (width, height, depth)
            'spacing': image.GetSpacing(),  # Physical spacing between pixels
            'origin': image.GetOrigin(),  # Physical coordinate of the first pixel
            'direction': image.GetDirection(),  # Direction cosines
            'pixel_type': sitk.GetPixelIDValueAsString(image.GetPixelID()),
            'number_of_components': image.GetNumberOfComponentsPerPixel(),
            'dimension': image.GetDimension(),
        }
        
        # Get all metadata tags
        metadata_keys = image.GetMetaDataKeys()
        metadata['dicom_tags'] = {}
        for key in metadata_keys:
            try:
                value = image.GetMetaData(key)
                metadata['dicom_tags'][key] = value
            except:
                metadata['dicom_tags'][key] = "Could not read"
        
        if verbose:
            print(f"Successfully extracted metadata from {mha_path}")
            print(f"Image size: {metadata['size']}")
            print(f"Pixel spacing: {metadata['spacing']}")
            print(f"Pixel type: {metadata['pixel_type']}")
        
        return metadata
        
    except Exception as e:
        if verbose:
            print(f"Error extracting metadata from {mha_path}: {e}")
        return {}

def parse_medical_report(report_text: str) -> Dict[str, str]:
    """
    Parse medical report text to separate findings and impressions.
    
    Args:
        report_text (str): The full medical report text
        
    Returns:
        Dict[str, str]: Dictionary with keys:
            - 'findings': Text from the Findings section
            - 'impression': Text from the Impression section  
            - 'full_text': Original complete text
            
    Features:
        - Case-insensitive section detection
        - Handles various report formats
        - Fallback for reports without clear structure
    """
    # Initialize result dictionary
    result = {
        'findings': '',
        'impression': '',
        'full_text': report_text
    }
    
    # Clean the text
    clean_text = report_text.strip()
    
    # Find the "Findings:" section
    findings_match = re.search(r'Findings:\s*(.*?)(?=Impression:|$)', clean_text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        result['findings'] = findings_match.group(1).strip()
    
    # Find the "Impression:" section
    impression_match = re.search(r'Impression:\s*(.*?)$', clean_text, re.IGNORECASE | re.DOTALL)
    if impression_match:
        result['impression'] = impression_match.group(1).strip()
    
    # If no clear sections found, try alternative patterns
    if not result['findings'] and not result['impression']:
        # Check if the text contains both sections without clear markers
        text_lower = clean_text.lower()
        
        # Find findings section (might be implicit)
        findings_start = text_lower.find('findings:')
        impression_start = text_lower.find('impression:')
        
        if findings_start != -1 and impression_start != -1:
            # Both sections found
            result['findings'] = clean_text[findings_start + 9:impression_start].strip()
            result['impression'] = clean_text[impression_start + 11:].strip()
        elif impression_start != -1:
            # Only impression found, everything before is findings
            result['findings'] = clean_text[:impression_start].strip()
            result['impression'] = clean_text[impression_start + 11:].strip()
        else:
            # No clear structure, put everything in findings
            result['findings'] = clean_text
    
    return result

def process_json_data_updated(json_data: List[Dict], base_path: str = "", verbose: bool = True) -> List[Dict]:
    """
    Process the JSON data by extracting metadata from .mha files using SimpleITK and parsing reports.
    
    Args:
        json_data (List[Dict]): List of dictionaries from JSON file
        base_path (str): Base path where .mha files are located
        verbose (bool): Whether to print processing details
        
    Returns:
        List[Dict]: Processed data with metadata and parsed reports.
                   Each entry contains original data plus:
                   - 'metadata': Dictionary with .mha file metadata
                   - 'parsed_report': Dictionary with separated findings and impression
    """
    processed_data = []
    
    for idx, entry in enumerate(json_data):
        if verbose:
            print(f"\nProcessing entry {idx + 1}/{len(json_data)}...")
        
        processed_entry = entry.copy()
        
        # Extract .mha metadata if file exists using SimpleITK
        if 'input_image_name' in entry:
            mha_path = Path(base_path) / entry['input_image_name']
            if mha_path.exists():
                metadata = extract_mha_metadata_sitk(str(mha_path), verbose=verbose)
                processed_entry['metadata'] = metadata
            else:
                if verbose:
                    print(f"Warning: .mha file not found: {mha_path}")
                processed_entry['metadata'] = {}
        
        # Parse the medical report
        if 'report' in entry:
            parsed_report = parse_medical_report(entry['report'])
            processed_entry['parsed_report'] = parsed_report
        
        processed_data.append(processed_entry)
    
    return processed_data

def process_all_data(json_path: str, mha_base_path: str = "", verbose: bool = True) -> List[Dict]:
    """
    Complete pipeline to process JSON file with .mha metadata and report parsing.
    
    Args:
        json_path (str): Path to the JSON file
        mha_base_path (str): Base path where .mha files are located
        verbose (bool): Whether to print processing details
    
    Returns:
        List[Dict]: List of processed entries with metadata and parsed reports
    """
    # Read JSON data
    if verbose:
        print("Step 1: Reading JSON file...")
    json_data = read_json_file(json_path)
    
    if not json_data:
        return []
    
    # Process each entry
    if verbose:
        print("Step 2: Processing entries...")
    processed_data = process_json_data_updated(json_data, mha_base_path, verbose=verbose)
    
    # Display summary
    if verbose:
        print(f"\nProcessing complete! Summary:")
        print(f"- Total entries: {len(processed_data)}")
        
        entries_with_metadata = sum(1 for entry in processed_data if entry.get('metadata'))
        entries_with_reports = sum(1 for entry in processed_data if entry.get('parsed_report'))
        
        print(f"- Entries with metadata: {entries_with_metadata}")
        print(f"- Entries with parsed reports: {entries_with_reports}")
    
    return processed_data

def print_entry_summary(entry: Dict, max_findings_chars: int = 200) -> None:
    """
    Print a formatted summary of a processed entry.
    
    Args:
        entry (Dict): Processed entry dictionary
        max_findings_chars (int): Maximum characters to show for findings
    """
    print(f"Image filename: {entry.get('input_image_name', 'N/A')}")
    
    if 'parsed_report' in entry:
        findings = entry['parsed_report']['findings']
        impression = entry['parsed_report']['impression']
        
        print(f"\nFindings ({len(findings)} chars):")
        print("-" * 30)
        if len(findings) > max_findings_chars:
            print(findings[:max_findings_chars] + "...")
        else:
            print(findings)
        
        print(f"\nImpression ({len(impression)} chars):")
        print("-" * 30)
        print(impression)
    
    if 'metadata' in entry and entry['metadata']:
        print(f"\nMetadata Summary:")
        print("-" * 30)
        metadata = entry['metadata']
        print(f"Size (W×H×D): {metadata.get('size', 'N/A')}")
        print(f"Spacing: {metadata.get('spacing', 'N/A')}")
        print(f"Pixel type: {metadata.get('pixel_type', 'N/A')}")
        print(f"Dimensions: {metadata.get('dimension', 'N/A')}")
        print(f"Number of DICOM tags: {len(metadata.get('dicom_tags', {}))}")
    else:
        print("\nMetadata: Not available")

def save_processed_data(processed_data: List[Dict], output_path: str) -> bool:
    """
    Save processed data to a JSON file.
    
    Args:
        processed_data (List[Dict]): Processed data to save
        output_path (str): Path where to save the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert tuples to lists for JSON serialization
        serializable_data = []
        for entry in processed_data:
            serializable_entry = entry.copy()
            if 'metadata' in serializable_entry and serializable_entry['metadata']:
                metadata = serializable_entry['metadata']
                # Convert tuples to lists
                for key in ['size', 'spacing', 'origin', 'direction']:
                    if key in metadata and isinstance(metadata[key], tuple):
                        metadata[key] = list(metadata[key])
            serializable_data.append(serializable_entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved processed data to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example usage of the module
    json_file = "prompts.json"
    
    if Path(json_file).exists():
        print("Running example processing...")
        processed_data = process_all_data(json_file)
        
        if processed_data:
            print("\n" + "="*60)
            print("EXAMPLE OUTPUT:")
            print("="*60)
            print_entry_summary(processed_data[0])
            
            # Save processed data
            save_processed_data(processed_data, "processed_data.json")
    else:
        print(f"Example JSON file '{json_file}' not found.")
        print("\nTo use this module:")
        print("1. Import the functions: from medical_data_processor import *")
        print("2. Load your data: data = read_json_file('your_file.json')")
        print("3. Process it: processed = process_json_data_updated(data)")
