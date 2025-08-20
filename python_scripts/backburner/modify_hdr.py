"""modify_hdr.py: Modifies .hdr of datacube to display correct wavelengths and units."""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"

import os
import re # Regular expressions

def extract_wavelength(filename: str):
    """Extract numeric wavelength value ending in 'nm' from filename.
        Ex. "image_50nm.tif", would find "50 nm" or "50nm", and extract 50.
    
    Args:
        filename (str): The filename to parse the wavelength from"""
    
    match = re.search(r'(\d+)\s*nm', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def modify_hdr_files(hdr_dir: str):
    """Adds and removes lines from ENVI cube .hdr, to display band unit and wavlength.

    Args:
        hdr_dir (str): Directory to hdr file. Assumes only one .hdr in directory.
    """
    
    hdr_files = [file for file in os.listdir(hdr_dir) if file.lower().endswith('.hdr')]
    wavelengths = {}

    for hdr_file in hdr_files:
        base = hdr_file.replace('.hdr', '')
        wavelength = extract_wavelength(base)
        if wavelength:
            wavelengths[hdr_file] = float(wavelength)

    # Sort by wavelength in ascending order (low->high)
    sorted_items = sorted(wavelengths.items(), key=lambda x: x[1])
    wavelengths = [f"{wavelengths[f]:.1f}" for f, _ in sorted_items]

    for hdr_file, _ in sorted_items:
        hdr_path = os.path.join(hdr_dir, hdr_file)

        # Read existing lines
        with open(hdr_path, 'r') as f:
            lines = f.readlines()
            
        # Find index of 'byte order = 0' line
        byte_order_index = next(
            (i for i, line in enumerate(lines) if line.strip().lower().startswith('byte order')), None
        )
    
        if byte_order_index is None:
            print(f"Warning: 'byte order' line not found in {hdr_file}. Skipping.")
            continue
    
        # Delete everything after 'byte order = 0'
        new_lines = lines[:byte_order_index + 1]

        # Append new metadata
        new_lines.append('wavelength units = nanometers\n')
        new_lines.append('wavelength = {' + ', '.join(wavelengths) + '}\n')

        # Write back modified header
        with open(hdr_path, 'w') as f:
            f.writelines(lines)


