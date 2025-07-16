"""run_idl.py: Run IDL scripts through Python. Saved output to data/output as TIFF."""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", or "Production". 

import subprocess
from typing import Any

def run_idl_script(idl_script: str, src_dir: str, dst_dir: str, args: list[Any] = []):
    """
    Run an IDL .pro script with required source and destination directories,
    and optional extra arguments.

    Parameters:
        idl_script (str): Name of the IDL script (without .pro extension).
        src_dir (str): Source directory path (passed as first arg).
        dst_dir (str): Destination directory path (passed as second arg).
        args (List[Any]): Any additional arguments required by the IDL script.
                          Order matters and is the user's responsibility.
    """
    
    # Build argument string
    all_args = [str(src_dir), str(dst_dir)] + [str(arg) for arg in args]
    arg_str = '", "'.join(all_args)
    
     # Format the IDL call to pass arguments
    idl_command = f'idl -e "{idl_script}" -args "{arg_str}"'
    
    try:
        subprocess.run(
            idl_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except subprocess.CalledProcessError as e:
        print(f"-- Error running IDL script: {idl_script}")
        print("- STDOUT -\n", e.stdout)
        print("- STDERR -\n", e.stderr)
        raise