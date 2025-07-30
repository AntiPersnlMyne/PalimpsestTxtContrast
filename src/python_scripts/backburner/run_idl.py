"""run_idl.py: Run IDL scripts through Python. Saves output to data/output as ENVI .dat raster."""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Prototype", "Development", "Production"

import subprocess
from typing import Any

# Magic string
idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"

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
    arg_str = ' '.join([f'"{str(arg)}"' for arg in all_args])

    
    # Format the IDL command, compiles dynamically
    idl_command = (
        f'"{idl_exe}" '
        f'-IDL_PATH src/IDL_scripts '
        f'-e {idl_script} '
        f'-args {arg_str}'
    )


    try:
        result = subprocess.run(
            idl_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except subprocess.CalledProcessError as e:
        print(f"-- Error calling IDL script: '{idl_script}'")
        print("- STDOUT -\n", e.stdout)
        print("- STDERR -\n", e.stderr)
        raise
    except Exception as e:
        print(f"-- Misc Error calling IDL script: {idl_script}\n{e}")
        