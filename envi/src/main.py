import subprocess
import os
import sys
import numpy

# --- Set environment variables temporarily within the script ---
os.environ['IDL_DIR'] = r'C:\Program Files\NV5\ENVI61\IDL91'
os.environ['ENVI_DIR'] = r'C:\Program Files\NV5\ENVI61' # Only if you use ENVI functions

# --- CRITICAL FIX HERE: Add IDL's bin directory to the PATH ---
# Get the current PATH environment variable
current_path = os.environ.get('PATH', '')
# Define the path to IDL's bin directory (where its DLLs are)
idl_bin_path = r'C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64'
# Prepend IDL's bin path to the current PATH
os.environ['PATH'] = idl_bin_path + os.pathsep + current_path

print(f"IDL_DIR set to: {os.environ.get('IDL_DIR')}")
print(f"ENVI_DIR set to: {os.environ.get('ENVI_DIR')}")
print(f"PATH now includes IDL bin: {idl_bin_path}") # Confirming the addition

# Point sys.path directly to the 'bridges' folder where idlpy.py is located
sys.path.append(r'C:\Program Files\NV5\ENVI61\IDL91\lib\bridges')
print(f"sys.path includes: {sys.path[-1]}")

try:
    from idlpy import *
    print("idlpy imported successfully!")

    # Example of using idlpy
    idl.print_("Hello from IDL via Python!")
    result_idl = idl.execute('print, 1 + 1', _RETURN=True) # Use _RETURN=True to capture output
    print(f"IDL calculation: {result_idl}")
    
except ImportError as e:
    print(f"Error importing idlpy: {e}")
    print("Please double-check the path and environment variables.")
except Exception as e:
    print(f"An unexpected error occurred after import: {e}")

# ... (rest of your main() function from previous code) ...

# Keep your subprocess call if you still need it for perform_OSP.pro
def main():
    # ... (your existing code for subprocess) ...

    # this is the start of the subprocess stuff
    try:
        working_directory = r"C:\Archimedes Palimsest\PalimpsestTxtContrast\envi\src"
        idl_script = os.path.join(working_directory, "perform_OSP.pro")
        idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"
        
        if not os.path.isfile(idl_script):
            print(f"Error: IDL script not found at expected path: {idl_script}")
            return
            
        idl_script_forward = idl_script.replace("\\", "/") 
        
        command = [
            idl_exe,
            "-e",
            f'.compile "{idl_script_forward}"',
            "-e",
            "perform_OSP"
        ]

        print(f"\nExecuting IDL command (output from subprocess will appear below): {' '.join(command)}")

        result = subprocess.run(command,
                                cwd=working_directory, 
                                text=True,             
                                shell=False,
                                stdout=sys.stdout,     
                                stderr=sys.stderr)     

        if result.returncode == 0:
            print("\nIDL script execution finished successfully.")
            print("success")
        else:
            print(f"\nIDL script execution failed with error code: {result.returncode}")
            print("Look for IDL error messages above.")

    except FileNotFoundError:
        print(f"Error: IDL executable not found at {idl_exe}. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
if __name__ == "__main__":
    main()