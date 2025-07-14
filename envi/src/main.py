import subprocess
import os

def main():
    print("Hello, world!")
    
    """THIS WILL NEED TO BE CHANGED TO BE CORRECT FOR YOUR INDIVIDUAL PATH
       make it so the path ends at "src" for this repository"""
    working_directory = r"C:\Archimedes Palimsest\PalimpsestTxtContrast\envi\src"
    
    #This is debugging to make sure that you did correctly change your working directory to something that exists
    if not os.path.isdir(working_directory):
        print("Invalid working directory:", working_directory)
    else:
        print("Working directory exists")
    #this is the start of the subprocess stuff
    try:
        """CHANGE OSP.pro TO YOUR FILE'S NAME"""
        idl_script = os.path.join(working_directory, "OSP.pro")
        """THIS MIGHT ALSO NEED TO BE CHANGED"""
        idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"
        
        command = f'"{idl_exe}" -e ".run \'{idl_script}\' ; open_file"'

        result = subprocess.run(command, cwd=working_directory, shell=True, capture_output=True, text=True)
        # subprocess.run[]
        # subprocess.run(["javac", "makeAFile.java"], check=True)
        # subprocess.run(["java", "makeAFile"], check=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

        print("success")
    except subprocess.CalledProcessError as e:
        print("error: ", e)
        
    
    


if __name__ == "__main__":
    main()