import subprocess

def main():
    print("Hello, world!")
    
    #this is the start of the subprocess stuff
    try:
        subprocess.run(["javac", "makeAFile.java"], check=True)
        subprocess.run(["java", "makeAFile"], check=True)
        
        print("success")
    except subprocess.CalledProcessError as e:
        print("error: ", e)
        
    
    


if __name__ == "__main__":
    main()