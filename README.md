# Palimpsest Text Contrast
An implementation of the Orthoginal Subspace Projection (OSP) target detection method on multi-spectral, historical document imaging. A student contribution to the MISHA project. 


## Setup
(1) Clone the repository to your local machine using a terminal:

git clone https://github.com/AntiPersnlMyne/PalimpsestTxtContrast


(2) Navigate to the project directory:

cd ~/PATH TO PROJECT/PalimpsestTxtContrast


(3) Run the setup script:

(Linux) \
`bash setup.sh`

(Windows) \
`setup.bat`

(4) Move data to be processed into `data/input`


This script will ---------------------------------------------------------------
- Install the required dependencies (libraries) from the `requirements.txt` file
- Create the necessary directory structure
- Optionally creates virtual environment - keeps dependencies isolated
- Delete the `setup.bat/.sh` file
--------------------------------------------------------------------------------



## Dependencies
This project uses the ENVI (Environment for Visualizing Imagery) software by NV5 Geospatial Software. This project is compatable with the now latest version of ENVI - 6.1 with IDL 9.1. Compatible Python versions are > 3.10.x and < 3.12.x. 

The software can be made available through a CIS (Chester F. Carlson College of Imaging Science) license. The MISHA (Multispectral Imaging System for Historical Artifacts) project is a system created by CIS and the RIT Museum Studies Program.

ENVI allows you to extract meaningful information from hyperspectral and multispectral (as well as MANY others) types of imagery to make better decisions. 

In this instance, ENVI is used to process historical manuscripts to enhance details (text contrast), allowing non-imaging personel to make better decisions about the document's contents.

This project includes the following libraries: 
1. Numpy (data arrays)
2. OpenCV (computer vision)
3. SPy (spectral data processing)
4. scikit-image (image processing)
5. SciPy (scientific computing)
6. Matplotlib (plotting)
7. Pytesseract (OCR)
8. Pillow/PIL (OCR dependancy)

The startup script AUTOMATICALLY downloads these libraries, ready to use, no user input required.
~ It is recommended to setup in a virtual environment for package isolation



## Usage
The current Python script is setup to accept TIFF (`.tif`, `.tiff`) raw image files, as well as ENVI image cubes (`.hdr`, `.dat`). 

Place all PRE PROCESSED into the following directory: 
`data/input`

Images that have been POST PROCESSED are stored in the following directory: 
`data/output`

Utility Python and IDL files are stored in their respective directories: 
`src/Python_script`
`src/IDL_script`

!Important! The execution file and main logic is "main.py"
[~/PalimpsestTxtContrast] `python main.py`


## Questions (or complaints)
Complaints may fall on deaf ears. Questions may fall on ignorant ears.

For questions concerning legality or contact from within CIS, please contact Gian-Mateo (AntiPersnlMyne) at: 
`mt9485@rit.edu`

Additional Personel:
Douglas Tavolette(dft8437@rit.edu) - RIT software engineering student, who devoted his free time to developing code for this project
Roger Easton Jr.(rlepci@rit.edu)   - My excellent adviser and project sponsor, MISHA personell
David Messinger(dwmpci@rit.edu)    - Another adviser and mentor, MISHA personell

