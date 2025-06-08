# Palimpsest Text Contrast
An [Unofficial] algorithm for the enhancing the hidden text from historical documents; in particular, the Archimedes Palimpsest. Independent contribution to the MISHA project.


## Setup
(1) Clone the repository to your local machine using a terminal:

git clone https://github.com/AntiPersnlMyne/PalimpsestTxtContrast

(2) Navigate to the project directory:

cd ~/PATH TO PROJECT/PalimpsestTxtContrast

(3) Run the setup script:

(Linux)
`bash setup.sh`

(Windows)
`setup.bat`


This script will
- Create a Python virtual environment named "envi"
- Activate the virtual environment
- Install the required dependencies from the `requirements.txt` file
- Create the necessary directory structure
- Move the `ENVI_script.py` file to the `envi/src` directory
- Delete the `setup.bat/.sh` file

4. Once the setup is complete, you can start using the project.


## Dependencies

This project uses the ENVI (Environment for Visualizing Imagery) software by NV5 Geospatial Software. This project is compatable with the now latest version of ENVI - 6.1 with IDL 9.1

The software can be made available through a CIS (Chester F. Carlson College of Imaging Science) license. The MISHA (Multispectral Imaging System for Historical Artifacts) project is a system created by CIS and the RIT Museum Studies Program.

ENVI allows you to extract meaningful information from hyperspectral and multispectral (as well as MANY others) types of imagery to make better decisions. 

In this instance, ENVI is used to process historical manuscripts to enhance details (text contrast), allowing non-imaging personel to make better decisions about the document's contents.

This project includes the following libraries: 
1. ENVI Py Engine (envipyengine)
2. OpenCV (Open Computer Vision)

The startup script AUTOMATICALLY downloads these libraries into the virtual environment, ready to use, no user input required.

More information on these libraries can be found at the following:
(envipyengine) https://envi-py-engine.readthedocs.io/en/latest/
(OpenCV) https://opencv.org/about/



## Usage
The current Python script is setup to accept TIFF (.tif or .tiff) raw image files. This is due to the Palimpsest data being multispectral, and stored in the .tif file format.

Place all PRE PROCESSED into the following directory:

`envi/data/input`

Images that have been POST PROCESSED are stored in the following directory:

`envi/data/input`

All processing logic is stored in the source (src) folder:

`envi/src`


## Questions (or complaints)
With all due respect, I didn't get paid to do this, so complains MAY fall on deaf ears. Questions may fall on ignorant ears.

For questions concerning legality or contact from within CIS, please contact Gian-Mateo (AntiPersnlMyne) at:

`mt9485@rit.edu`


