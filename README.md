# GOSP
An implementation of the Generalized Orthoginal Subspace Projection (GOSP) target detection method. A students' contribution to the MISHA project. The MISHA (Multispectral Imaging System for Historical Artifacts) project is a system created by College of Imaging Science at RIT and the RIT Museum Studies Program.



## Setup
(1) Clone the repository to your local machine using a terminal:

git clone https://github.com/AntiPersnlMyne/PalimpsestTxtContrast


(2) Navigate to the project directory:

cd PalimpsestTxtContrast


(3) Run the setup script:

(Linux) \
`bash setup.sh`

(Windows) \
`setup.bat`

(4) [optional] Move `data/`, `gosp/` and `main.py` to `(gospvenv)` if it exists


This script will:
- Install the required dependencies (libraries) from the `requirements.txt` file
- Compile and build project files 
- Optionally creates virtual environment - keeps dependencies isolated
- Delete the setup.bat/.sh and requirements.txt file
--------------------------------------------------------------------------------



## Getting started
Just import the gosp function into your project. An example can be seen in provided `main.py`:
```python
from gosp import gosp

gosp(
    # Required parameters
    input_dir="data/input"
    output_dir="data/output"
    # Optional parameters
    ...
)
```



## Dependencies
- Python version 3.15+
- [optional][future] CUDA 13.0 compatible GPU

This project includes the following libraries: 
1. Numpy (data arrays)
2. Cython (C extensions)
3. Setuptools (Build/Compiler)
4. Numba (JIT compiler)
5. rasterio (raster I/O)
6. GDAL (virtual raster title)
7. tqdm (terminal progress bar)

The startup script AUTOMATICALLY downloads these libraries, ready to use, no user input required. venv ('gospenv' during setup.sh/.bat) recommended for package isolation.



## Structure
The current Python script is setup to accept TIFF (`.tif`) files. `gosp` function accepts all rasterio compatible types.

Place all raw images into the input directory:

`data/input`

Targets from gosp are written as multiband TIFF: 

`<OUTDIR>targets_classified/targets_classified.tif`

Source (.py, .c) and build (.so/.pyd) files are in gosp directory: 

`gosp/build`
`gosp/src`

The execution file is `main.py`



## Questions
Complaints may fall on deaf ears. Questions may fall on ignorant ears.

For questions concerning legality or contact from within CIS, please contact Gian-Mateo (AntiPersnlMyne) at: 
`mt9485@rit.edu` (until 2026)
`mtifone2022@disroot.org`



### Additional Personel

Douglas Tavolette, BS   | RIT software engineering student, who devoted his free time to developing code for this project


Roger Easton Jr., Ph. D | My adviser who gave me freedom to explore the world of R&D, and project sponsor.


David Messinger, Ph. D  | Providing project direction.


Juliee Decker, Ph.D     | Graciously teaching and included myself into all things MISHA.

