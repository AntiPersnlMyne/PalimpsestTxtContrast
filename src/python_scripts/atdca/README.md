# ATDCA: Automatic Target Detection and Classification Algorithm

---

## Overview

**ATDCA** (Automatic Target Detection and Classification Algorithm) is an unsupervised algorithm designed to detect and classify targets in multispectral or hyperspectral imagery. It requires no prior knowledge of the scene (i.e. *requires no user input*). 

This makes ATDCA ideal for:
- Detecting unusual materials
- Classifying objects in scenes with unknown content

The algorithm is based on **Orthogonal Subspace Projection (OSP)** algorithm and consists of three core stages:

1. **BGP** – Band Generation Process
2. **TGP** – Target Generation Process  
3. **TCP** – Target Classification Process

ATDCA works efficiently on large image datasets using memory-safe, block-wise processing powered by `rasterio` and `NumPy`.


## How to use the ATDCA Pipeline 

- Place **single-band TIFF files** in `data/input/`, one per spectral band.
  - Example: `band370nm.tif`, `band400nm.tif`, ..., `band970nm.tif`

To run the pipeline either:
1) Run the pipeline directly from the terminal
- `python src/python_scripts/atdca_pipeline.py`
2) Import the pipeline into your pre-existing Python environment
- ```Python
    import atdca
    atdca.ATDCA(...)```

