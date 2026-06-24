# **Fast-growing intracellular _Mycobacterium tuberculosis_ populations evade antibiotic treatment**

***macrohet*** is a code repository designed to investigate ***Macro***phage ***het***erogeneity. It accompanies the aforementioned manuscript exploring single-cell heterogeneity in _Mtb_-infected macrophages using time-lapse microscopy, tracking, and single-cell growth rate analysis.

Interactive figures and plots for this project can be explored via GitHub Pages:
[nthndy.github.io/macrohet](https://nthndy.github.io/macrohet)

<p align="left">
  <img src="https://github.com/nthndy/macrohet/raw/main/docs/images/landing_img.png" alt="macrohet image" width="800">
</p>
Image description: A pseudocoloured timelapse image of Mtb, projected along the time axis visualise spatiotemporal evolution.

## Contents

- `notebooks/`: Reproducible analysis notebooks for data loading, segmentation, tracking, and quantification
- `macrohet/`: Python module with core analysis functions
- `data/`: Subset of image data with associated segmentation and tracks
- `models/`: Bespoke segmentation model and _btrack_ tracking parameters
- `docs/`: HTML manuscript and supporting content (hosted via GitHub Pages)
- `environment.yml`: Conda environment specification
- `.pre-commit-config.yaml`: Code formatting and linting hooks

---

## Installation and Reproducibility

The following instructions are configured for an Ubuntu workstation. 

Clone the repository:

```bash
git clone https://github.com/nthndy/macrohet.git
cd macrohet
```

Create, install and activate the environment:

```bash
mamba env create -f environment.yml
mamba activate macrohet
```

Parts of the image tiling and stitching pipeline were adapted from [Volker Hilsenstein’s DaskFusion project](https://github.com/VolkerH/DaskFusion), used under the MIT License.
Details of the hardware and software used to generate the analyses in this repository are provided in [reproducibility.md](reproducibility.md).

---

## Example data

To let the pipeline run end to end without the full dataset, the repository ships a
small example acquisition under `data/`:

- `data/untiled_images/` — a down-scaled 3×3-tile Opera Phenix field of view (well
  r03c05), comprising the individual acquisition fields and the `Index.idx.xml`
  metadata. This is the only example image input tracked in the repository.

Everything downstream is derived from this and is generated locally rather than
committed, to keep the repository lean:

1. `notebooks/tile_image.ipynb` parses the Harmony metadata with
   `dataio.read_harmony_metadata` and stitches the fields into a contiguous mosaic
   with `tile.compile_mosaic`.
2. The save step writes the stitched mosaic to `data/example_data.zarr` (OME-Zarr,
   recommended) or `data/example_data.ome.tiff` (OME-TIFF).

These outputs are git-ignored; generate them by running `tile_image.ipynb` once after
cloning. The full processed single-cell dataset and the raw imaging data are held
externally, see the Data Availability statement for the repository DOI.

---

## Contact

For questions or access to underlying data/code, please contact:

**Nathan J. Day** <br>
_Host–Pathogen Interactions in Tuberculosis Laboratory_ <br>
The Francis Crick Institute <br>
nthndy@gmail.com <br>
[@nthndy.bsky.social](https://bsky.app/profile/nthndy.bsky.social) <br>
[github.com/nthndy](https://github.com/nthndy) <br>
