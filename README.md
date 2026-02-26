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

Create and activate the environment:

```bash
mamba env create -f environment.yml
conda activate macrohet
```

Install the macrohet repository in that environment.

```bash
pip install -e .
```

Parts of the image tiling and stitching pipeline were adapted from [Volker Hilsenstein’s DaskFusion project](https://github.com/VolkerH/DaskFusion), used under the MIT License.
Details of the hardware and software used to generate the analyses in this repository are provided in [reproducibility.md](reproducibility.md).

---

## Contact

For questions or access to underlying data/code, please contact:

**Nathan J. Day** <br>
_Host–Pathogen Interactions in Tuberculosis Laboratory_ <br>
The Francis Crick Institute <br>
nathan.day@crick.ac.uk <br>
[@nthndy.bsky.social](https://bsky.app/profile/nthndy.bsky.social) <br>
[github.com/nthndy](https://github.com/nthndy) <br>
