<p align="center">
  <img src="https://github.com/nthndy/macrohet/raw/main/docs/images/landing_img.png" alt="macrohet image" width="600">
</p>

# macro*het*

**Fast-growing intracellular _Mycobacterium tuberculosis_ populations evade antibiotic treatment**

This repository accompanies the manuscript exploring single-cell heterogeneity in _Mtb_-infected macrophages using time-lapse microscopy, tracking and single-cell growth rate analysis.

GitHub Pages for this project (figures + interactive plots are work in progress, pending publication):
[nthndy.github.io/macrohet](https://nthndy.github.io/macrohet)

---

## Contents

- `notebooks/`: Reproducible analysis notebooks for data loading, segmentation, tracking, and quantification
- `macrohet/`: Python module with core analysis functions
- `docs/`: HTML manuscript and supporting content (hosted via GitHub Pages)
- `environment.yml`: Conda environment specification
- `.pre-commit-config.yaml`: Code formatting and linting hooks
- `README.md`: Project overview and usage instructions

---

## Installation and reproducibility

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

Details of the hardware and software used to generate the analyses in this repository are provided in [reproducibility.md](reproducibility.md).

---

## Contact

For questions or access to underlying data/code, please contact:

**Nathan Day** <br>
_Host–Pathogen Interactions in Tuberculosis Laboratory_ <br>
The Francis Crick Institute <br>
nathan.day@crick.ac.uk <br>
[@nthndy.bsky.social](https://bsky.app/profile/nthndy.bsky.social) <br>
[github.com/nthndy](https://github.com/nthndy)<br>
