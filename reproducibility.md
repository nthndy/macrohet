# Reproducibility and Computational Environment

This project was developed and executed on a high-performance Linux workstation using open-source tools and a Conda-based environment. The following hardware and software specifications are provided to support reproducibility of all analyses and results.

## Hardware

- **CPU**: Intel® Core™ i9-10980XE @ 3.00GHz
  - 18 cores / 36 threads
  - 24.8 MiB L3 cache
- **RAM**: 256 GB DDR4
- **GPU**: NVIDIA RTX A6000 (48 GB VRAM)
  - Driver version: 555.42.06
  - CUDA version: 12.5

## Operating System

- **Distribution**: Ubuntu 20.04.6 LTS (Focal Fossa)
- **Architecture**: x86_64
- **Virtualisation support**: VT-x enabled
- **Kernel address size**: 46-bit physical, 48-bit virtual

## Software Stack

| Component  | Version   |
| ---------- | --------- |
| Python     | 3.10.18   |
| Conda      | 23.3.1    |
| Mamba      | 1.4.2     |
| JupyterLab | 4.4.3     |
| NVIDIA-SMI | 555.42.06 |

## Notes

All analysis was performed using POSIX-compliant paths and assumes a Linux environment. Code may require minor path modifications if used on macOS or Windows.

An isolated conda environment is recommended for dependency resolution. See `environment.yml` for installation details.
