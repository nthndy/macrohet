# macro _het_

A repo for bringing together all of the threads of my analyses for studying the single cell heterogeneity of human macrophages infected with Mtb.

WORK IN PROGRESS

Currently mainly using it to back up various messy notebooks etc.

### This repo contains 5 main components, each explained in their own Jupyter Notebooks that depend upon a main python module called macrohet:

1. A tiling approach that takes the fragmented images and compiles them into mosaic images, either for viewing or segmenting.
2. A segmentation approach that labels each cell in the mosaic field of view, designed to utilise the latest developments in cell segmentation.
3. A tracking section that unites each single-cell segment over time in order to follow their temporal evolution.
4. A downstream analysis that utilises the aforementioned sections to understand the single-cell heterogeneity of human macrophages infected with Mtb.
5. A Napari based viewer for inspecting the tiling, segmentation, and tracking.
