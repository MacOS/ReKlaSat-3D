# Semantic Segmentation of Point Clouds Derived from Tri-Stereo Pleiades Satellite Imagery
We study the utility of point clouds derived from tri-stereo satellite imagery for semantic1segmentation for Convolutional Neural Networks (CNNs). In particular, we examine if the geometric information, additional to color, has an influence on the segmentation performance for segmenting clutter, roads, buildings, trees, and vehicles.

# Requirements
- Ubuntu 14.04 or higher
- Python 3.6 or higher
- CUDA 10.0 or higher
- pytorch 1.2 or higher

# Installation
We recommend that you use anaconda to separate the environment.


The following command creates the conda environment ```py3-mink``` and installs the necessary python dependencies.
```sh
conda env create -f py3-mink.yml
```
To install the Minkowski Engine in the created environment run
```sh
conda activate py3-mink
sh install_minkovski_engine.sh
```

# Authors by Institution

## Vienna University of Economics and Business
[Assoc. Prof. Dr. Ronald Hochreiter (Projekt Manager)](https://scholar.google.at/citations?hl=de&user=NdGSq4EAAAAJ)

BSc. (WU) Andrea Siposova

BSc. (WU) Niklas Schmidinger

[BSc. (WU) Stefan Bachhofner](https://scholar.google.at/citations?hl=de&user=-WZ0YuUAAAAJ)

## Vienna University of Technology
[Univ.Prof. Dipl.-Ing. Dr.techn. Pfeifer Norbert](https://scholar.google.at/citations?user=-HuwYEMAAAAJ&hl=en)

[MSc. Ana-Maria Loghin](https://scholar.google.at/citations?hl=en&user=E_HkvF8AAAAJ&view_op=list_works)

Dipl.-Ing. Dr.techn. Johannes Otepka-Schremmer

## Siemens AG Austria
[Dr. Michael Hornacek](https://scholar.google.at/citations?user=llItOJ8AAAAJ&hl=en)

[Dr. Olaf Kähler](http://www.robots.ox.ac.uk/~olaf/)

## Vermessung Schmid ZT GmbH
[Mag. Nikolaus Schiller](https://at.linkedin.com/in/nikolaus-schiller-37921418)


# Projekt Partners
[Vienna University of Economics and Business, Research Institute for Computational Methods. (Projet Coordinator)](https://www.wu.ac.at/en/firm)

[Vienna University of Technology, Department of Geodesy and Geoinformation.](https://www.geo.tuwien.ac.at/)

[Siemens AG Österreich, Corporate Technology.](https://new.siemens.com/at/de.html)

[Vermessung Schmid ZT GmbH.](http://www.geoserve.co.at/)

[Federal Ministry of Defence, Austria.](http://www.bundesheer.at/english/index.shtml)

# Funding
This research was funded by the Austrian Research Promotion Agency (FFG) project [“3D Reconstruction and Classification from Very High Resolution Satellite Imagery (ReKlaSat 3D)” (grant agreement No. 859792)](https://projekte.ffg.at/projekt/1847316).
