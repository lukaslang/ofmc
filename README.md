# Joint Motion Estimation and Source Identification using Convective Regularisation

This repository contains a Python implementation of the methods described in the paper [Joint Motion 
Estimation and Source Identification using Convective Regularisation with an Application 
to the Analysis of Laser Nanoablations](https://doi.org/10.1101/686261)

<img src="https://user-images.githubusercontent.com/551031/68870917-f24e5800-06fb-11ea-96f7-02a1724c1a37.png" width="40%"><img src="https://user-images.githubusercontent.com/551031/68870935-f7130c00-06fb-11ea-885d-5ea482655be8.png" width="40%">

If you use this code in your work please cite our paper:

```
L. F. Lang, N. Dutta, E. Scarpa, B. Sanson, C.-B. Schönlieb, and J. Étienne. Joint Motion 
Estimation and Source Identification using Convective Regularisation with an Application 
to the Analysis of Laser Nanoablations, bioRxiv 686261, 2019.
```

Here is the BibTeX entry:
```
@article {LanDutScaSanSchoEti19,
	author = {Lang, Lukas F. and Dutta, Nilankur and Scarpa, Elena and Sanson, B{\'e}n{\'e}dicte and Sch{\"o}nlieb, Carola-Bibiane and {\'E}tienne, Jocelyn},
	title = {Joint Motion Estimation and Source Identification using Convective Regularisation with an Application to the Analysis of Laser Nanoablations},
	elocation-id = {686261},
	year = {2019},
	doi = {10.1101/686261},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/07/02/686261},
	eprint = {https://www.biorxiv.org/content/early/2019/07/02/686261.full.pdf},
	journal = {bioRxiv}
}
```

This code is accompanied by a [dataset](http://doi.org/10.5281/zenodo.3257654):
```
L. F. Lang, N. Dutta, E. Scarpa, B. Sanson, C.-B. Schönlieb, and J. Étienne. (2019). Microscopy 
image sequences and annotated kymographs of laser ablation experiments in Drosophila 
embryos [Data set]. Zenodo.
```

## Dependencies

This software was written for and tested with:
- MacOS Mojave (version 10.14.6)
- Anaconda (conda version 4.7.12)
- Python (version 3.6.7)

The following libraries are required for parts of this code:

- FEniCS (version 2018.1.0)
- scipy
- matplotlib
- pillow
- read-roi
- imageio

## Installation

Download and install [Anaconda](https://anaconda.org/).

There are two ways to create the conda environment with the correct library versions:

a) Use the provided [`environment.yml`](environment.yml) file:

```bash
conda env create -f environment.yml 
```

b) Manually create the environment:

```bash
conda create -n fenicsproject -c conda-forge python=3.6 fenics=2018.1.0 scipy matplotlib pillow read-roi imageio 
```

Then, activate the environment:

```bash
conda activate fenicsproject 
```

In order to run/edit the scripts using an IDE, install e.g. Spyder:

```bash
conda install -c conda-forge spyder 
```

Alternatively, you can use e.g. PyCharm and create a run environment by selecting anaconda3/envs/fenicsproject environment.

## Usage

To run the test cases, execute

```bash
python -m unittest discover 
```

We have added scripts that generate the figures in the paper.

1. Download the microscopy data (ZIP file) from [https://doi.org/10.5281/zenodo.3257654](https://doi.org/10.5281/zenodo.3257654)

2. Uncompress the data, and place it some directory.

3. Set the path to this directory in the script [`datapath.py`](datapath.py).

4. Run the scripts (e.g. [`paper_figures_01.py`](paper_figures_01.py)) to re-create the figures in the paper.

In order to generate the results from the evaluation you must run [`pipeline_eval.py`](pipeline_eval.py) first (this may take a few hours).

## License & Disclaimer

This code is released under GNU GPL version 3. 
For the full license statement see the file [`LICENSE`](LICENSE).

Moreover, the package includes a third-party library:

Name: tifffile.py  
Author: Christoph Gohlke  
URL: [http://www.lfd.uci.edu/~gohlke/](http://www.lfd.uci.edu/~gohlke/) 

See [`ofmc/external/tifffile.py`](ofmc/external/tifffile.py) for its license.

## Contact

[Lukas F. Lang](https://lukaslang.github.io) and [Carola-Bibiane Schönlieb](http://www.damtp.cam.ac.uk/user/cbs31)  
Department of Applied Mathematics and Theoretical Physics  
University of Cambridge  
Wilberforce Road, Cambridge CB3 0WA, United Kingdom

[Nilankur Dutta](https://www-liphy.ujf-grenoble.fr/infos_pratiques/fiches_identites/danr.html) and [Jocelyn Étienne](https://www-liphy.ujf-grenoble.fr/pagesperso/etienne/)  
Laboratoire Interdisciplinaire de Physique  
Université Grenoble Alpes  
F-38000 Grenoble, France

[Bénédicte Sanson](https://www.pdn.cam.ac.uk/directory/benedicte-sanson) and Elena Scarpa  
Department of Physiology, Development and Neuroscience  
University of Cambridge  
Downing Site, Cambridge CB2 3DY, United Kingdom

## Acknowledgements

LFL and CBS acknowledge support from the Leverhulme Trust project "Breaking the non-convexity barrier", the EPSRC grant EP/M00483X/1, the EPSRC Centre Nr. EP/N014588/1, the RISE projects ChiPS and NoMADS, the Cantab Capital Institute for the Mathematics of Information, and the Alan Turing Institute.

ND and JE were supported by ANR-11-LABX-0030 "Tec21", by a CNRS Momentum grant, and by IRS "AnisoTiss" of Idex Univ. Grenoble Alpes.
ND and JE are members of GDR 3570 MecaBio and GDR 3070 CellTiss of CNRS.
Some of the computations were performed using the Cactus cluster of the CIMENT infrastructure, supported by the Rhone-Alpes region (GRANT CPER07_13 CIRA) and the authors thank Philippe Beys, who manages the cluster.

Overall laboratory work was supported by Wellcome Trust Investigator Awards to BS (099234/Z/12/Z and 207553/Z/17/Z).
ES was also supported by a University of Cambridge Herchel Smith Fund Postdoctoral Fellowship.

The authors also wish to thank Pierre Recho for fruitful discussions and the re-use of his numerical simulation code.
