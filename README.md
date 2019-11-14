# Joint Motion Estimation and Source Identification using Convective Regularisation
==================================

This repository contains a Python implementation of the methods described in the paper [Joint Motion 
Estimation and Source Identification using Convective Regularisation with an Application 
to the Analysis of Laser Nanoablations](https://doi.org/10.1101/686261)

## Cite
----

If you use this software in your work please cite our paper in
resulting publications:

L. F. Lang, N. Dutta, E. Scarpa, B. Sanson, C.-B. Schönlieb, and J. Étienne. Joint Motion 
Estimation and Source Identification using Convective Regularisation with an Application 
to the Analysis of Laser Nanoablations, bioRxiv 686261, 2019.

URL: https://doi.org/10.1101/686261

BibTeX:

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

## Dependencies
--------

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

Installation instructions:

1. Download and install Anaconda from https://anaconda.org/

There are two ways to create the conda environment using the correct library versions:

a) Use the provided environment file (environment.yml):

```bash
conda env create -f environment.yml
```

b) Manually create the environment:

```bash
conda create -n fenicsproject -c conda-forge python=3.6 fenics=2018.1.0 scipy matplotlib pillow read-roi imageio
```

2. Activate the environment:

```bash
conda activate fenicsproject
```

3. In order to run/edit the scripts in an IDE install e.g. Spyder:

```bash
conda install -c conda-forge spyder
```

Alternatively, you can use e.g. PyCharm and create a run environment by selecting anaconda3/envs/fenicsproject environment.

## Usage

To run the test cases execute

```bash
>> conda activate fenicsproject
>> python -m unittest discover
```

We have added scripts that generate the figures in the paper. First, download 
the microscopy data (ZIP file) from:

https://doi.org/10.5281/zenodo.3257654

Uncompress the data, and place it some directory. Second, set the path to this 
directory in the script "datapath.py".

Run the scripts, e.g. [`paper_figures_01.py`](paper_figures_01.py), to re-create the figures in the paper.

In order to generate the results from the evaluation you must run [`pipeline_eval.py`](pipeline_eval.py) first.

## License & Disclaimer
--------------------

Copyright 2019 Lukas Lang.

This file is part of OFMC. OFMC is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

OFMC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
OFMC. If not, see <http://www.gnu.org/licenses/>.

For the full license statement see the file LICENSE.

Moreover, the package includes third-party libraries:

Name: tifffile.py
Author: Christoph Gohlke
URL: http://www.lfd.uci.edu/~gohlke/

See ofmc/external/tifffile.py for its license.

## Contact
-------

Lukas F. Lang (ll542@cam.ac.uk)
Carola-Bibiane Schönlieb (cbs31@cam.ac.uk)
Department of Applied Mathematics and Theoretical Physics
University of Cambridge
Wilberforce Road, Cambridge CB3 0WA, United Kingdom

Nilankur Dutta (nilankur.dutta@univ-grenoble-alpes.fr)
Jocelyn Étienne (jocelyn.etienne@univ-grenoble-alpes.fr)
Laboratoire Interdisciplinaire de Physique
Université Grenoble Alpes
F-38000 Grenoble, France

Bénédicte Sanson (bs251@cam.ac.uk)
Elena Scarpa (es697@cam.ac.uk)
Department of Physiology, Development and Neuroscience
University of Cambridge
Downing Site, Cambridge CB2 3DY, United Kingdom

## Acknowledgements

LFL and CBS acknowledge support from the Leverhulme Trust project "Breaking the non-convexity barrier", the EPSRC grant EP/M00483X/1, the EPSRC Centre Nr.\ EP/N014588/1, the RISE projects ChiPS and NoMADS, the Cantab Capital Institute for the Mathematics of Information, and the Alan Turing Institute.
ND and JE were supported by ANR-11-LABX-0030 "Tec21", by a CNRS Momentum grant, and by IRS "AnisoTiss" of Idex Univ. Grenoble Alpes. 
ND and JE are members of GDR 3570 MecaBio and GDR 3070 CellTiss of CNRS.
Some of the computations were performed using the Cactus cluster of the CIMENT infrastructure, supported by the Rhone-Alpes region (GRANT CPER07_13 CIRA) and the authors thank Philippe Beys, who manages the cluster.
Overall laboratory work was supported by Wellcome Trust Investigator Awards to BS (099234/Z/12/Z and 207553/Z/17/Z).
ES was also supported by a University of Cambridge Herchel Smith Fund Postdoctoral Fellowship.
The authors also wish to thank Pierre Recho for fruitful discussions and the re-use of his numerical simulation code.
