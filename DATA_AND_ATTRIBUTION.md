# Data and attribution

## Ownership of this project

The study design, analysis code, documentation and scientific interpretation in
this repository are the original work of **Paola Aguirre**, who is the author,
maintainer and copyright holder of the original code. The licence for the
original code is given in `LICENSE`.

The items below are third-party resources. Their authors and licences are
listed so that their terms are respected; listing them does not make them part
of this project's authorship.

## Dataset

**Sleep-EDF Database Expanded, version 1.0.0** (Sleep-Cassette subset), hosted
by PhysioNet.

- DOI: <https://doi.org/10.13026/C2X676>
- Page: <https://physionet.org/content/sleep-edfx/1.0.0/>
- Dataset author listed by PhysioNet: Bob Kemp
- Licence: **Open Data Commons Attribution License v1.0 (ODC-By 1.0)**,
  <https://opendatacommons.org/licenses/by/1-0/>

**No data from this dataset are redistributed in this repository.** No EDF
recordings, hypnograms, archives or epoch-level features are included. Users
must obtain the data from PhysioNet and comply with its licence.

The published files in `results/` are aggregate summaries and figures derived
from the dataset. They contain public Sleep-EDF record identifiers (for example
`SC4072E`) and two-digit subject numbers as used in the dataset's own file
names; they contain no other subject information.

### Required citations for the dataset

Please cite, as requested by PhysioNet:

1. Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a
   sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
   EEG. *IEEE Transactions on Biomedical Engineering.* 2000;47(9):1185–1194.
   <https://ieeexplore.ieee.org/document/867928/>
2. The dataset itself: Kemp B. Sleep-EDF Database Expanded (version 1.0.0).
   PhysioNet. <https://doi.org/10.13026/C2X676>
3. The PhysioNet platform: Pollard T, Moody BE, Lehman L, Gow B, Fernandes C,
   Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for
   biomedical research. *Nature Health.* 2026;1(8):792–795.
   <https://doi.org/10.1038/s44360-026-00096-z>

## Software dependencies

The code depends on, but does not include, the following open-source packages.
Each is used under its own licence.

| Package | Version used | Licence |
|---|---|---|
| Python | 3.13.12 | PSF License |
| MNE-Python | 1.13.2 | BSD-3-Clause |
| NumPy | 2.4.4 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| pandas | 3.0.2 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| matplotlib | 3.10.8 | Matplotlib License (PSF-based) |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| pytest (tests only) | 9.2.0 | MIT |

If you use the analysis, please also cite MNE-Python (Gramfort et al., 2013,
*Frontiers in Neuroscience* 7:267) and scikit-learn (Pedregosa et al., 2011,
*JMLR* 12:2825–2830).

## Third-party code

No third-party source code is copied into this repository. The scripts call
the public APIs of the packages listed above.
