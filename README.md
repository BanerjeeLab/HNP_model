# HNP Closure Model
Computational model for Hindbrain neuropore morphogenesis, as detailed in "Self-organized cell patterning via mechanical feedback in hindbrain neuropore morphogenesis".

## Authors / Contributors
- Fernanda Pérez-Verdugo (Carnegie Mellon University)
- Shiladitya Banerjee (Carnegie Mellon University)

## System Requirements
- Python 3.9.7

## QUICKSTART GUIDE
The /Simulation/ directory contains the following:

 **1. /initial_data/:** Contains all the geometrical information required to create the initial condition of the tissue. These conditions are generated as explained in the Supplementary Information (SI).

 **2. MainFile.py:** The main script to run the simulation. Outputs: 
 - Vertex and cell-level properties, as well as network topology over time, are saved in the directory /resultados/.
 - Cell IDs forming rows 1, 2, and 3 are stored in /first_row/, /second_row/, and /third_row/, respectively.
 - Creation of 4-fold vertices is saved in 4-fold.txt.
 - Resolution of 4-fold vertices is saved in T1_perpendicular.txt (T1 events) and T1_original.txt (reverse T1 events).

 **3. tissue_creation.py:** A script called by MainFile.py to generate the tissue object.

 **4. classes_and_functions.py:** Includes Vertex, Cell, and Tissue classes, and all the relevant functions for the simulation.

 **5. config_XXX.json:** An external configuration file that allows the customization of simulation parameters and enables switching between different models. It is used as input in MainFile.py. Key configurable parameters include:
- Th: Gap tension.
- v0: Cell crawling speed.
- ma: Anisotropic stress alignment rate.
- md: Anisotropic stress decay rate.
- mf: Mechanical feedback rate.
- Sigma0: Anisotropic stress amplitude.
- CrawlingLeader: Specifies the crawling model. Options:
    - "R1": Row 1 leader.
    - "R2": Row 2 leader.
- PrePatterned: Specifies whether a pre-patterned nematic field around the HNP gap is considered. Options:
"YES" or "NO". Examples:
    - config_fig2A.json: Reproduces the simulation shown in Figure 2A.
    - config_fig3B.json: Reproduces the simulation shown in Figure 3B.

 **6. snapshots.py:** This script generate visual snapshots of the tissue, showing the nematic field across the tissue as in Fig3B. Uses data from /resultados/ to create visualizations. Snapshots are saved in the /snapshots/ directory.

## Contribution guidelines
- Email: (shiladtb@andrew.cmu.edu)
  
## Who do I talk to?
- Fernanda Pérez-Verdugo (fverdugo@andrew.cmu.edu)
- Shiladitya Banerjee (shiladtb@andrew.cmu.edu)
