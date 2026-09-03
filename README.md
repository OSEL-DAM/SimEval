# SimEval Toolbox


## Introduction


The SimEval Toolbox is a repository of Python code and interactive Jupyter notebooks for performing tasks 
related to **credibility assessment of computational modeling and simulation**. 
This initial release (v1.0) is focused on **verification**. Verification is defined as “the process of determining that a computational model accurately represents the underlying mathematical model and its solution from the perspective of the intended uses of modeling and simulation”. It involves two activities: code verification, which assesses whether numerical algorithms are implemented correctly, and calculation verification, which estimates numerical error in specific simulations.  

The toolbox contains the following components:
  1. Code verification of traditional PDE solvers
  2. Calculation verification of traditional PDE solvers
  3. Code verification of physics-informed neural networks (PINNs)


## Component 1: Code verification of traditional PDE solvers

This is an executable Jupyter notebook and supporting Python code demonstrating an end-to-end example of code verification. It provides the following:
 * Introduction to the method of manufactured solutions (MMS)
 * List of test problems with exact solutions available in the literature, across modeling fields
 * Discussion of practical considerations including error norm choice
 * End-to-end MMS example
 * Reusable code (in notebook) 
 
## Component 2: Calculation verification of traditional PDE solvers

This is a Python module named `calcverif` for computing calculation verification metrics, with an accompanying Jupyter notebook. 
  * Module typical inputs: results from a mesh discretization study, i.e., mesh sizes, quantity of interest values, theoretical order of convergence (optional)
  * Module outputs: observed order of convergence, Richardson extrapolated value, grid convergence index (GCI), plots.
   
The accompanying Jupyter notebook demonstrates how the module is used.
  
## Component 3: Code verification of PINNs

This is a Jupyter notebook and supporting Python code demonstrating a PINN-specific code verification workflow. It includes the following:
 * End-to-end PINN MMS-based code verification example
 * Executable, flexible, PINN solvers (1D elliptic equation and 2D monodomain equation solvers) for experimentation
 * PINN numerical convergence analysis results examining the effects of network training choices and stochastic variability
 * PINN-specific code verification workflow based on the convergence analysis results
 * Reusable code (in notebook and `src` folder)  


## Repository Structure
  * `notebooks`: Interactive Jupyter notebooks
  * `html_notebooks`: Pre-rendered HTML versions of the notebooks 
  * `src`: source code, including calculation verification metrics module, PINN solvers, and other functionality used by the notebooks 
  * `scripts`: code for generating results and figures in the supporting publication
  * `test`: Unit tests for `src` functions   
  * `saved_results`: pre-computed PINN convergence results 


## Using the Toolbox without Running Code

The recommended starting point for each SimEval component is its accompanying Jupyter notebook. 

Users who want to explore the toolbox without installing Python, Jupyter, or any Python dependencies can view the notebooks directly on GitHub, which renders the notebook text, code, outputs, and figures in the browser:

- Component 1 - Traditional solver code verification:
  * See [PdeCodeVerification.ipynb](https://github.com/OSEL-DAM/SimEval/blob/main/notebooks/PdeCodeVerification.ipynb) which provides the end-to-end MMS example
- Component 2 – Traditional solver calculation verification
  * See [PdeCalculationVerification.ipynb](https://github.com/OSEL-DAM/SimEval/blob/main/notebooks/PdeCalculationVerification.ipynb) which provides comprehensive examples demonstrating use of the `calcverif` functionality.
- Component 3 – PINN code verification
  * See [PINN_Verification.ipynb](https://github.com/OSEL-DAM/SimEval/blob/main/notebooks/PINN_Verification.ipynb) which describes the PINN solvers, presents the convergence studies and results, and proposes and tests the PINN-specific verification strategy.

For users who have downloaded the toolbox and want to view the notebooks locally without installing Python, Jupyter, or any Python dependencies, pre-rendered HTML versions are provided in the `html_notebooks` folder.

## Installation

Clone the toolbox repository
```
git clone https://github.com/OSEL-DAM/simeval.git
```

There are two installation options depending on whether you want to run the PINN code in Component 3.

### Option 1: Standard libraries only (no PINN dependencies)

This is recommended if you do **not** want to run the PINN interactive notebook. It is sufficient for the other two notebooks and supporting Python code, i.e., components 1 and 2.  

The following standard libraries are required
```
numpy
matplotlib
pandas
scipy
``` 
Development and testing were performed with versions:
 * python: 3.11.4
 * numpy: 1.26.4
 * matplotlib: 3.10.0
 * pandas: 2.2.3
 * scipy: 1.15.2


### Option 2: With PINN dependencies

The PINN solver uses `DeepXDE` (https://deepxde.readthedocs.io). `DeepXDE` supports various backends (e.g., `tensorflow`, `pytorch`) but the notebook assumes a `tensorflow` backend. After installing the Option 1 libraries, install:
```
pip install tensorflow
pip install deepxde
```
Development and testing were performed with versions:
 * tensorflow: 2.18.0
 * deepxde: 1.14.0

A virtual environment is recommended.

### Anaconda minimal installation instructions

If you use Anaconda, the following instructions can be used to install all dependencies, within a conda virtual environment. It also installs Jupyter within the environment, which is used to open and run the notebooks.
```
conda create -n YOUR_ENVIRONMENT_NAME python=3.11.4
conda activate YOUR_ENVIRONMENT_NAME

conda install numpy=1.26.4
conda install pandas=2.2.3
conda install scipy=1.15.2
conda install matplotlib=3.10.0  

pip install tensorflow==2.18.0  # only if want to run PINN code/notebook
pip install deepxde==1.14.0     # only if want to run PINN code/notebook

conda install Jupyter 

# to open interactive notebooks
jupyter notebook 
```

## Running tests

The tests in the `test` folder can be run to confirm the toolbox has been successfully installed.
```
cd test

python test_fe_solver_for_mms.py
python test_calcverif.py

# only run the below if Option 2 (PINN dependencies) was chosen above
python test_PINN_1d_elliptic.py
python test_PINN_monodomain.py
python test_PINN_verify_1d_elliptic.py
```

## Using the Toolbox (Running Code Version)

(See above for instructions on using the toolbox without running code).

The recommended starting point for each SimEval component is its accompanying Jupyter notebook. 

After installation, launch Jupyter from the SimEval repository:
```bash
jupyter notebook
```
Open the notebook corresponding to the SimEval component you want to use:
- Component 1 – Traditional solver code verification
  - `notebooks/PdeCodeVerification.ipynb` provides the end-to-end MMS code verification example.
- Component 2 – Traditional solver calculation verification
  - `notebooks/PdeCalculationVerification.ipynb` provides comprehensive examples demonstrating use of the `calcverif` functionality.
- Component 3 – PINN code verification
  - `notebooks/PINN_Verification.ipynb` describes the PINN solvers, presents the convergence studies and results, and proposes and tests the PINN-specific code verification strategy.

Each notebook contains an introduction to the component, executable code, and detailed instructions for using that corresponding component.



## Authors
**Developers**: Pras Pathmanathan, Kenny Aycock, Brent Craven

**Contact:** pras.pathmanathan@fda.hhs.gov


## Citation

If you use this toolbox in your research, please cite the accompanying paper: 
 * P. Pathmanathan, K. Aycock, B. Craven, "SimEval: a toolbox of software and interactive notebooks for credibility assessment of medical device modeling and simulation", under submission.
