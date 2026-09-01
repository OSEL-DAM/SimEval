# SimEval Toolbox


## Introduction


The SimEval Toolbox is a repository of Python code and interactive Jupyter notebooks for performing tasks 
related to **credibility assessment of computational modeling and simulation**. 
This initial release (v1.0) is focused on **code and calculation verification**. 
Code and calculation verification of both traditional partial differential equation (PDE)-based solvers, such as the finite element method, and of newer methods such as physics-informed neural networks (PINNs), are covered in this release. 

The toolbox contains the following components:
  1. Code verification of traditional PDE solvers
  2. Calculation verification of traditional PDE solvers
  3. PINN code and calculation verification


### Component 1: Code verification of traditional PDE solvers

This is a notebook and supporting Python code demonstrating an end-to-end example of code verification. It provides the following:
 * Introduction to the method of manufactured solutions
 * List of test problems with exact solution available in the literature, across modeling fields
 * Discussion of practical considerations including error norm choice
 * End-to-end MMS example
 * Reusable code (in notebook) 
 
### Component 3: Calculation verification of traditional PDE solvers

This is a Python module named `calcverif` for computing calculation verification metrics, with an accompanying notebook. 
  * Module typical inputs: results from a mesh discretization study, i.e., mesh sizes, quantity of interest values, theoretical order of convergence (optional)
  * Module outputs: observed order of convergence, Richardson extrapolated value, grid convergence index (GCI), plots.
   
The accompanying Jupyter notebook demonstrates how the module is used.
  
### Component 3: Code verification of traditional PDE solvers

This is a notebook and supporting Python code demonstrating the process and challenges of performing code and calculation verification for PINN solvers. It provides the following:
 * PINN solvers (1D solver and 2D monodomain) for experimentation
 * End-to-end PINN example of using the method of manufactured solutions to investigate convergence (numerical convergence to true solution of equations - not just training convergence)
 * Discussion of PINN-specific verification considerations and observations based on the convergence analysis results
 * Reusable code (in notebook and `src` folder)  


## Repository Structure
  * `notebooks`: Interactive Jupyter notebooks
  * `html_notebooks`: Static versions of the notebooks in HTML
  * `src`: source code, including calculation verification metrics module, and other functionality used by the notebooks 
  * `test`: Unit tests for `src` functions   
  * `saved_results`: pre-computed PINN convergence results 


## Installation

Clone the toolbox repository
```
git clone https://github.com/OSEL-DAM/simeval.git
```

There are two installation options depending on whether you want to run the PINN code.

### Option 1: Standard libraries only (no machine learning or PINN dependencies)

This is recommended if you do **not** want to run the PINN interactive notebook. It is sufficient for the other two notebooks and supporting code, i.e., code and calculation verification of traditional solvers.  

The following standard libraries are required
```
numpy
matplotlib
pandas
scipy
``` 
Development and testing were performed with versions :
 * python: 3.11.4
 * numpy: 1.26.4
 * matplotlib: 3.10.0
 * pandas: 2.2.3
 * scipy: 1.15.2


### Option 2: With DeepXDE and tensorflow (to run PINN notebook or PINN code)

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
python test_PINN_solve.py
python test_PINN_error_convergence.py
````


## Using the toolbox

The recommended starting point is to run the interactive notebooks in the `notebooks` folder, e.g.
```
jupyter notebook 
```
Each notebook contains detailed background and instructions. 


## Authors
**Developers**: Pras Pathmanathan, Kenny Aycock, Brent Craven

**Contact:** pras.pathmanathan@fda.hhs.gov


## Citation

If you use this toolbox in your research, please cite the accompanying paper: 
 * P. Pathmanathan, K. Aycock, B. Craven, "SimEval: a toolbox of software and interactive notebooks for credibility assessment of medical device modeling and simulation", under submission.
