# qcproject_toolkit

`qcproject_toolkit` is a Python-based experimental framework designed to automate transition state (TS) search workflows in chemical and biochemical systems. 

This toolkit implements an algorithmic approach to **combinatorial selection of relaxation paths**, developed as part of the research thesis: *"Transition State Search in Chemical and Biochemical Reactions via Combinatorial Selection of Relaxation Paths"*.

## Core Functionality
The framework automates the interaction between computational chemistry software (e.g., MOPAC) and custom optimization algorithms to identify Minimum Energy Paths (MEP).

- **Automated Workflow**: Orchestrates the interaction between Python scripts and external quantum chemistry calculation engines.
- **Path Optimization**: Implements combinatorial selection to refine relaxation paths, reducing manual intervention in TS search.
- **Data Pipelines**: Streamlines the parsing, storage, and analysis of large datasets generated during the reaction modeling process.
- **Extensible Architecture**: Designed to allow integration of different quantum chemistry packages.

## Prerequisites
The toolkit requires a Python 3.x environment and access to the relevant quantum chemistry software (e.g., MOPAC) installed on your system.

- **Python Libraries**: `numpy`, `scipy`, `pandas`
- **Dependencies**: Ensure the path to your calculation engine (e.g., `mopac.exe` or equivalent) is correctly configured in `config.py`.

## Installation

```bash
# Clone the repository
git clone [https://github.com/Sebastianrn3/qcproject_toolkit.git](https://github.com/Sebastianrn3/qcproject_toolkit.git)

# Navigate to the directory
cd qcproject_toolkit

# Install dependencies
pip install -r requirements.txt
