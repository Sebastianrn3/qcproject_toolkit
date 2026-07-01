# ⚛️ qcproject_toolkit

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Automated Minimum Energy Path (MEP) Search & Transition State Optimization**

`qcproject_toolkit` is a high-performance Python library designed to automate the search for reaction pathways and transition states (TS) in both small-molecule chemical catalysis and complex enzymatic QM/MM systems. 

Bridging the gap between computationally cheap semi-empirical prototyping and highly accurate DFT/ab-initio refinements, this toolkit allows researchers to rapidly build, optimize, and analyze multidimensional Potential Energy Surfaces (PES).

## 🚀 Key Features

* **Automated Initial Path Generation:** Utilizes interpolation methods (e.g., IDPP) to generate chemically sensible initial pathways, avoiding unphysical atomic clashes before optimization starts.
* **Advanced Chain-of-States Optimization:** Implements robust Nudged Elastic Band (NEB) algorithms alongside specialized variations:
  * **Zoom-NEB (Z-NEB):** Focuses computational effort on the active site.
  * **Energy-weighted springs (NEB-TS):** Improves resolution near the transition state.
* **Enzyme & QM/MM Ready:** Specifically engineered to handle the "spectator degrees of freedom" (spectator DOF) problem commonly encountered when dealing with bulk solvent and massive protein environments.
* **Performance & Stability:** Designed to seamlessly integrate with local TS search algorithms (like eigenvector-following or dimer methods) after the initial MEP is obtained.

## 📦 Installation

Clone the repository and install the dependencies directly via pip:

```bash
git clone [https://github.com/Sebastianrn3/qcproject_toolkit.git](https://github.com/Sebastianrn3/qcproject_toolkit.git)
cd qcproject_toolkit
pip install .
