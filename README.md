# qcproject_toolkit

`qcproject_toolkit` is a Python-based computational chemistry toolset designed to investigate alternative approaches reaction mechanism modeling problem: the generation of physically sound initial pathways.

Determining Transition States (TS) and Minimum Energy Paths (MEP) in sterically constrained systems—such as complex enzyme-ligand active centers—is a computationally challenging task. Standard interpolation methods like Linear Interpolation or IDPP (Image Dependent Pair Potential) frequently propose unphysical atomic dynamics in both structurally and mechanistically complicated enzyme environments. This inevitably results in common convergence failures when passed to double-ended optimization algorithms like the Nudged Elastic Band (NEB), severely limiting their practical application.

`qcproject_toolkit` implements a heuristic approach based on the **combinatorial selection of relaxation trajectories** and the formal mathematical definition of the MEP. Instead of forcing a rigid interpolation, the toolkit:
1. Generates an initial sequence of discrete images between reaction start and end.
2. Independently relaxes these structures on the Potential Energy Surface (PES).
3. Filters, evaluates all possible trajectory combinations from sampled pool of partially relaxed states.
4. Selects reaction path guess with the best score based on gradient compatibility and spatial distribution uniformity.
5. Examines and compares NEB results obtained using choosen guesses.

From a framework perspective, `qcproject_toolkit` provides an end-to-end pipeline—starting from two endpoint molecular states and concluding with a proposed reaction mechanism—while allowing to customize over 50 quantitative and qualitative variables.

* **Focus on Fast Semi-Empirical Evaluations:** Designed with accessibility in mind. You can perform complex runs using computing resources affordable for an very average personal PC or laptop. A single energy and gradient evaluation for a 100–300 atom geometry takes just seconds (or 1,000–10,000 times faster than popular DFT B3LYP). Consequently, a complete NEB simulation (thousands of evaluations) for an enzyme reaction center is possible overnight job. While semi-empirical methods miss exact quantitative barrier heights, they provide qualitatively accurate mechanistic insights—perfect for exploratory runs before investing in expensive *ab initio* methods.
* **Customizable Optimization:** Easily adjust parameters to choose compromise between calculation speed and result reliability.
* **Modular Scenarios:** Highly configurable stages throughout the workflow, allowing you to select different algorithmic methods for the most impactful steps.
* **Accessible Result Interpretation:** Includes built-in tools for generating multiple plot types and multi-frame `.xyz` trajectories. These can be easily visualized in Avogadro for a clear, understandable glimpse of the reaction mechanism, complete with intermediate stages for testing and debugging.

## Table of Contents
- [I/O and General Pipeline](#io-and-general-pipeline)
- [Core Features](#core-features)
- [Molecular State Objects & Units](#molecular-state-objects--units)
- [Installation](#installation)
- [Usage Example](#usage-example)
- [Academic Background & Citation](#academic-background--citation)
- [License](#license)

---

## I/O and General Pipeline

### Starting Inputs:
* **Stationary Geometries:** Reaction endpoints in `.xyz` format (Cartesian coordinates of atoms in reactant and product states).
* **Physical Parameters:** Total charge of the system, dielectric constant of the environment, fixed (frozen) atoms, rigid groups, etc.
* **Procedure Parameters:** Job selection, stages to execute, guess generation methods, chain bead (image) count, combinatorial selection settings, and NEB configuration (optimization algorithm, convergence criteria).

### Final Results:
* **Optimized Geometries:** Discrete sets of coordinates representing the optimal reaction paths found after NEB optimization for the chosen guess methods.
* **Execution Logs:** Detailed job logs, particularly concerning combinatorial sampling progress and NEB convergence overviews.
* **Analytics:** Summarizing statistics, comparative metrics between guess methods, and energy profiles.

### The 7-Stage Workflow
The full modeling cycle consists of 7 consecutive stages designed to overcome the limitations of standard interpolation:

1. **Endpoint Preparation:** Stationary points (Reactants and Products) are aligned and completely relaxed to their local minima.
2. **Initial Guess Generation:** A primary chain of intermediate images (interpolants) is generated between the endpoints using the chosen method.
3. **Intermediate Image Relaxation & Sampling:** Interpolants are independently relaxed on the Potential Energy Surface (PES).
4. **Combinatorial Path Selection (Core Innovation):** Filtered optimization trajectories are sampled to form a state matrix. All possible image combinations (paths) connecting the endpoints are evaluated, returning the best combination with the maximum physical score.
5. **Reparametrization:** The winning chain is re-discretized to a higher resolution (e.g., 15 images). 
6. **NEB Optimization:** The refined initial guess is submitted to the NEB optimizer to find the true MEP.
7. **Final Reporting:** Generation of final MEP energetic profiles, path geometries, and NEB optimization statistics.

### Supported Guess Protocols
Guesses are generated in 3 possible protocols:
* **Mode 0 (Standard):** Interpolated only (Stages 1-2 and 6-7). Used strictly as a baseline reference.
* **Mode 1 ("Wave 1"):** The core protocol of this toolkit. Integrates all stages (1-7), modifying the standard guess via partial chain relaxation and combinatorial selection. Evaluating the practical advantage of this method is the primary reason `qcproject_toolkit` was created.
* **Mode 2 ("Wave 2"):** A deeper modification where the Wave 1 guess is subjected to an additional refinement cycle (repeating stages 3-5). It incurs the highest computational costs with the least predictable outcomes and is generally reserved for highly complex barriers.

---

## Core Features

### Overview
Beyond its internal author-scripted pipeline and standard Python libraries, `qcproject_toolkit` relies on addressing the external **MOPAC** computational chemistry software (for semi-empirical energy and gradient evaluations) and the **ASE (Atomic Simulation Environment)** Python library for NEB optimization.

### Job Attributes
A "Job" represents a specific molecular model assigned with:
1. **Operating Directory:** Located in `data/jobs/`, containing at least the start and end (R and P) geometries in `.xyz` format.
2. **Configuration File:** An assigned `.py` file in the `settings/` directory defining:
   * Job ID
   * Total electric charge & unpaired electron count
   * Fixed (frozen) atom indices and defined rigid groups (optional)
   * Associated system directories
3. **Run Protocol:** An execution script in `runs/` that specifies the list of stationary points (Reactants, Products, optional Intermediates). Runs are performed strictly for each separate reaction phase between neighboring points.

### Core Utilities
* **I/O Handling:** Functions for `.xyz` and `.npz` saves, `.mop` input generation, MOPAC initialization, and `.out` parsing.
* **Endpoint Preparation:** Atomic compatibility checks, Kabsch alignment, and complete geometry optimization.
* **Geometry Optimization:** Unit image relaxation is performed using the LBFGS algorithm (used for both endpoint preparation and independent interpolant relaxation).
* **Interpolation:** Generates a discrete chain of $N$ images (default: 9 total; 2 endpoints + 7 interpolants). Methods include Linear (default for reparametrization) and IDPP (default for the initial chain). 

### Custom Guess Preparation (Combinatorial Engine)
* **Intermediates Relaxation:** Each interpolant independently relaxes to its local stationary minimum in the PES.
* **Trajectory Filtering:** Cleans optimization trajectories from outliers.
* **Pool Sampling:** $N$ representative images are sampled from each trajectory to define a Groups matrix with the endpoints.
* **Combinatorial Selection:** Evaluates every possible image chain (unique combination from the Groups matrix).
* **Zooming Refinement:** Additional cycles of detailed resampling for trajectory points closest to the current best combination.
* **Brute-force Chain Scoring:** Returns the criterion $S_{total}$ of the proposed path. This combines a physically justified **Cosine Score** (measuring affinity to the MEP, derived from the mathematical definition that gradient and tangent vectors must be collinear) and a **Spatial Distribution Penalty** (controlled by a $\lambda$ multiplier, preferring chains with even geometric distribution).
* **High-Performance Computing:** Because combinatorial complexity scales exponentially $O(K^{N-2})$, the scoring engine uses Just-In-Time (JIT) compilation to C code via Numba and parallel processing across multiple CPU cores.

### Guesses NEB Examination
For each job, the MEP search evaluates 7 starting chains by default:
* 1x Standard (IDPP) guess
* 3x "Wave 1" protocol guesses (using $\lambda$ penalties of 0.2, 0.5, and 1.0)
* 3x "Wave 2" protocol guesses (using the same $\lambda$ variants)
*(Note: Smaller $\lambda$ values lead to bolder/riskier guesses. Wave 2 carries double the preparation cost relative to Wave 1).*

---

## Molecular State Objects & Units

The core objects of the protocol are **Images** and **Chains**, both implemented as Python dictionaries.

* **Single Image:** Primarily defined by its geometry (NumPy 2D array `[atoms, xyz]`), its PM7 evaluated Energy (float, enthalpy of formation), and its Gradients (first derivative of energy). The resulting object encapsulates `(geometry, E, grad)` along with facultative metadata.
* **Chain:** A series of $N$ images (NumPy 3D array `[image, atoms, xyz]`). Physically, it represents a geometric path of the model between stationary points, or a reaction mechanism.

**Unit Conversions:** All internal calculations are strictly performed in **Atomic Units** (Bohr, Hartree). The toolkit automatically handles conversions for external software:
* `.xyz` format: Angstroms (Å)
* ASE NEB: Angstroms (Å) and Electronvolts (eV)
* MOPAC: Angstroms (Å) and kcal/mol

---

## Installation

To install `qcproject_toolkit`, clone the repository and install it directly via `pip`. Using a virtual environment is highly recommended. Ensure you have MOPAC installed and accessible in your system's PATH.

```bash
# Clone the repository
git clone [https://github.com/Sebastianrn3/qcproject_toolkit.git](https://github.com/Sebastianrn3/qcproject_toolkit.git)
cd qcproject_toolkit

# Install the package and Python dependencies
pip install .
