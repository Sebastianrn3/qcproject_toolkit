# QCProject Toolkit: MEP Search in Enzymatic Catalysis

Computational toolkit for locating and optimizing Minimum Energy Paths (MEP) in high-dimensional potential energy surfaces (PES), specifically optimized for enzymatic systems.

## Abstract

Standard interpolation methods often fail in complex biochemical systems due to atomic overlaps and high energy barriers in the initial guess. This toolkit implements a multi-stage combinatorial search protocol to automate the generation of robust initial trajectories for Nudged Elastic Band (NEB) calculations.

## Core Methodology

- **Backend:** Integration with MOPAC 23.2.2 (PM7 semi-empirical method).
- **Optimization:** Nudged Elastic Band (NEB) implementation via Atomic Simulation Environment (ASE).
- **Algorithm:** A multi-step combinatorial search that relaxes images independently and reconstructs the optimal path based on gradient alignment and geometric evenness.
- **Interpolation:** Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) for trajectory smoothing.

## Key Features

- **Combinatorial Path Selection:** Evaluates potential paths using a scoring function (Scos and Peven).
- **Adaptive Refinement:** Implementation of "Zoom-in" laps for localized precision in transition state regions.
- **Validation:** Benchmarked against enzymatic systems including L-aspartate alpha-decarboxylase, CocE, HheC, SET7/9, and 4-OT.

## Dependencies

- Python 3.10+
- ASE (Atomic Simulation Environment)
- SciPy / NumPy
- MOPAC 23 (External executable)

## Attribution

Developed by **Sebastijonas Valaitis**
