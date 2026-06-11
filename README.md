# Topology Optimization for 2D Mechanical Metamaterials

Python scripts for homogenization-based topology optimization of 2D auxetic mechanical metamaterials. The code uses FEniCS, dolfin-adjoint and IPOPT to generate periodic unit-cell geometries with targeted negative Poisson-type responses.

## Overview

This repository contains the numerical scripts developed for a master thesis project on topology optimization and additive manufacturing of auxetic mechanical metamaterials.

The aim of the code is to generate two-dimensional periodic unit-cell geometries whose effective mechanical response is controlled by the internal material distribution. Instead of starting from predefined auxetic geometries, the unit cells are generated through finite element homogenization and density-based topology optimization.

The scripts solve a periodic unit-cell problem, compute the homogenized in-plane stiffness matrix, and use selected stiffness components to drive the optimization toward a prescribed negative Poisson-type response.

## Method

The general workflow is:

1. Define a periodic 2D unit cell.
2. Initialize a continuous density field.
3. Apply density filtering and projection.
4. Interpolate material stiffness using SIMP-type material interpolation.
5. Solve three homogenization load cases.
6. Assemble the effective 2D stiffness matrix.
7. Evaluate the selected objective function.
8. Update the density field using gradient-based optimization.
9. Export the optimized density field and displacement solutions for visualization.

The homogenization is based on three independent macroscopic strain cases:

* axial strain in the x-direction,
* axial strain in the y-direction,
* in-plane shear strain.

The averaged stress responses from these cases are used to construct the effective stiffness matrix:

```text
C = [[C11, C12, C16],
     [C21, C22, C26],
     [C61, C62, C66]]
```

The Poisson-type response is evaluated from selected stiffness components. These values are used as numerical objective measures during optimization and should not be interpreted as direct experimental Poisson's ratio values.

## Included scripts

The repository contains three main optimization scripts.

| Script                             | Description                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `neo_hookean_direct_poisson.py`    | Neo-Hookean formulation using a direct Poisson-type objective based on `C12 / C22`.                    |
| `neo_hookean_symmetric_poisson.py` | Neo-Hookean formulation using a symmetric Poisson-type objective based on `0.5 * (C12 + C21) / C22`.   |
| `fourth_order_direct_poisson.py`   | Fourth-order elasticity tensor formulation using a direct Poisson-type objective based on `C12 / C22`. |

The Neo-Hookean scripts use a compressible hyperelastic material formulation. The fourth-order elasticity tensor script uses a stiffness-based elastic formulation.

All three scripts follow the same overall workflow, but differ in the constitutive formulation and/or the Poisson-type objective function.

## Numerical parameters

The main numerical parameters used in the final simulations are:

| Parameter                     | Value       |
| ----------------------------- | ----------- |
| Unit-cell size                | `1.0 x 1.0` |
| Mesh resolution               | `96 x 96`   |
| Fast-mode mesh resolution     | `48 x 48`   |
| Volume fraction               | `0.40`      |
| Minimum density               | `1e-3`      |
| Target Poisson-type response  | `-0.80`     |
| Macroscopic strain amplitude  | `1e-2`      |
| Base material Young's modulus | `1.0`       |
| Base material Poisson's ratio | `0.30`      |

The scripts also use continuation over the SIMP penalization and projection sharpness parameters to gradually drive the solution toward a clearer solid-void material distribution.

## Requirements

The scripts require a Python environment with:

* FEniCS / dolfin
* dolfin-adjoint
* IPOPT
* NumPy
* UFL
* MUMPS linear solver
* MPI support

The original simulations were run in a Docker-based FEniCS environment to improve reproducibility.

## Running the scripts

Run one of the scripts directly with Python:

```bash
python neo_hookean_direct_poisson.py
```

For a faster test run, use:

```bash
TOP_FAST=1 python neo_hookean_direct_poisson.py
```

Fast mode reduces the mesh resolution and the number of optimization iterations. This is useful for testing that the environment and dependencies are working before running the full-resolution optimization.

By default, the scripts can ask for the optimization mode interactively. To run without the interactive prompt, use:

```bash
TOP_INTERACTIVE=0 python neo_hookean_direct_poisson.py
```

## Output

Each run creates a timestamped output folder containing simulation results. The output can include:

* filtered density fields,
* thresholded density fields,
* displacement solutions from the homogenization cases,
* displacement solutions from verification cases,
* convergence information,
* final homogenized stiffness values.

The main visualization files are exported in `.pvd` format and can be opened in ParaView.

## Optimization modes

The scripts include three optimization modes:

| Mode            | Purpose                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `poisson_ratio` | Generates auxetic unit cells with a prescribed negative Poisson-type response. |
| `orthotropic`   | Generates directionally stiff material layouts.                                |
| `shear_stiff`   | Generates structures with increased shear stiffness.                           |

The final thesis structures were generated using the `poisson_ratio` mode.

## Notes on interpretation

The optimized Poisson-type values are objective-based numerical measures derived from the homogenized stiffness matrix. They are used to guide the topology optimization and compare different generated unit cells.

They should not be treated as direct experimental Poisson's ratio values. Experimental deformation measurements require separate mechanical testing and deformation analysis.

## Recommended repository structure

```text
.
├── README.md
├── scripts/
│   ├── neo_hookean_direct_poisson.py
│   ├── neo_hookean_symmetric_poisson.py
│   └── fourth_order_direct_poisson.py
├── results/
│   └── example_outputs/
└── docs/
    └── figures/
```

Large output files should preferably not be committed directly to the repository. Use a small example output or figures instead.

## Background

This code was developed as part of the master thesis:

**Topology Optimization and Additive Manufacturing of Auxetic Metamaterials**

The thesis investigates a complete workflow for generating, manufacturing and evaluating auxetic mechanical metamaterials. The computational part focuses on periodic unit-cell generation using finite element homogenization and topology optimization. Selected generated geometries were later reconstructed as CAD models and manufactured using additive manufacturing.

## Limitations

This repository contains research code developed for a thesis project. The scripts are intended to document and reproduce the numerical workflow used in the project, not to provide a general-purpose topology optimization package.

Some limitations are:

* the implementation is script-based,
* the simulations require a specific FEniCS/dolfin-adjoint environment,
* full-resolution optimization can be computationally expensive,
* objective-based Poisson values are not direct experimental measurements,
* geometry post-processing and CAD reconstruction are not fully automated in these scripts.

## Citation

If you use or refer to this code, please cite the related thesis or reference this repository.

## Author

Hugo Sörby
