# Physics Background

Boundary time crystals (BTCs) are non-equilibrium phases that emerge in open
many-body quantum systems due to the interplay of coherent dynamics and
dissipation.

The **spectral localizer** provides a powerful diagnostic tool for identifying
topological structure in non-Hermitian operators such as Liouvillians.
In this project, the spectral localizer is applied in **operator space**
to characterize the structure of Liouvillian eigenmodes associated with
boundary time crystal oscillations.

The methods implemented here allow one to:

- compute spectral localizer indices for Liouvillian superoperators
- visualize topological domains across an emergent chain
- visualize topological islands in the complex frequency plane

# Citation

If you use our work or build upon our code please cite our work:

```text
@misc{nemeth2026topologicalboundarytimecrystal,
      title={Topological Boundary Time Crystal Oscillations}, 
      author={Dominik Nemeth and Ahsan Nazir and Alessandro Principi and Robert-Jan Slager},
      year={2026},
      eprint={2602.17765},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2602.17765}, 
}
```

# Spectral Localizer for Boundary Time Crystals

The code implements both standard and accelerated methods for evaluating localizer indices and visualizing the resulting topological structure of Liouvillian spectra.

The project provides tools for:

- Constructing **Liouvillian superoperators** for collective spin models (via QuTiP)
- Computing **spectral localizer indices** in operator space
- Visualizing **topological domains in the complex Liouvillian spectrum**
- Performing **efficient parameter sweeps** using fast LDL-based inertia methods

The code was developed to support research on **topological boundary time crystal oscillations** and related phenomena in dissipative quantum many-body systems.

# Features

The repository contains implementations of:

## Liouvillian Construction

- Collective spin models for open quantum systems
- Weak-coupling Markovian (Lindblad) dynamics

## Spectral Localizer Methods

Two implementations are provided.

### Standard Localizer
- Direct construction of the spectral localizer
- Robust and straightforward

### Fast Localizer
- Uses **LDL inertia factorization**
- Avoids full diagonalization
- Enables efficient large parameter sweeps

## Diagnostics

Tools for computing:

- Localizer index as a function of:
  - dissipation strength
  - rank coordinate
  - localizer strength

- 2D spectral maps showing **topological domains ("islands")**

## Analysis

### Standard
- Topological domains across the emergent k-chain for the steady-state (Fig. 2c): `topological_domains.py`
- Eigenmode delocalization across the chain (Fig. 4): `eigenmode_delocalization.py`
- Topological islands in the complex-frequency plane (Fig. 3): `topological_islands.py`

### Fast LDL optimized
- Topological domains across the emergent k-chain for the steady-state (Fig. 2c): `fast_topological_domains.py`
- Topological islands in the complex-frequency plane (Fig. 3): `fast_topological_islands.py`


# Installation

## 1. Clone the repository

Recommended using **conda**:

```bash
git clone <d-nemeth/spectral-localizer-boundary-time-crystal>
```

## 2. Create a Python environment
```bash
conda create -n spectral_localizer python=3.11
conda activate spectral_localizer
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the package

```bash
pip install -e .
```

# Project Structure

```text
spectral-localizer
├── analysis
│   ├── fast
│   │   ├── fast_topological_domains.py
│   │   └── fast_topological_islands.py
│   └── standard
│       ├── eigenmode_delocalization.py
│       ├── topological_domains.py
│       └── topological_islands.py
├── diagnostics
│   ├── kappa_sweep_complex_plane.py
│   ├── vary_gamma_1d_scan.py
│   └── vary_kappa_1d_scan.py
├── notebooks
│   └── spectral_localizer_notebook.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
     └── spectral_localizer
        ├── __init__.py
        ├── __pycache__
        ├── btc_model.py
        ├── fast_localizer.py
        ├── kq_basis.py
        ├── mode_tools.py
        └── standard_localizer.py
```

# Running Diagnostics

## Topological Domains

- Compute the spectral localizer index as a function of dissipation strength using `python diagnostics/vary_gamma_1d_scan.py`.

- Compute the spectral localizer index as a function of the localizer strength using `python diagnostics/vary_kappa_1d_scan.py`.

## Topological Islands

- Compute the localizer index as a function of the localizer strength using `python diagnostics/kappa_sweep_complex_plane.py`.








