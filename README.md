# Spectral Localizer for Topological Boundary Time Crystals

Numerical tools for studying spectral-localizer topology, operator-space
delocalization, and harmonic dynamics in dissipative boundary time crystals.

This repository accompanies the work

> D. Nemeth, A. Nazir, A. Principi, and R.-J. Slager,  
> *Topological Boundary Time Crystal Oscillations*,  
> arXiv:2602.17765 (2026).

The package provides numerical implementations for constructing collective-spin
Liouvillians, resolving their operator-space structure, computing spectral
localizer invariants, analysing Liouvillian eigenmodes, and reproducing the
main and supplemental numerical results of the associated work.

## Features

The project provides tools to:

- construct the BTC Liouvillian using QuTiP;
- organize operator space into angular-momentum rank sectors;
- compute spectral-localizer indices and localizer gaps;
- accelerate index calculations using LDL factorization and matrix inertia;
- resolve left and right Liouvillian eigenmodes along the emergent rank chain;
- generate topological-domain and complex-frequency-island datasets;
- simulate spin-correlation time series and harmonic scaling;
- reproduce the associated main and supplemental figures.

## Installation

Python 3.10 or newer is required.

### 1. Clone the repository

```bash
git clone https://github.com/d-nemeth/spectral-localizer-boundary-time-crystal.git
cd spectral-localizer-boundary-time-crystal
```

### 2. Create an environment

Using Conda:

```bash
conda create -n spectral-localizer python=3.11
conda activate spectral-localizer
```

Alternatively, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install the package

For the package and plotting dependencies:

```bash
python -m pip install -e ".[plots]"
```

For development tools as well:

```bash
python -m pip install -e ".[plots,dev]"
```

A pinned numerical environment can instead be installed with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Minimal example

```python
from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    localizer_gap_and_index,
    spectral_localizer,
)

params = BTCParams(N_spins=10, omega=1.0)

build_liouvillian = build_liouvillian_builder(params)
liouvillian = build_liouvillian(gamma=1.0)

_, rank_operator, _ = build_operator_space_coordinates(params)

localizer = spectral_localizer(
    liouvillian,
    rank_operator,
    lam0=0.0 + 0.0j,
    x0=1.0,
    kappa=1.0,
)

gap, index = localizer_gap_and_index(localizer)

print(f"localizer gap: {gap}")
print(f"localizer index: {index}")
```

## Physical model

The model consists of $N$ collectively coupled spin-$\frac{1}{2}$ particles in
the permutation-symmetric sector, with total angular momentum

```math
j = \frac{N}{2}.
```

Its density matrix evolves according to

```math
\dot{\rho}
=
-i[H,\rho]
+
\frac{\Gamma}{N}
\left(
J_- \rho J_+
-
\frac{1}{2}
\left\{
J_+J_-,
\rho
\right\}
\right).
```

where

```math
H = \Omega J_x.
```

Here, $\Omega$ is the coherent driving frequency and $\Gamma$ is the collective
dissipation strength. In the code, the collapse operator is

```math
C
=
\sqrt{\frac{\Gamma}{N}}\,J_-,
```

and QuTiP constructs the corresponding Liouvillian superoperator
$\mathcal L$.

The symmetric Hilbert-space dimension is $N+1$, so the Liouvillian acts on an
operator space of dimension $(N+1)^2$.

## Operator-space rank coordinate

The collective-spin adjoint generators are

```math
\mathcal K_\alpha
=
\operatorname{spre}(J_\alpha)
-
\operatorname{spost}(J_\alpha),
\qquad
\alpha \in \{x,y,z\}.
```

Their Casimir operator is

```math
\mathcal K^2
=
\mathcal K_x^2
+
\mathcal K_y^2
+
\mathcal K_z^2.
```

The eigenvectors of $\mathcal K^2$ can be labelled by tensor rank $k$ and
magnetic index $q$:

```math
\mathcal K^2
\lvert T_q^{(k)} \rangle\rangle
=
k(k+1)
\lvert T_q^{(k)} \rangle\rangle,
```

with

```math
k=0,1,\ldots,N,
\qquad
q=-k,-k+1,\ldots,k.
```

This decomposition turns operator space into an emergent $k$-chain. The
position operator used by the spectral localizer is

```math
\mathcal X
=
\sum_{k,q}
k\,
\lvert T_q^{(k)} \rangle\rangle
\langle\langle T_q^{(k)} \rvert.
```

Consequently, the eigenvalue of $\mathcal X$ identifies the tensor-rank
position of an operator-space state.

The package constructs this coordinate by diagonalizing $\mathcal K^2$,
matching its eigenvalues to $k(k+1)$, and resolving the $q$ labels using
$\mathcal K_z$.

## Spectral localizer

For a Liouvillian $\mathcal L$, operator-space coordinate $\mathcal X$,
reference position $x_0$, reference complex frequency $\lambda_0$, and
localizer strength $\kappa$, define

```math
A
=
\mathcal L
-
\lambda_0 I.
```

Its Hermitian real and imaginary parts are

```math
\operatorname{Re}A
=
\frac{A+A^\dagger}{2},
\qquad
\operatorname{Im}A
=
\frac{A-A^\dagger}{2i}.
```

The spectral localizer is

```math
L_{(x_0,\lambda_0)}(\mathcal L,\mathcal X)
=
\operatorname{Re}A \otimes \sigma_x
+
\operatorname{Im}A \otimes \sigma_y
+
\kappa(\mathcal X-x_0I)\otimes\sigma_z.
```

It is Hermitian and has dimension $2(N+1)^2$.

### Localizer index

The localizer index is half of the matrix signature:

```math
\nu^L_{(x_0,\lambda_0)}
=
\frac{1}{2}
\operatorname{sig}
\left[
L_{(x_0,\lambda_0)}
\right].
```

where

```math
\operatorname{sig}(L)
=
n_+(L)-n_-(L),
```

and $n_+$ and $n_-$ are the numbers of positive and negative eigenvalues,
respectively.

A change in $\nu^L$ marks a topological boundary in either the rank coordinate
$x_0$ or the complex-frequency plane $\lambda_0$.

### Localizer gap

The localizer gap is

```math
\mu_{(x_0,\lambda_0)}
=
\min
\left\{
|\ell|
:
\ell
\in
\operatorname{spec}
\left(
L_{(x_0,\lambda_0)}
\right)
\right\}.
```

The gap closes when an eigenvalue of the localizer crosses zero, allowing the
index to change.

## Standard and accelerated implementations

Two spectral-localizer implementations are included.

### Standard implementation

`standard_localizer.py` explicitly constructs the Hermitian localizer and uses
a full Hermitian eigensolver. It returns both the localizer gap and the index
and serves as the reference implementation.

### Fast LDL implementation

`fast_localizer.py` evaluates the index through an LDL factorization. In block
form, the implementation works with

```math
L_{\mathrm{block}}
=
\begin{pmatrix}
\kappa(\mathcal X-x_0I) & A \\
A^\dagger & -\kappa(\mathcal X-x_0I)
\end{pmatrix}.
```

For a Hermitian matrix $H$, an LDL factorization gives

```math
H
=
LDL^\dagger,
```

where $L$ is triangular and $D$ is block diagonal, with $1\times1$ and
$2\times2$ pivot blocks.

Sylvester's law of inertia states that a congruence transformation preserves
the numbers of positive, negative, and zero eigenvalues. Therefore,

```math
\operatorname{inertia}(H)
=
\operatorname{inertia}(D).
```

The signature can therefore be found by inspecting the small blocks of $D$,
avoiding a full diagonalization of the localizer. The package handles the sign
convention associated with the chosen block ordering internally.

The accelerated implementation also supports:

- adaptive one-dimensional refinement near index changes;
- inexpensive updates when only $x_0$ changes;
- inexpensive updates when only $\lambda_0$ changes;
- parallel coarse-grid and refined-cell calculations for complex-frequency
  scans.

## Liouvillian eigenmode delocalization

For a right or left Liouvillian eigenmode
$\lvert \psi_a \rangle\rangle$, its weight in tensor-rank sector $k$ is

```math
w_k^{(a)}
=
\sum_{q=-k}^{k}
\left|
\langle\langle
T_q^{(k)}
\rvert
\psi_a
\rangle\rangle
\right|^2.
```

The numerical profiles are normalized so that

```math
\sum_k w_k^{(a)}
=
1.
```

These distributions quantify how strongly an eigenmode is localized or
delocalized along the emergent $k$-chain. The package computes the profiles
for both left and right Liouvillian eigenvectors.

## Package modules

The installable Python package is located under `src/`.

| Module | Purpose |
| --- | --- |
| `spectral_localizer.btc_model` | BTC parameters, collective-spin operators, Liouvillian construction, and operator-space coordinates |
| `spectral_localizer.standard_localizer` | Direct localizer construction, gap, and signature-based index |
| `spectral_localizer.fast_localizer` | LDL inertia calculation and adaptive rank-coordinate sweeps |
| `spectral_localizer.kq_basis` | Construction of the spherical-tensor $(k,q)$ basis |
| `spectral_localizer.mode_tools` | Left/right eigensystems and tensor-rank weight profiles |
| `spectral_localizer.mode_table` | Sorting, displaying, and saving Liouvillian eigenvalue tables |
| `run_utils.run_manager` | Creation of sequential `run_###` output directories |

The most commonly used classes and functions are re-exported from
`spectral_localizer`.

## Repository structure

```text
spectral-localizer-boundary-time-crystal/
├── generate/
│   ├── delocalization/
│   │   ├── generate_delocalization.py
│   │   └── plot_delocalization.py
│   ├── diagnostics/
│   │   ├── generate_diagnostics.py
│   │   ├── generate_localizer_islands_diagnostics.py
│   │   ├── generate_mode_table.py
│   │   ├── plot_diagnostics.py
│   │   └── plot_localizer_islands_diagnostics.py
│   ├── spin_correlations/
│   │   ├── generate_scaling_data.py
│   │   ├── generate_time_series.py
│   │   ├── plot_exponents.py
│   │   ├── plot_harmonics.py
│   │   ├── plot_mixed_state_analytical_vs_numerics.py
│   │   ├── plot_saturation.py
│   │   ├── plot_scaling.py
│   │   ├── plot_spin_correlations.py
│   │   ├── plot_time_series.py
│   │   └── plot_time_series_ffts.py
│   └── topology/
│       ├── domains/
│       │   ├── generate_topological_domains.py
│       │   └── plot_topological_domains.py
│       └── islands/
│           ├── generate_topological_islands.py
│           └── plot_topological_islands.py
├── main_figures_datasets/
│   ├── delocalization/
│   ├── eigenmode_tables/
│   ├── topological_domains/
│   └── topological_islands/
├── supplemental_figures_datasets/
│   ├── localizer_islands/
│   ├── localizer_scans/
│   ├── spin_correlations_scaling_data/
│   └── spin_correlations_time_series_data/
├── src/
│   ├── run_utils/
│   │   └── run_manager.py
│   └── spectral_localizer/
│       ├── __init__.py
│       ├── btc_model.py
│       ├── fast_localizer.py
│       ├── kq_basis.py
│       ├── mode_table.py
│       ├── mode_tools.py
│       └── standard_localizer.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Generating results

Run scripts from the repository root so that relative input and output paths
resolve correctly.

The scripts currently use configuration constants near the top of each file
rather than command-line arguments. Review values such as `N_SPINS`, `KAPPA`,
`N_JOBS`, parameter grids, and `RUN` before execution.

### Output convention

Dataset generators create the next available numbered directory:

```text
<dataset-directory>/
└── run_###/
    ├── data.npz or data.pkl
    ├── info.json
    └── figures/
```

`info.json` records the numerical parameters and runtime metadata. Plotting
scripts load an existing dataset selected by their `RUN` constant and save
figures inside that run's `figures/` directory.

Some default calculations use dense Liouvillians, large localizers, fine
two-dimensional grids, or several parallel workers. Check the parameters
before running them on machines with limited memory.

## Main-figure workflows

### Topological domains along the rank chain

Generate the spectral-localizer index as a function of $x_0$:

```bash
python generate/topology/domains/generate_topological_domains.py
```

Select the generated run in `plot_topological_domains.py`, then plot it:

```bash
python generate/topology/domains/plot_topological_domains.py
```

### Topological islands in the complex-frequency plane

Generate adaptive maps of $\nu^L(x_0,\lambda_0)$ for selected dissipation
strengths:

```bash
python generate/topology/islands/generate_topological_islands.py
```

Select the run and plot the index maps together with the Liouvillian spectrum:

```bash
python generate/topology/islands/plot_topological_islands.py
```

### Left and right eigenmode delocalization

Generate tensor-rank profiles:

```bash
python generate/delocalization/generate_delocalization.py
```

Select the run and plot the left/right profiles:

```bash
python generate/delocalization/plot_delocalization.py
```

### Liouvillian eigenmode tables

Generate sorted CSV and plain-text eigenvalue tables:

```bash
python generate/diagnostics/generate_mode_table.py
```

The tables are written to `main_figures_datasets/eigenmode_tables/`.

## Supplemental diagnostics

### One-dimensional localizer scans

`generate_diagnostics.py` supports dissipation-strength and localizer-strength
scans. Set its `MODE` constant to either `gamma_scan` or `kappa_scan`, then run:

```bash
python generate/diagnostics/generate_diagnostics.py
```

Plot a selected run with:

```bash
python generate/diagnostics/plot_diagnostics.py
```

The resulting panels show both the index $\nu^L(x_0)$ and localizer gap
$\mu(x_0)$.

### Localizer-island convergence with $\kappa$

Generate complex-frequency maps for several localizer strengths:

```bash
python generate/diagnostics/generate_localizer_islands_diagnostics.py
```

Plot the index and gap maps:

```bash
python generate/diagnostics/plot_localizer_islands_diagnostics.py
```

## Spin-correlation workflows

Generate representative time-series data:

```bash
python generate/spin_correlations/generate_time_series.py
```

Generate finite-size and dissipation-scaling data:

```bash
python generate/spin_correlations/generate_scaling_data.py
```

The accompanying plotting scripts produce:

- time-domain rank-$1$, rank-$2$, and rank-$3$ signals;
- Fourier spectra and harmonic peak locations;
- amplitude scaling with dissipation;
- fitted scaling exponents;
- finite-size saturation plots;
- comparisons between numerical and analytical mixed-state dynamics.

Run the desired plotting script after setting its `RUN` constant to a
compatible dataset.

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@misc{nemeth2026topologicalboundarytimecrystal,
  title         = {Topological Boundary Time Crystal Oscillations},
  author        = {Dominik Nemeth and Ahsan Nazir and Alessandro Principi and Robert-Jan Slager},
  year          = {2026},
  eprint        = {2602.17765},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2602.17765}
}
```

## License

The Python package metadata declares this project under the MIT License.