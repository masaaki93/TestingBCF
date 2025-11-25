# Bath Correlation Function Tests Using a Harmonic Oscillator System

This Python package provides a tool for testing model bath correlation functions (BCFs) used in open quantum dynamics simulations.
It benchmarks the accuracy of thermalization behavior against exact solutions obtained from surrogate harmonic oscillator systems.

The theoretical background and methodology are described in:

Masaaki Tokieda, *Testing bath correlation functions for open quantum dynamics simulations*, [Phys. Rev. Research 7, 043178 (2025).](https://doi.org/10.1103/bv19-dtb1)

[A companion repository](https://github.com/masaaki93/HOHEOM) is also available for running HEOM simulations in the moment representation for specified harmonic-oscillator parameters.

---

## Usage Overview

The full testing workflow involves four main steps:

1. **Prepare spectral density data**
2. **Construct model BCFs** (`fit.py`)
3. **Set system parameters and testing configurations** (`settings.py`)
4. **Run thermalization error evaluations** (`main.py`)

---
...
### Spectral Density Library

The package provides several standard spectral densities.
All spectral density parameters are defined in each directory’s `data.py` file.


**GMT (Generalized Maier-Tannor form)**  
```math
Im L(t) = Σ_k c[k] e^{-μ[k]}
```

including brownian
```math
J(ω) = ξ γ² ωb² ω / ((ω² − ωb²)² + (γ ω)²)
```

and drude
```math
J(ω) = ξ γ² ω / (γ² + ω²)
```

**expcutoff (Exponential cutoff)**  
```math
J(ω≧0) = (π/2) α ωc (ω/ωc)ˢ exp(−ω/ωc)
```

**circcutoff (Circular cutoff)**  
```math
J(ω≧0) = (π/2) α ωc (ω/ωc)ˢ [1 − (ω/ωc)²]ᵐ  θ(ωc−ω)
```

**expcutoff_filter** is an example taken from Sec. IV B 2 of the paper.

**purcell** is an example of numerical data taken from the GitHub repository [`spectral_density_fit`](https://github.com/jfeist/spectral_density_fit)

---

### Creating a Custom `data.py`

If your target spectral density is not included, you can create a new directory with a `data.py` file. This file should define the following required functions:

```python
def get_title():          # String describing the parameter set
def get_λ():              # Reorganization energy
def get_β():              # Inverse temperature (β = -1 for zero temperature)
def J(ω):                 # J(ω) over ℝ    
def output_Lt():          # Export BCF in the time domain (used in fit.py)
def output_Lω():          # Export BCF in the frequency domain (used in fit.py)
```

**Optional functions:**

Computing exact solutions using the analytic expression of η̂(s) or the integral formula:
```python
def get_η():              # Laplace transform of the friction kernel η̂(s) for Re(s) ≥ 0 (η == None if unavailable)
def J_(ω):                # J(ω) for ω ≥ 0 (used for the integral formula)
```

---

## Main Files Overview

### `fit.py`

Constructs a model BCF by fitting time-domain or frequency-domain data using a linear combination of complex exponentials.

- Input data is generated via `output_Lt()` and `output_Lω()` in `data.py`.
- Specify the spectral density via `SD()` in `settings.py`.
- Methodology is described in Sec. IV A of the paper.
- Fitting results are saved as shown [here](Lmod.pdf)

---

### `settings.py`

- `SD()` - choose the spectral density data directory
- `system()` - define the target system
- `test_config()` - define the testing configuration
- `hparams_exact()` - define hyperparameters for computing exact solutions

See each function’s docstring for details.

---

### `main.py`

For the system specified by `system()` in `settings.py`, `main.py` tests the validity of the fitted model BCFs using surrogate harmonic oscillator systems as described in Sec. IV A of the paper.

---

## Environment

This package has been tested with the following software versions:

- Python **3.14.0**
- NumPy **2.3.5**
- SciPy **1.16.3**
- Matplotlib **3.10.7**
- mpmath **1.3.0** (required for `expcutoff`)
