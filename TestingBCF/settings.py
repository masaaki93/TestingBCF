import numpy as np

# =============================================================================
# Spectral density information
# =============================================================================
def SD():
    """
    Spectral density selector.

    Returns
    -------
    SD_name : str
        Name of the folder where `data.py` live.
    """
    SD_name = "expcutoff"
    # SD_name = "GMT"
    # SD_name = "circcutoff"
    # SD_name = "expcutoff_filter"
    # SD_name = "purcell"
    return SD_name

# =============================================================================
# Main system information
# =============================================================================
def system():
    """
    Define the main system.

    Returns
    -------
    system_name : str
        Used as a folder under `{SD_name}/{title}/{method_dir}/{system_name}` for saving test results.
    H_S : np.ndarray (complex)
        System Hamiltonian with the convention H_tot = H_S + H_B + V_S ⊗ X_B.
    V_S : np.ndarray (complex)
        System coupling operator in the interaction term V_S ⊗ X_B.
    """

    # Example: two-spin system
    system_name = 'two-spin'

    def system_operators():
        σx = np.array([[0, 1], [1, 0]])
        σy = np.array([[0, -1j], [1j, 0]])
        σz = np.array([[1, 0], [0, -1]])

        σ1x = np.kron(σx, np.eye(2))
        σ1y = np.kron(σy, np.eye(2))
        σ1z = np.kron(σz, np.eye(2))

        σ2x = np.kron(np.eye(2), σx)
        σ2y = np.kron(np.eye(2), σy)
        σ2z = np.kron(np.eye(2), σz)

        return σ1x, σ1y, σ1z, σ2x, σ2y, σ2z

    ω1 = 1.2; ω2 = 0.8; g = 0.4
    σ1x, σ1y, σ1z, σ2x, σ2y, σ2z = system_operators()

    H_S = (ω1/2) * σ1z + (ω2/2) * σ2z + g * σ1x @ σ2x
    V_S = σ1x + σ2x

    return system_name, H_S, V_S

# =============================================================================
# Configuration for testing
# =============================================================================
def test_config():
    """
    Hyperparameters for testing.

    Files expected
    --------------
    Model BCF parameters are saved as:
        f"{SD_name}/{title}/{method_dir}/{method_name}_K*.csv"

    Returns
    -------
    method_dir : str
        Directory name under SD where the fitting outputs are stored.
    method_name : str
        Base filename prefix for saved parameter sets (often identical to `method_dir`).
    K_values : list[float] | None
        Specific K-values to test. If None, test all files matching "{method_name}_K*.csv".
    compare_corr : bool
        If True, compare correlation functions F[C_oo](ω) (o = q,p).
        If False, compare only equilibrium expectation values ⟨o²⟩_eq (o = q,p).
    ω_corr : np.ndarray | None
        Frequency grid for comparing spectra. Needed if `compare_corr` is True.
        If None, ω_corr = np.linspace(-5*ω0, 5*ω0, 500).
    dt_corr : float
        Time step used for the Fourier transform. Needed if `compare_corr` is True.
    """
    method_dir = "ESPRIT"
    method_name = method_dir
    K_values = None
    compare_corr = False
    ω_corr = None
    dt_corr = 1e-1
    return method_dir, method_name, K_values, compare_corr, ω_corr, dt_corr

# =============================================================================
# Hyperparameters for exact-solution routines
# =============================================================================
def hparams_exact():
    """
    Hyperparameters for exact-solution evaluation.

    Returns
    -------
    tols : np.ndarray (float)
        Convergence tolerances for evaluating ⟨o²⟩_eq (o=q,p) from infinite-series expressions.
        Provide multiple values to check convergence robustness.
    Kj : int | None
        ESPRIT order used when fitting Im L(t). Larger Kj generally improves fidelity.
        This is necessary to estimate the time at which the steady state is reached.
        If you hit `ValueError`, try reducing Kj (or increasing the time range in `data_time.csv`).
        # Use None if you want the code to auto-select (prone to `ValueError` for complex spectral densities) (global search and hence not efficient).
        Use None if you want the code to auto-select (global search and hence not efficient).
    GMT : bool
        If True, compute η̂(s) using the fit of Im L(t).
        If False, use the method designated by `get_η` in `data.py`.
    ωmax : float | None
        The upper limit of the integral for evaluating η̂(s) (J(ω >= ωmax) = 0 is assumed).
        Use None if GMT is True or `get_η` in `data.py` is not None.
    """
    tols = np.array([1e-8, 1e-10])
    Kj = None
    # Kj = 30
    GMT = True
    ωmax = 1e+1
    return tols, Kj, GMT, ωmax
