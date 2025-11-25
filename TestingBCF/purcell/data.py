import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# The example spectral density used in [M. Lednev et al., PRL 132, 106902 (2024)]
# The data file is taken from https://github.com/jfeist/spectral_density_fit

# J(ω): 'Purcell.csv'
SD_name = 'purcell'

β = -1    # inverse temperature  # beta = -1 for the zero temperature

ωzero = 1e-10 # ω = 0 is replaced by ωzero
ωinf = 1e+1 # ω = ∞ is replaced by ωinf
ωintegral = np.linspace(ωzero, ωinf, int((ωinf-ωzero)/(1e-3))+1)    # ω-integral for Lt_exact and ηt

#
# spectral density
#

Purcell_data = np.loadtxt('purcell/Purcell.csv')
ω_data = Purcell_data[:,0]
J_data = np.pi * Purcell_data[:,1]  # π needs to be multiplied to be consistent with our definition of the spectral density (see Eq.1 in [M. Lednev et al., PRL 132, 106902 (2024)])
ωmax_data = np.max(ω_data)

# Ohmic near ω = 0, interpolating data using Akima1D, and exponential cutoff in the large ω region
Jinterp = Akima1DInterpolator(np.hstack(([0.], ω_data)), np.hstack(([0.], J_data)))  # J(0) = 0 is imposed

# the exponential cutoff with parameterse determined so that J(ω) and (d/dω)J(ω) are continuous at ω = ωmax_data
Jinterp_ωmax = Jinterp(ωmax_data)
δω = 1e-12; Jinterp_dot_ωmax = (Jinterp(ωmax_data) - Jinterp(ωmax_data-δω)) / δω # (d/dω)J(ω = ωmax_data)
cutoff = - Jinterp_dot_ωmax/Jinterp_ωmax

def J_(ω):   # ω >= 0 (faster)
    return np.where(ω <= ωmax_data, Jinterp(ω), Jinterp_ωmax*np.exp(-cutoff*(ω-ωmax_data)))

def J(ω):   # ω can be negative too
    ω = np.asarray(ω, float)
    out = np.sign(ω) * J_(np.abs(ω))
    return out.item() if np.isscalar(ω) else out

#
# get
#

def get_λ():
    return ηt(0.)/2.

def get_β():
    return β

def get_ωinf():
    return ωinf

def get_title():
    return f'_β{β}'

#
# correlation function
#

def Lω_exact(ω):
    # Use numpy's where to handle the array input
    if β == -1:
        return np.where(
            ω >= 0,
            2 * J(ω),  # This handles the case when ω >= 0
            0.  # Otherwise
        )
    elif β > 0:
        return np.where(
            ω == 0,
            Lω_exact_0(),   # This handles the case when ω == 0
            2 * J(ω) / (1 - np.exp(- β * ω))  # Otherwise
        )

def Lω_exact_0():  # Lω_exact(ω = 0) for β > 0
    return 2 * J_(ωzero) / (1 - np.exp(- β * ωzero))

def Lt_exact(t):
    t_max = np.max(t)

    def Lt_exact_value(t):
        print(f'{t:.1f} / {t_max}')

        # simpson
        if β > 0:
            return (1/np.pi) * simpson(J(ωintegral) * (np.cos(ωintegral*t) / np.tanh(β*ωintegral/2) - 1j * np.sin(ωintegral*t)), x=ωintegral)
        elif β == -1:
            return (1/np.pi) * simpson(J(ωintegral) * np.exp(-1j*ωintegral*t), x=ωintegral)

    if isinstance(t, (list, np.ndarray)):  # Check if u is a list or NumPy array
        return np.vectorize(Lt_exact_value)(t)
    else:
        return Lt_exact_value(t)

def ηt(t): # ηt: friction kernel in the time domain. Evaluated numerically

    def ηt_value(t):
        # simpson
        return (2/np.pi) * simpson(J(ωintegral) / ωintegral * np.cos(ωintegral*t), x=ωintegral)

    if isinstance(t, (list, np.ndarray)):  # Check if u is a list or NumPy array
        return np.vectorize(ηt_value)(t)
    else:
        return ηt_value(t)

def get_η(): # η: friction kernel in the Laplace domain
    η = None

#
# plot
#

def output_Lt():

    filename = f'{SD_name}/{get_title()}/data_time'

    t_max = 300
    dt = 1e-1
    t = np.arange(0., t_max, dt)
    Lt = Lt_exact(t)
    f = open(filename + '.csv', mode="w")
    for l in range(len(t)):
        f.write('{:8.3f}'.format(t[l]) + '{:30.5e}'.format(Lt[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(t,Lt.real,color='red',lw=3,label='Real')
    plt.plot(t,Lt.imag,color='blue',lw=3,label='Imag')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$L(t)$", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()


def output_Lω():

    filename = f'{SD_name}/{get_title()}/data_frequency'

    ω_min = -5
    ω_max = 10
    dω = 1e-3
    ω = np.arange(ω_min, ω_max, dω)
    Lω = Lω_exact(ω)
    f = open(filename + '.csv', mode="w")
    for l in range(len(ω)):
        f.write('{:8.3f}'.format(ω[l]) + '{:15.5e}'.format(Lω[l]) + '\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(ω,Lω,color='red',lw=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.ylabel(r"$\mathcal{F}[L](\omega)$", fontsize=15)
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename + '.png', format='png')
    plt.close()
