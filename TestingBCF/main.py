import numpy as np
import importlib
import os, re
import input.expfit as expfit
import input.expfit_GMT as expfit_GMT
import input.exact as exact
from scipy.integrate import simpson

import time

from input.heom import HEOMSolver
import matplotlib.pyplot as plt

import settings as settings
import input.omega0v0 as omega0v0

# =============================================================================
# Input
# =============================================================================

### spectral density name
SD_name = settings.SD()

### input data
module_name = f'{SD_name}.data'
data = importlib.import_module(module_name)
λ = data.get_λ(); # print(f'λ={λ}')
β = data.get_β()
dir_name = f'{SD_name}/{data.get_title()}'

Lt_filename = f'{dir_name}/data_time.csv'
Lt_data = np.loadtxt(Lt_filename, dtype=complex)
t = Lt_data[:,0].real; Lt = Lt_data[:,1]

Lω_filename = f'{dir_name}/data_frequency.csv'
Lω_data = np.loadtxt(Lω_filename)
ω = Lω_data[:,0]; Lω = Lω_data[:,1]

### Configuration for testing
method_dir, method_name, K_values, compare_corr, ω_corr, dt_corr = settings.test_config()

if K_values == None:
    base_names = set()
    for filename in os.listdir(f'{dir_name}/{method_dir}'):
        match = re.match('(' + method_name + r'_K\d+)', filename)
        if match:
            base_names.add(match.group(1))
    filename = sorted(base_names, key=lambda x: int(re.search(r'\d+', x).group()))
    results_name = [f'{method_dir}/{filename[i]}' for i in range(len(filename))]
else:
    results_name = [f'{method_dir}/{method_name}_K{K}' for K in K_values]

### Main system information
system_name, H_S, V_S = settings.system()
Ω, C2, p = omega0v0.system_transitions(λ, β, H_S, V_S)
print(f'Bohr frequency identification is completed')
if np.any(Ω == 0.):
    raise ValueError(f'The present method is not applicable when Ω = 0.')
os.makedirs(f'{dir_name}/{method_dir}/{system_name}', exist_ok=True)

### Hyperparameters for exact-solution routines
tols, Kj, GMT, ωmax = settings.hparams_exact()

if Kj == None:
    Kj, c, μ = expfit_GMT.cμ_ESPRIT_fit_relerr(f'{dir_name}/data_time.csv')
else:
    c, μ = expfit_GMT.cμ_ESPRIT_fit(f'{dir_name}/data_time.csv', Kj)
print(f'J(ω) fitting is completed')

ωj = ω[ω > 0]; expfit_GMT.plot_J(ωj, data.J(ωj), c, μ, f'{dir_name}/{method_dir}/{system_name}/Jfit_Kj{Kj}')

if GMT:
    η = exact.η_cμ(c, μ); J = None; ωmax = None
else:
    η = data.get_η()

### J & ωmax (these should be given if η == None)
if η == None:
    J = getattr(data, 'J_')  # we call J a lot for 'η_integral' and hence we should import data.J_ (not data.J, which may take more time to call)
else:
    J = None; ωmax = None

# =============================================================================
# Testing BCF
# =============================================================================

ω0_array = np.zeros(len(Ω)); v0_array = np.zeros_like(ω0_array)

### Create error file
f = open(f'{dir_name}/{method_dir}/{system_name}/error.csv', mode="w")
f.write('***  K  :  L_time  :  L_frequency  ' + '\n')
Ks = np.zeros(len(results_name)); δLt = np.zeros_like(Ks); δLω = np.zeros_like(Ks)
for i in range(len(results_name)):
    Lfit_filename = f'{dir_name}/{results_name[i]}'
    K, d, z, LT_correction = expfit.read_correlation_function(Lfit_filename)
    Ks[i] = K
    δLt[i] = 1/len(t) * np.sum(np.abs(expfit.model_Lt(t, d, z) - Lt))
    δLω[i] = 1/len(ω) * np.sum(np.abs(expfit.model_Lω(ω, d, z, LT_correction) - Lω))
    f.write('{:3.0f}'.format(Ks[i])); f.write('{:15.3e}'.format(δLt[i])); f.write('{:15.3e}'.format(δLω[i])); f.write('\n')
f.write('\n')
if compare_corr:
    f.write('***  K  :  q2  :  p2  :  Cqq_frequency  :  Cpp_frequency  :' + '\n')
else:
    f.write('***  K  :  q2  :  p2  :' + '\n')

δq2 = np.zeros((len(Ω), len(results_name))); δp2 = np.zeros_like(δq2); δCqqω = np.zeros_like(δq2); δCppω = np.zeros_like(δq2) # relative error
for n in range(len(Ω)):

    ### Find ω0 and v0
    print(f'### Bohr-frequency = {Ω[n]:.2e} ({n+1} / {len(Ω)})')
    ω0, v0, rel_error = omega0v0.ω0v0(Ω[n], C2[n], λ, β, η, J, ωmax)
    print(f'ω0 = {ω0:.2e}; v0 = {v0:.2e}')
    print(f'|v0^2 <q^2> -  C2|/C2 = {rel_error:.2e}')
    print()

    dir_name_ω0v0 = f'{dir_name}/{method_dir}/{system_name}/ω0 = {ω0:.3e}; v0 = {v0:.3e}'
    os.makedirs(dir_name_ω0v0, exist_ok=True)
    f.write(f'Ω = {Ω[n]}; C2 = {C2[n]} => ω0 = {ω0}; v0 = {v0}' + '\n')
    ω0_array[n] = ω0; v0_array[n] = v0

    ### Exact solutions
    # parameters
    M = 1/(ω0*v0**2)
    params = np.array([M, ω0, β])

    # q2p2
    print('q2p2 evaluation')
    exact.q2p2_tols(params, tols, dir_name_ω0v0 + '/exact', η = η, J = J, ωmax = ωmax)

    exact_q2p2 = np.loadtxt(f'{dir_name_ω0v0}/exact_q2p2.csv', skiprows=1)
    if exact_q2p2.ndim == 1:
        _, q2_eq, p2_eq = exact_q2p2[:]
    else:
        _, q2_eq, p2_eq = exact_q2p2[-1,:]

    # CqqCpp
    if compare_corr:
        if ω_corr == None:
            ω_corr_fourier = np.linspace(-5*ω0, 5*ω0, 500)
        else:
            ω_corr_fourier = ω_corr
        exact.CqqCpp_η(params, ω_corr_fourier, dir_name_ω0v0 + '/exact', η = η, J = J, ωmax = ωmax)
        exact_corr_fourier = np.loadtxt(f'{dir_name_ω0v0}/exact_corr_fourier.csv', skiprows=1)
    else:
        ω_corr_fourier = ω_corr

    ### HEOM % Error assessment
    for i in range(len(results_name)):

        dir_name_heom = f'{dir_name_ω0v0}/{results_name[i]}'
        Lfit_filename = f'{dir_name}/{results_name[i]}'
        os.makedirs(dir_name_heom, exist_ok=True)

        solver = HEOMSolver(ω0=ω0, v0=v0, λ=λ, β=β,
                            c=c, μ=μ, tss=None, tcorr=None, dt_corr=dt_corr,
                            ω=ω_corr_fourier, dir_name_heom=dir_name_heom, Lfit_filename=Lfit_filename)

        solver.operators()
        solver.write_settings()

        if compare_corr:
            solver.equilibrium_correlation_function()
            heom_corr_fourier = np.loadtxt(f'{solver.title}_corr_fourier.csv', skiprows=1)

            Cωmin = np.min( np.hstack([exact_corr_fourier[:,1], exact_corr_fourier[:,2]])   )
            Cωmax = np.max( np.hstack([exact_corr_fourier[:,1], exact_corr_fourier[:,2]])   )
            # Cωmax = np.max( exact[:,2] )

            ΔCω = (Cωmax-Cωmin) / 10

            # overall behavior
            plt.figure(figsize=(7,5))
            plt.plot(exact_corr_fourier[:,0], exact_corr_fourier[:,1], label=r'$\mathcal{F}[C_{qq}](\omega)$', color='red', lw='1.5')
            plt.plot(heom_corr_fourier[:,0], heom_corr_fourier[:,1], label=r'$\mathcal{F}[C_{qq}^{\rm HEOM}](\omega)$', linestyle = 'None', color='red', marker='o', markersize=4)
            plt.plot(exact_corr_fourier[:,0], exact_corr_fourier[:,2], label=r'$\mathcal{F}[C_{pp}](\omega)$', color='blue', lw='1.5')
            plt.plot(heom_corr_fourier[:,0], heom_corr_fourier[:,2], label=r'$\mathcal{F}[C_{pp}^{\rm HEOM}](\omega)$', linestyle = 'None', color='blue', marker='o', markersize=4)
            plt.ylim(Cωmin-ΔCω,Cωmax+ΔCω)
            plt.xticks(fontsize=17)
            plt.yticks(fontsize=17)
            plt.xlabel(r'$\omega$', fontsize=18)
            plt.ylabel('Correlation function', fontsize=18)
            plt.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{solver.title}_corr_fourier.png', format='png')
            plt.close()  # Close the first figure
        else:
            solver.time_evolution()

        # error
        heom_obs = np.loadtxt(f'{solver.title}_obs.csv', skiprows = 1)
        q2 = heom_obs[-1,3]
        p2 = heom_obs[-1,4]
        δq2[n,i] = np.abs(q2-q2_eq)/q2_eq
        δp2[n,i] = np.abs(p2-p2_eq)/p2_eq

        if compare_corr:
            δCqqω[n,i] = simpson(np.abs(exact_corr_fourier[:,1] - heom_corr_fourier[:,1]), x=ω_corr_fourier) / simpson(np.abs(exact_corr_fourier[:,1]), x=ω_corr_fourier)
            δCppω[n,i] = simpson(np.abs(exact_corr_fourier[:,2] - heom_corr_fourier[:,2]), x=ω_corr_fourier) / simpson(np.abs(exact_corr_fourier[:,2]), x=ω_corr_fourier)
            f.write('{:3.0f}'.format(Ks[i])); f.write('{:15.3e}'.format(δq2[n,i])); f.write('{:15.3e}'.format(δp2[n,i])); f.write('{:15.3e}'.format(δCqqω[n,i])); f.write('{:15.3e}'.format(δCppω[n,i])); f.write('\n')
        else:
            f.write('{:3.0f}'.format(Ks[i])); f.write('{:15.3e}'.format(δq2[n,i])); f.write('{:15.3e}'.format(δp2[n,i])); f.write('\n')
    f.write('\n')
    print(); print()

### Evaluate the average error
f.write(f'Average:' + '\n')
f.write('p = [' + ', '.join(f'{w:.2e}' for w in p) + ']\n')
δq2_mean = np.einsum("i,ij->j", p, δq2) / np.sum(p); δp2_mean = np.einsum("i,ij->j", p, δp2) / np.sum(p)
if compare_corr:
    δCqqω_mean = np.einsum("i,ij->j", p, δCqqω) / np.sum(p); δCppω_mean = np.einsum("i,ij->j", p, δCppω) / np.sum(p)
    for i in range(len(results_name)):
            f.write('{:3.0f}'.format(Ks[i])); f.write('{:15.3e}'.format(δq2_mean[i])); f.write('{:15.3e}'.format(δp2_mean[i])); f.write('{:15.3e}'.format(δCqqω_mean[i])); f.write('{:15.3e}'.format(δCppω_mean[i])  + '\n')
else:
    for i in range(len(results_name)):
            f.write('{:3.0f}'.format(Ks[i])); f.write('{:15.3e}'.format(δq2_mean[i])); f.write('{:15.3e}'.format(δp2_mean[i]) + '\n')
print(); print()
f.close()

# =============================================================================
# Plot the errors
# =============================================================================

# δLt
plt.figure(figsize=(7,5))
plt.plot(Ks,δLt,linestyle='None',marker='o')
plt.xticks(Ks,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$K$", fontsize=15)
plt.ylabel(r"$\delta L$", fontsize=15)
plt.grid(axis = 'both', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_Lt.png', format='png')
plt.close()

# δLω
plt.figure(figsize=(7,5))
plt.plot(Ks,δLω,linestyle='None',marker='o')
plt.xticks(Ks,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$K$", fontsize=15)
plt.ylabel(r"$\delta \mathcal{F}[L]$", fontsize=15)
plt.grid(axis = 'both', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_Lω.png', format='png')
plt.close()

# δq2
plt.figure(figsize=(7,5))
for n in range(len(Ω)):
    plt.plot(Ks,δq2[n,:],linestyle='None',marker='o',label=r'$\omega_0=$'+f'{ω0_array[n]:.3e}')
plt.xticks(Ks,fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-0.01,0.5)
plt.xlabel(r"$K$", fontsize=15)
plt.ylabel(r"$\delta \langle q^2 \rangle$", fontsize=15)
plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_q2.png', format='png')
plt.close()

# δp2
plt.figure(figsize=(7,5))
for n in range(len(Ω)):
    plt.plot(Ks,δp2[n,:],linestyle='None',marker='o',label=r'$\omega_0=$'+f'{ω0_array[n]:.3e}')
plt.xticks(Ks,fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-0.01,0.5)
plt.xlabel(r"$K$", fontsize=15)
plt.ylabel(r"$\delta \langle p^2 \rangle$", fontsize=15)
plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_p2.png', format='png')
plt.close()

# δq2, δp2 (mean)
plt.figure(figsize=(7,5))
δq2_mean = np.einsum("i,ij->j", p, δq2) / np.sum(p)
δp2_mean = np.einsum("i,ij->j", p, δp2) / np.sum(p)
plt.plot(Ks,δq2_mean,linestyle='None',marker='s',markersize=7,label=r'$o=q$')
plt.plot(Ks,δp2_mean,linestyle='None',marker='o',markersize=6,label=r'$o=p$')
plt.xticks(Ks,fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-0.01,0.5)
plt.xlabel(r"$K$", fontsize=15)
plt.ylabel(r"$\delta \langle o^2 \rangle$", fontsize=15)
plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_q2p2_mean.png', format='png')
plt.close()

if compare_corr:
    # δCqqω
    plt.figure(figsize=(7,5))
    for n in range(len(Ω)):
        plt.plot(Ks,δCqqω[n,:],linestyle='None',marker='o',label=r'$\omega_0=$'+f'{ω0_array[n]:.3e}')
    plt.xticks(Ks,fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.01,0.5)
    plt.xlabel(r"$K$", fontsize=15)
    plt.ylabel(r"$\delta C_{qq}$", fontsize=15)
    plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_Cqqω.png', format='png')
    plt.close()

    # δCppω
    plt.figure(figsize=(7,5))
    for n in range(len(Ω)):
        plt.plot(Ks,δCppω[n,:],linestyle='None',marker='o',label=r'$\omega_0=$'+f'{ω0_array[n]:.3e}')
    plt.xticks(Ks,fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.01,0.5)
    plt.xlabel(r"$K$", fontsize=15)
    plt.ylabel(r"$\delta C_{pp}$", fontsize=15)
    plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_Cppω.png', format='png')
    plt.close()

    # δCqqω, δCppω (mean)
    plt.figure(figsize=(7,5))
    δCqqω_mean = np.einsum("i,ij->j", p, δCqqω) / np.sum(p)
    δCppω_mean = np.einsum("i,ij->j", p, δCppω) / np.sum(p)
    plt.plot(Ks,δCqqω_mean,linestyle='None',marker='s',markersize=7,label=r'$o=q$')
    plt.plot(Ks,δCppω_mean,linestyle='None',marker='o',markersize=6,label=r'$o=p$')
    plt.xticks(Ks,fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.01,0.5)
    plt.xlabel(r"$K$", fontsize=15)
    plt.ylabel(r"$\delta C_{oo}$", fontsize=15)
    plt.grid(axis = 'both', linestyle = '--', linewidth = 1.)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{dir_name}/{method_dir}/{system_name}/error_CqqωCppω_mean.png', format='png')
    plt.close()
