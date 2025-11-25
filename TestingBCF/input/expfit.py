import numpy as np
import matplotlib.pyplot as plt
import os

from input.ESPRIT import ESPRIT

from input.AAA import AAA, AAA_prz

# ----------------------------------------
#   Fitting by a sum of exponential functions
# ----------------------------------------

"""
Fitting L(t≧t0) (data) using the ESPRIT algorithm
"""
def ESPRIT_fit(Lt_filename, M, Lt_data = np.array([]), Lω_filename = None, dir = None, tol = 1e-8):

    if len(Lt_data) == 0:
        Lt_data = np.loadtxt(Lt_filename, dtype=complex)

    # data
    t = np.real(Lt_data[:,0])
    t0 = t[0]
    dt = t[1]-t[0]
    Lt = Lt_data[:,1]

    # optimization
    c, z, _ = ESPRIT(dt, Lt, M, t[0])
    d = c * np.exp(z * t[0])

    # regularization
    d, z, K = regularization(d, z, tol)
    η = 0.

    # output
    if dir == None:
        filename = f'ESPRIT_K{K}'
    else:
        os.makedirs(dir, exist_ok=True)
        filename = dir + '/' + f'ESPRIT_K{K}'

    savefile(Lt_filename, Lω_filename, d, z, η, K, filename)

    return d, z, η

"""
Fitting \tilde{L}(ω) (data) using the AAA algorithm
"""
def AAA_fit(Lω_filename, ε_rel, Lω_data = np.array([]), Lt_filename = None, dir = None, tol = 1e-8):

    if len(Lω_data) == 0:
        Lω_data = np.loadtxt(Lω_filename)

    # data
    ω = Lω_data[:,0]
    dω = ω[1]-ω[0]
    Lω = Lω_data[:,1]

    # optimization
    idx_smpl, w, fmod, _, err_rel = AAA(ω, Lω, ε_rel = ε_rel)

    ωsmpl = ω[idx_smpl]
    Lωsmpl = Lω[idx_smpl]

    pol, res, _ = AAA_prz(ωsmpl, w, Lωsmpl, fmod)

    d = np.array([])
    z = np.array([])
    # η = 0.5 * np.sum(w*Lωsmpl) / np.sum(w)
    η = 0.

    for l in range(len(pol)):
        if pol[l].imag < - tol:
            d = np.append(d, -1j*res[l])
            z = np.append(z, 1j*pol[l])

    # regularization
    d, z, K = regularization(d, z, tol)

    # # output
    # if dir == None:
    #     filename = f'AAA_K{K}_err{err_rel:.2e}'
    # else:
    #     filename = dir + '/' + f'AAA_K{K}_err{err_rel:.2e}'

    # output
    if dir == None:
        filename_root = f'AAA_K{K}'
    else:
        os.makedirs(dir, exist_ok=True)
        filename_root = dir + '/' + f'AAA_K{K}'

    filename = [filename_root + f'_err{err_rel:.2e}', filename_root]

    savefile(Lt_filename, Lω_filename, d, z, η, K, filename)

    return d, z, η

"""
The AAA algorithm is applied while gradually decreasing the target relative tolerance until the designated lower bound of K, denoted by Klb, is reached.
"""
def AAA_K_fit(Lω_filename, Klb, ε_rel_start = np.inf, Lω_data = np.array([]), Lt_filename = None, dir = None, tol = 1e-8):

    if len(Lω_data) == 0:
        Lω_data = np.loadtxt(Lω_filename)

    # data
    ω = Lω_data[:,0]
    dω = ω[1]-ω[0]
    Lω = Lω_data[:,1]

    ε_rel = ε_rel_start
    while True:

        # optimization
        idx_smpl, w, fmod, _, ε_rel = AAA(ω, Lω, ε_rel = ε_rel, print_err = False)
        ε_rel -= 1e-5*ε_rel # gradually decreasing the target relative tolerance
        if len(w) == 1:
            continue

        ωsmpl = ω[idx_smpl]
        Lωsmpl = Lω[idx_smpl]

        pol, res, _ = AAA_prz(ωsmpl, w, Lωsmpl, fmod)

        d = np.array([])
        z = np.array([])
        # η = 0.5 * np.sum(w*Lωsmpl) / np.sum(w)
        η = 0.

        for l in range(len(pol)):
            if pol[l].imag < - tol:
                d = np.append(d, -1j*res[l])
                z = np.append(z, 1j*pol[l])

        # regularization
        d, z, K = regularization(d, z, tol)

        if K >= Klb:
            break

        if ε_rel < 1e-14:
            # print(f'minimum (1e-14) is reached')
            break

        # d_new = np.array([])
        # z_new = np.array([])
        # # η = 0.5 * np.sum(w*Lωsmpl) / np.sum(w)
        # η = 0.
        #
        # for l in range(len(pol)):
        #     if pol[l].imag < - tol:
        #         d_new = np.append(d_new, -1j*res[l])
        #         z_new = np.append(z_new, 1j*pol[l])
        #
        # # regularization
        # d_new, z_new, K_new = regularization(d_new, z_new, tol)
        #
        # print(K_new)
        # if K_new > Klb:
        #     break
        #
        # d = d_new; z = z_new; K = K_new

    # output
    if dir == None:
        filename_root = f'AAA_K{K}'
    else:
        os.makedirs(dir, exist_ok=True)
        filename_root = dir + '/' + f'AAA_K{K}'

    filename = [filename_root + f'_err{ε_rel:.2e}', filename_root]

    savefile(Lt_filename, Lω_filename, d, z, η, K, filename)

    return d, z, η

# ----------------------------------------
#   Model functions
# ----------------------------------------

def model_Lω(ω, d, z, η):

    def Lω_value(ω):
        return np.real( np.sum(2 * d / (z - 1j*ω)) ) + 2*η

    if isinstance(ω, (list, np.ndarray)):
        return np.vectorize(Lω_value)(ω)
    else:
        return Lω_value(ω)

def model_Lt(t, d, z):

    def Lt_value(t):
        return np.sum(d * np.exp(- z * t))

    if isinstance(t, (list, np.ndarray)):
        return np.vectorize(Lt_value)(t)
    else:
        return Lt_value(t)





# ----------------------------------------
#   Output / Input files
# ----------------------------------------

def savefile(Lt_filename, Lω_filename, d, z, η, K, filename):

    if isinstance(filename, list):
        if not len(filename) == 2:
            raise ValueError('filename must be a string or a list of two strings')
        filename_plot = filename[0]; filename_result = filename[1]
    else:
        filename_plot = filename; filename_result = filename

    if Lt_filename is not None:
        plot_Lt_data(Lt_filename, d, z, filename_plot + '_time', ylog = False)
    if Lω_filename is not None:
        plot_Lω_data(Lω_filename, d, z, η, filename_plot + '_frequency')

    write_exp_coeff(d, z, η, K, filename_result)

def write_exp_coeff(d, z, η, K, filename):

    i_taken = np.array([], dtype=int)

    f = open(filename + '.csv', mode="w")
    f.write(str(K) + '\n')
    for i in range(len(d)):
        if i in i_taken:
            continue

        if z[i].imag == 0:
            f.write('{:35.8e}'.format(d[i]) + '{:35.8e}'.format(z[i].real) + '\n')
            continue

        ibar = np.where(z == np.conjugate(z[i]))[0]
        if len(ibar) == 1:   # A pair is found
            i_taken = np.append(i_taken, ibar[0])
            f.write('{:35.8e}'.format(d[i]) + '{:35.8e}'.format(z[i]) + '\n')
            f.write('{:35.8e}'.format(d[ibar[0]]) + '{:35.8e}'.format(z[ibar[0]]) + '\n')
        elif len(ibar) == 0:  # No pair
            f.write('{:35.8e}'.format(d[i]) + '{:35.8e}'.format(z[i]) + '\n')
            f.write('{:35.8e}'.format(0.) + '{:35.8e}'.format(np.conjugate(z[i])) + '\n')
        elif len(ibar) > 1:
            raise ValueError("Error: duplication in the array is found.")

    f.write('LT correction' + '\n')
    f.write('{:35.8e}'.format(η))

def read_correlation_function(dir_name):

    with open(dir_name + ".csv", "r") as file:
        K = int(file.readline())
        d = np.array([], dtype = complex)
        z = np.array([], dtype = complex)
        for i in range(K):
            di, zi = file.readline().split()
            d = np.append(d, complex(di))
            z = np.append(z, complex(zi))

        _ = file.readline()
        η = float(file.readline())

    return K, d, z, η





# ----------------------------------------
#   Plots
# ----------------------------------------

def plot_Lt_data(Lt_filename, d, z, figure_name, ylog = False, save = True, show = False):

    data = np.loadtxt(Lt_filename, dtype=complex)

    # data
    t = np.real(data[:,0])
    Lt = data[:,1]

    plot_Lt(t, Lt, model_Lt(t, d, z), figure_name, ylog = ylog, save = save, show = show)

def plot_Lt(t, Lt, Lt_fit, figure_name, ylog = False, save = True, show = False):

    Ltmin = np.min( np.append(Lt.real, Lt.imag)   )
    Ltmax = np.max( np.append(Lt.real, Lt.imag)   )
    ΔLt = (Ltmax-Ltmin) / 10

    plt.figure(figsize=(8, 6))
    plt.plot(t,Lt.real,color='black',lw=3,label='Data.real')
    plt.plot(t,Lt.imag,color='black',linestyle='dashed',lw=3,label='Data.imag')
    plt.plot(t,Lt_fit.real,color='red',lw=1.5,label='Fit.real')
    plt.plot(t,Lt_fit.imag,color='red',linestyle='dashed',lw=1.5,label='Fit.imag')
    if not ylog:
        plt.ylim(Ltmin-ΔLt,Ltmax+ΔLt)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$t$", fontsize=15)
    plt.ylabel(r"$L(t)$", fontsize=15)
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(figure_name + '.png', format='png')
    if show:
        plt.show()
    plt.close()

def plot_Lω_data(Lω_filename, d, z, η, figure_name, ylog = False, save = True, show = False):

    data = np.loadtxt(Lω_filename)

    # data
    ω = data[:,0]
    Lω = data[:,1]

    plot_Lω(ω, Lω, model_Lω(ω, d, z, η), figure_name, ylog = ylog, save = save, show = show)

def plot_Lω(ω, Lω, Lω_fit, figure_name, Lωmax_default = 8., ylog = False, save = True, show = False):

    Lωmin = np.min(Lω)
    # Lωmax = np.min([np.max(Lω),Lωmax_default])
    Lωmax = np.max(Lω)
    ΔLω = (Lωmax-Lωmin) / 10

    plt.figure(figsize=(8, 6))
    plt.plot(ω,Lω,color='black',lw=3,label='Data')
    plt.plot(ω,Lω_fit,color='red',lw=1.5,label='Fit')
    if not ylog:
        plt.ylim(Lωmin-ΔLω,Lωmax+ΔLω)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\omega$", fontsize=15)
    plt.ylabel(r"$\mathcal{F}[L](\omega)$", fontsize=15)
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(figure_name + '.png', format='png')
    if show:
        plt.show()
    plt.close()



# ----------------------------------------
#   Regularizing the parameters
# ----------------------------------------

def regularization(d, z, tol):
    if check_duplication(z, tol):
        raise ValueError('Elements of z are duplicated and the optimization failed')
    d, z = remove_zero(d, z, tol)
    z = zero_small_imaginary_parts(z, tol)
    z, K = conjugate_pairing(z, tol)
    return d, z, K

def check_duplication(array, tol):
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            if np.abs(array[i] - array[j]) < tol:
                return True
    return False

def conjugate_pairing(array, tol):

    i_taken = np.array([], dtype=int)

    K = 0
    for i in range(len(array)):
        if i in i_taken:
            continue

        if array[i].imag == 0:
            K += 1
            continue

        ibar = np.where(np.abs(array - np.conjugate(array[i])) < tol)[0]
        if len(ibar) == 1:   # A pair is found
            i_taken = np.append(i_taken, ibar[0])
            array[ibar[0]] = np.conjugate(array[i])
            K += 2
        elif len(ibar) == 0:  # No pair
            K += 2
        elif len(ibar) > 1:
            raise ValueError("Error: duplication in the array is found.")

    return array, K

def zero_small_imaginary_parts(array, tol):

    return np.where(np.abs(array.imag) < tol,
                              array.real,
                              array)

def remove_zero(d, z, tol):

    inds_zero = np.array([], dtype=int)
    for i in range(len(d)):
        if np.abs(d[i]) < tol:
        # if np.abs(d[i]) == 0.:
            inds_zero = np.append(inds_zero, i)

    if len(inds_zero) == 0:
        return d, z
    else:
        return np.delete(d, inds_zero), np.delete(z, inds_zero)
