import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from scipy.special.cython_special cimport psi, gamma, loggamma
from scipy.stats import gamma as gamma_dist
import time, os
import ctypes

from libc.stdio cimport printf
from pathlib import Path

## Note: As of the end of 2018, MSVC is still stuck with OpenMP 2.0 (released 2002), which does not support
## parallel for loops with unsigend iterators. If you are using a different compiler, this part can be safely removed
## See also: https://github.com/cython/cython/issues/3136
IF UNAME_SYSNAME == "Windows":
    obj_ind_type = ctypes.c_longlong
    ctypedef long long ind_type
    ctypedef double long_double_type
    obj_long_double_type = ctypes.c_double
    LD_HUGE_VAL = HUGE_VAL
ELSE:
    obj_ind_type = ctypes.c_size_t
    ctypedef size_t ind_type
    ctypedef long double long_double_type
    obj_long_double_type = ctypes.c_longdouble
    LD_HUGE_VAL = HUGE_VALL

### Helper functions
####################
def cast_real_t(n):
    return <real_t> n

def cast_int(n):
    return <int> n

def cast_ind_type(n):
    return <ind_type> n

def save_parameters(verbose, save_folder, file_names, obj_list):
    if verbose:
        print("Saving final parameters to .csv files...")

    for i in range(len(file_names)):
        np.savetxt(os.path.join(save_folder, file_names[i]), obj_list[i], fmt="%.10f", delimiter=',')

def assess_convergence(int i, check_every, stop_crit, last_crit, last_rmse, stop_thr, approx_rte,
                       np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Theta_prev,
                       np.ndarray[real_t, ndim=2] Beta, ind_type nY,
                       np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u,
                       np.ndarray[ind_type, ndim=1] ix_i, ind_type nYv,
                       np.ndarray[real_t, ndim=1] Yval, np.ndarray[ind_type, ndim=1] ix_u_val,
                       np.ndarray[ind_type, ndim=1] ix_i_val,
                       np.ndarray[real_t, ndim=1] errs, ind_type k, int nthreads, int verbose, int full_llk,
                       has_valset, int Y_max):
    is_converge = False
    output_llk = last_crit
    output_rmse = last_rmse

    llk_plus_rmse(&Theta[0, 0], &Beta[0, 0], &Y[0],
                  &ix_u[0], &ix_i[0], nY, k,
                  &errs[0], nthreads, verbose, full_llk,
                  Y_max)
    # errs[0] -= Theta.sum(axis=0).dot(Beta.sum(axis=0))
    errs[1] = np.sqrt(errs[1] / nY)

    if verbose:
        print_llk_iter(<int> (i + 1), <long long> errs[0], <double> errs[1], has_valset)

    if (i + 1) % check_every == 0:
        if (1. - errs[0] / last_crit) <= stop_thr:
            if stop_crit != 'maxiter':
                # if approx_rte:
                #     approx_rte = False
                # else:
                is_converge = True
            if errs[0] / last_crit <= 1:
                output_llk = errs[0]
                output_rmse = errs[1]
        else:
            output_llk = errs[0]
            output_rmse = errs[1]
    return is_converge, output_llk, output_rmse, approx_rte

def eval_after_term(stop_crit, int verbose, int nthreads, int full_llk, ind_type k, ind_type nY, ind_type nYv,
                    has_valset,
                    np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Beta,
                    np.ndarray[real_t, ndim=1] errs,
                    np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u,
                    np.ndarray[ind_type, ndim=1] ix_i,
                    np.ndarray[real_t, ndim=1] Yval, np.ndarray[ind_type, ndim=1] ix_u_val,
                    np.ndarray[ind_type, ndim=1] ix_i_val, int Y_max):
    if (stop_crit == 'diff-norm') or (stop_crit == 'maxiter'):
        if verbose > 0:
            if has_valset:
                llk_plus_rmse(&Theta[0, 0], &Beta[0, 0], &Yval[0],
                              &ix_u_val[0], &ix_i_val[0], nYv, k,
                              &errs[0], nthreads, verbose, full_llk, Y_max)
                # errs[0] -= Theta[ix_u_val].sum(axis=0).dot(Beta[ix_i_val].sum(axis=0))
                errs[1] = np.sqrt(errs[1] / nYv)
            else:
                llk_plus_rmse(&Theta[0, 0], &Beta[0, 0], &Y[0],
                              &ix_u[0], &ix_i[0], nY, k,
                              &errs[0], nthreads, verbose, full_llk, Y_max)
                # errs[0] -= Theta.sum(axis=0).dot(Beta.sum(axis=0))
                errs[1] = np.sqrt(errs[1] / nY)
            return errs[0]

### Random initializer for parameters
#####################################
def initialize_parameters_k_t(Theta, Beta, random_seed,
                              a, a_prime, b_prime, c, c_prime, d_prime, cut_extreme_initial, save_folder):
    nU = Theta.shape[0]
    nI = Beta.shape[0]
    k = Theta.shape[1]

    # cdef real_t cut_extreme_initial = 1e-1

    rng = np.random.default_rng(seed=random_seed if random_seed > 0 else None)
    cdef np.ndarray[double, ndim=2] ksi = gamma_dist.ppf(rng.uniform(cut_extreme_initial, 1 - cut_extreme_initial, nU),
                                                         a=a_prime,
                                                         scale=b_prime / a_prime).reshape(nU, 1)

    np.savetxt(os.path.join(save_folder, 'initial_value/ksi.csv'), ksi, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/ksi.csv', ksi, delimiter=',')

    for i in range(nU):
        Theta[i, :] = gamma_dist.ppf(rng.uniform(cut_extreme_initial, 1 - cut_extreme_initial, k), a=a,
                                     scale=1 / ksi[i])
    Theta[:, :] = Theta.astype(c_real_t)

    cdef np.ndarray[double, ndim=2] eta = gamma_dist.ppf(rng.uniform(cut_extreme_initial, 1 - cut_extreme_initial, nI),
                                                         a=c_prime,
                                                         scale=d_prime / c_prime).reshape(nI, 1)

    np.savetxt(os.path.join(save_folder, 'initial_value/eta.csv'), eta, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/eta.csv', eta, delimiter=',')

    for i in range(nI):
        Beta[i, :] = gamma_dist.ppf(rng.uniform(cut_extreme_initial, 1 - cut_extreme_initial, k), a=c, scale=1 / eta[i])
    Beta[:, :] = Beta.astype(c_real_t)

    k_rte = a_prime / b_prime + Theta.sum(axis=1, keepdims=True).astype(c_real_t)
    t_rte = c_prime / d_prime + Beta.sum(axis=1, keepdims=True).astype(c_real_t)

    # np.nan_to_num(Theta, copy=False)
    # np.nan_to_num(Beta, copy=False)
    # np.nan_to_num(k_rte, copy=False)
    # np.nan_to_num(t_rte, copy=False)

    return Theta, Beta, k_rte, t_rte

### Main function
#################
def fit_hpf(real_t a, real_t a_prime, real_t b_prime,
            real_t c, real_t c_prime, real_t d_prime,
            # input_df.Count.values = Y
            np.ndarray[real_t, ndim=1] Y,
            # input_df.UserId.values = ix_u
            np.ndarray[ind_type, ndim=1] ix_u,
            # input_df.ItemId.values = ix_i
            np.ndarray[ind_type, ndim=1] ix_i,
            np.ndarray[real_t, ndim=2] Theta,
            np.ndarray[real_t, ndim=2] Beta,
            int maxiter, str stop_crit, int check_every, real_t stop_thr,
            users_per_batch, items_per_batch, step_size, int sum_exp_trick,
            # _st_ix_user = st_ix_u
            np.ndarray[ind_type, ndim=1] st_ix_u,
            str save_folder, int random_seed, int verbose,
            # ncores = nthreads, allow_inconsistent_math = par_sh, use_valset = has_valset
            int nthreads, int par_sh, int has_valset,
            # val_set.Count.values = Yval
            np.ndarray[real_t, ndim=1] Yval,
            # val_set.UserId.values = ix_u_val
            np.ndarray[ind_type, ndim=1] ix_u_val,
            # val_set.ItemId.values = ix_i_val
            np.ndarray[ind_type, ndim=1] ix_i_val,
            int full_llk, int keep_all_objs, int alloc_full_phi,
            int approx_rte, real_t cut_extreme_initial):
    ## useful information
    cdef ind_type nU = Theta.shape[0]
    cdef ind_type nI = Beta.shape[0]
    cdef ind_type nY = Y.shape[0]
    cdef ind_type k = Theta.shape[1]
    cdef real_t Y_max = Y.max()
    cdef ind_type nYv
    if has_valset > 0:
        nYv = Yval.shape[0]

    cdef real_t k_shp = a_prime + k * a
    cdef real_t t_shp = c_prime + k * c

    cdef np.ndarray[real_t, ndim=2] phi
    cdef np.ndarray[real_t, ndim=2] Latent
    # cdef np.ndarray[real_t, ndim=2] exp_T_dot_B
    if ((users_per_batch == 0) and (items_per_batch == 0)) or alloc_full_phi:
        if verbose > 0:
            print("Allocating Phi matrix...")
        phi = np.empty((nY, k), dtype=c_real_t)
        Latent = np.empty((nY, k), dtype=c_real_t)
        # exp_T_dot_B = np.empty((nY, nI), dtype=c_real_t)

    full_updates = True

    rng = np.random.default_rng(seed=random_seed if random_seed > 0 else None)

    cdef real_t add_k_rte = a_prime / b_prime
    cdef real_t add_t_rte = c_prime / d_prime
    cdef np.ndarray[real_t, ndim=1] errs = np.zeros(2, dtype=c_real_t)

    cdef real_t last_crit = - LD_HUGE_VAL
    cdef real_t last_rmse
    cdef np.ndarray[real_t, ndim=2] Theta_prev
    if stop_crit == 'diff-norm':
        Theta_prev = Theta.copy()
    else:
        Theta_prev = np.empty((0, 0), dtype=c_real_t)

    if verbose > 0:
        print("Initializing parameters...")

    cdef np.ndarray[real_t, ndim=2] Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, Gamma_rte_new, Lambda_rte_new
    # cdef np.ndarray[bool, ndim=2] Gamma_rte_ind, Lambda_rte_ind
    cdef np.ndarray[real_t, ndim=2] k_rte, t_rte
    cdef np.ndarray[real_t, ndim=1] exp_T_dot_B, T_dot_B

    Path(os.path.join(save_folder, 'initial_value')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(save_folder, 'updating_parameter/parameter')).mkdir(parents=True, exist_ok=True)

    Theta, Beta, k_rte, t_rte = initialize_parameters_k_t(Theta, Beta, random_seed, a,
                                                          a_prime, b_prime, c, c_prime, d_prime, cut_extreme_initial,
                                                          save_folder)


    np.savetxt(os.path.join(save_folder, 'initial_value/Theta.csv'), Theta, fmt="%.10f", delimiter=',')
    np.savetxt(os.path.join(save_folder, 'initial_value/Beta.csv'), Beta, fmt="%.10f", delimiter=',')
    np.savetxt(os.path.join(save_folder, 'initial_value/k_rte.csv'), k_rte, fmt="%.10f", delimiter=',')
    np.savetxt(os.path.join(save_folder, 'initial_value/t_rte.csv'), t_rte, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/Theta.csv', Theta, delimiter=',')
    # np.savetxt('initial_value/Beta.csv', Beta, delimiter=',')
    # np.savetxt('initial_value/k_rte.csv', k_rte, delimiter=',')
    # np.savetxt('initial_value/t_rte.csv', t_rte, delimiter=',')
    # print('initialize_parameters_k_t: OK')

    initialize_parameters_Latent_par(&Theta[0, 0], &Beta[0, 0], k,
                                     &Y[0], &Latent[0, 0],
                                     &ix_u[0], &ix_i[0], nY, nthreads)

    # np.nan_to_num(Latent, copy=False)
    np.savetxt(os.path.join(save_folder, 'initial_value/Latent.csv'), Latent, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/Latent.csv', Latent, delimiter=',')
    # print('initialize_parameters_Latent: OK')

    Gamma_shp = np.empty((nU, k), dtype=c_real_t)
    Lambda_shp = np.empty((nI, k), dtype=c_real_t)

    Gamma_shp[:, :] = a
    Lambda_shp[:, :] = c

    initialize_parameters_G_L_sh_par(&Gamma_shp[0, 0], &Lambda_shp[0, 0],
                                     &Latent[0, 0], k,
                                     &ix_u[0], &ix_i[0], nY, nthreads,
                                     &Y[0])

    # np.nan_to_num(Gamma_shp, copy=False)
    # np.nan_to_num(Lambda_shp, copy=False)
    np.savetxt(os.path.join(save_folder, 'initial_value/Gamma_shp.csv'), Gamma_shp, fmt="%.10f", delimiter=',')
    np.savetxt(os.path.join(save_folder, 'initial_value/Lambda_shp.csv'), Lambda_shp, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/Gamma_shp.csv', Gamma_shp, delimiter=',')
    # np.savetxt('initial_value/Lambda_shp.csv', Lambda_shp, delimiter=',')
    # print('initialize_parameters_G_L_sh: OK')

    T_dot_B = np.zeros(nY, dtype=c_real_t)

    calc_T_dot_B_par(
        &T_dot_B[0],
        &Theta[0, 0], &Beta[0, 0],
        &ix_u[0], &ix_i[0], nY, k, nthreads
    )

    # T_dot_B = np.where(T_dot_B > Y_max, Y_max, T_dot_B)

    # np.nan_to_num(T_dot_B, copy=False)
    np.savetxt(os.path.join(save_folder, 'initial_value/T_dot_B.csv'), T_dot_B, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/T_dot_B.csv', T_dot_B, delimiter=',')
    # print('calc_T_dot_B: OK')

    Gamma_rte = np.tile(k_shp / k_rte, (1, k))
    Lambda_rte = np.tile(t_shp / t_rte, (1, k))

    initialize_parameters_G_L_rt_par(
        &T_dot_B[0],
        &ix_u[0], &ix_i[0], nY,
        &Theta[0, 0], &Beta[0, 0], k,
        &Gamma_rte[0, 0], &Lambda_rte[0, 0],
        &Y[0], nthreads
    )

    # np.nan_to_num(Gamma_rte, copy=False)
    # np.nan_to_num(Lambda_rte, copy=False)
    np.savetxt(os.path.join(save_folder, 'initial_value/Gamma_rte.csv'), Gamma_rte, fmt="%.10f", delimiter=',')
    np.savetxt(os.path.join(save_folder, 'initial_value/Lambda_rte.csv'), Lambda_rte, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/Gamma_rte.csv', Gamma_rte, delimiter=',')
    # np.savetxt('initial_value/Lambda_rte.csv', Lambda_rte, delimiter=',')
    # print('initialize_parameters_G_L_rt: OK')

    exp_T_dot_B = np.zeros(nY, dtype=c_real_t)

    calc_exp_T_dot_B_par(
        &exp_T_dot_B[0],
        &Gamma_shp[0, 0], &Gamma_rte[0, 0],
        &Lambda_shp[0, 0], &Lambda_rte[0, 0],
        &ix_u[0], &ix_i[0], nY, k, nthreads
    )

    # exp_T_dot_B = np.where(exp_T_dot_B > Y_max, Y_max, exp_T_dot_B)

    # np.nan_to_num(exp_T_dot_B, copy=False)
    np.savetxt(os.path.join(save_folder, 'initial_value/exp_T_dot_B.csv'), exp_T_dot_B, fmt="%.10f", delimiter=',')
    # np.savetxt('initial_value/exp_T_dot_B.csv', exp_T_dot_B, delimiter=',')
    # print('calc_exp_T_dot_B: OK')

    cdef int one = 1
    if verbose > 0:
        print("Initializing optimization procedure...")
    cdef double st_time = time.time()

    ### Main loop
    cdef int i
    for i in range(maxiter):

        ## Full-batch updates
        if full_updates:
            update_phi_par(&Gamma_shp[0, 0], &Gamma_rte[0, 0], &Lambda_shp[0, 0], &Lambda_rte[0, 0],
                           &phi[0, 0], &Y[0], k, sum_exp_trick,
                           &ix_u[0], &ix_i[0], nY, nthreads)

            # np.nan_to_num(phi, copy=False)
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/phi_' + str(i + 1) + '.csv'), phi,
                       fmt="%.10f", delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/phi_' + str(i + 1) + '.csv', phi, delimiter=',')
            # print('update_phi ' + str(i + 1) + '-th iter: OK')

            ### Comment: don't put this part before the update for Gamma rate
            Gamma_shp[:, :] = a
            Lambda_shp[:, :] = c

            update_G_n_L_sh_par(&Gamma_shp[0, 0], &Lambda_shp[0, 0],
                                &phi[0, 0], k,
                                &ix_u[0], &ix_i[0], nY, nthreads,
                                &Y[0])

            # np.nan_to_num(Gamma_shp, copy=False)
            # np.nan_to_num(Lambda_shp, copy=False)
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Gamma_shp_' + str(i + 1) + '.csv'),
                       Gamma_shp, fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Lambda_shp_' + str(i + 1) + '.csv'),
                       Lambda_shp, fmt="%.10f", delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Gamma_shp_' + str(i + 1) + '.csv', Gamma_shp, delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Lambda_shp_' + str(i + 1) + '.csv', Lambda_shp, delimiter=',')
            # print('update_G_n_L_sh ' + str(i + 1) + '-th iter: OK')

            if approx_rte:
                Gamma_rte_new = np.tile(k_shp / k_rte, (1, k))
                Lambda_rte_new = np.tile(t_shp / t_rte, (1, k))

                update_G_n_L_rt_approx(
                    &Gamma_rte_new[0, 0], &Lambda_rte_new[0, 0],
                    &exp_T_dot_B[0],
                    &Gamma_shp[0, 0], &Gamma_rte[0, 0],
                    &Lambda_shp[0, 0], &Lambda_rte[0, 0],
                    k_shp, &k_rte[0, 0],
                    t_shp, &t_rte[0, 0],
                    &ix_u[0], &ix_i[0],
                    nY, k, nU, nI,
                    &Y[0], nthreads
                )

                Gamma_rte = Gamma_rte_new
                Lambda_rte = Lambda_rte_new

                exp_T_dot_B = np.zeros(nY, dtype=c_real_t)

                calc_exp_T_dot_B_par(
                    &exp_T_dot_B[0],
                    &Gamma_shp[0, 0], &Gamma_rte[0, 0],
                    &Lambda_shp[0, 0], &Lambda_rte[0, 0],
                    &ix_u[0], &ix_i[0], nY, k, nthreads
                )
            else:
                update_G_n_L_rt_par(
                    &exp_T_dot_B[0],
                    &Gamma_shp[0, 0], &Gamma_rte[0, 0],
                    &Lambda_shp[0, 0], &Lambda_rte[0, 0],
                    k_shp, &k_rte[0, 0],
                    t_shp, &t_rte[0, 0],
                    &ix_u[0], &ix_i[0],
                    nY, k, nU, nI,
                    &Y[0], nthreads
                )

            # np.nan_to_num(Gamma_rte, copy=False)
            # np.nan_to_num(Lambda_rte, copy=False)
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Gamma_rte_' + str(i + 1) + '.csv'),
                       Gamma_rte, fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Lambda_rte_' + str(i + 1) + '.csv'),
                       Lambda_rte, fmt="%.10f", delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Gamma_rte_' + str(i + 1) + '.csv', Gamma_rte, delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Lambda_rte_' + str(i + 1) + '.csv', Lambda_rte, delimiter=',')
            # print('update_G_n_L_rt ' + str(i + 1) + '-th iter: OK')

            Theta[:, :] = Gamma_shp / Gamma_rte
            Beta[:, :] = Lambda_shp / Lambda_rte

            # np.nan_to_num(Theta, copy=False)
            # np.nan_to_num(Beta, copy=False)
            # np.nan_to_num(exp_T_dot_B, copy=False)
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Theta_' + str(i + 1) + '.csv'), Theta,
                       fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/Beta_' + str(i + 1) + '.csv'), Beta,
                       fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/exp_T_dot_B_' + str(i + 1) + '.csv'),
                       exp_T_dot_B, fmt="%.10f", delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Theta_' + str(i + 1) + '.csv', Theta, delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/Beta_' + str(i + 1) + '.csv', Beta, delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/exp_T_dot_B_' + str(i + 1) + '.csv', exp_T_dot_B, delimiter=',')

            k_rte = add_k_rte + Theta.sum(axis=1, keepdims=True)
            t_rte = add_t_rte + Beta.sum(axis=1, keepdims=True)

            # np.nan_to_num(k_rte, copy=False)
            # np.nan_to_num(t_rte, copy=False)
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/k_rte_' + str(i + 1) + '.csv'), k_rte,
                       fmt="%.10f", delimiter=',')
            np.savetxt(os.path.join(save_folder, 'updating_parameter/parameter/t_rte_' + str(i + 1) + '.csv'), t_rte,
                       fmt="%.10f", delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/k_rte_' + str(i + 1) + '.csv', k_rte, delimiter=',')
            # np.savetxt('D:/updating_parameter/parameter/t_rte_' + str(i + 1) + '.csv', t_rte, delimiter=',')

        ## assessing convergence
        if check_every > 0:
            has_converged, last_crit, last_rmse, approx_rte = assess_convergence(
                i, check_every, stop_crit, last_crit, last_rmse, stop_thr, approx_rte,
                Theta, Theta_prev,
                Beta, nY,
                Y, ix_u, ix_i, nYv,
                Yval, ix_u_val, ix_i_val,
                errs, k, nthreads, verbose, full_llk, has_valset,
                Y_max
            )
        if has_converged:
            break

    cdef double end_tm = (time.time() - st_time) / 60
    if verbose:
        print_final_msg(i + 1, <long long> last_crit, <double> last_rmse, end_tm)

    if keep_all_objs:
        temp = (Gamma_shp, Gamma_rte, Lambda_shp, Lambda_rte, k_rte, t_rte)
    else:
        temp = None
    return i, temp, last_crit
### External llk function
#########################
def calc_llk(np.ndarray[real_t, ndim=1] Y, np.ndarray[ind_type, ndim=1] ix_u,
             np.ndarray[ind_type, ndim=1] ix_i,
             np.ndarray[real_t, ndim=2] Theta, np.ndarray[real_t, ndim=2] Beta, ind_type k,
             int nthreads, int full_llk, int Y_max):
    cdef np.ndarray[real_t, ndim=1] o = np.zeros(1, dtype=c_real_t)
    llk_plus_rmse(&Theta[0, 0], &Beta[0, 0],
                  &Y[0], &ix_u[0], &ix_i[0],
                  <ind_type> Y.shape[0], k,
                  &o[0], nthreads, 0, full_llk, Y_max)
    cdef int kint = k
    # o[0] -= sum_prediction(&Theta[0, 0], &Beta[0, 0], &ix_u[0], &ix_i[0], <ind_type> Y.shape[0], kint, nthreads)
    return o[0]

### Internal C functions
########################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void initialize_parameters_G_L_sh_par(real_t * G_sh, real_t * L_sh,
                                           real_t * Latent, ind_type k,
                                           ind_type * ix_u, ind_type * ix_i, ind_type nY, int nthreads,
                                           real_t * Y) nogil:
    cdef ind_type i, j
    for i in prange(nY, schedule='static', num_threads=nthreads):
        for j in range(k):
            G_sh[ix_u[i] * k + j] += Latent[i * k + j]
            L_sh[ix_i[i] * k + j] += Latent[i * k + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void initialize_parameters_G_L_rt_par(
        real_t * T_dot_B,
        ind_type * ix_u, ind_type * ix_i, ind_type nY,
        real_t * T, real_t * B, ind_type k,
        real_t * G_rt, real_t * L_rt,
        real_t * Y, int nthreads
) nogil:
    cdef ind_type i, j
    for j in range(k):
        for i in prange(nY, schedule='static', num_threads=nthreads):
            G_rt[ix_u[i] * k + j] += Y[i] * B[ix_i[i] * k + j] / (T_dot_B[i] - T[ix_u[i] * k + j] * B[ix_i[i] * k + j])
            L_rt[ix_i[i] * k + j] += Y[i] * T[ix_u[i] * k + j] / (T_dot_B[i] - T[ix_u[i] * k + j] * B[ix_i[i] * k + j])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void calc_exp_T_dot_B_par(
        real_t * exp_T_dot_B,
        real_t * G_sh, real_t * G_rt,
        real_t * L_sh, real_t * L_rt,
        ind_type * ix_u, ind_type * ix_i, ind_type nY, ind_type k, int nthreads
) nogil:
    # cdef int k = <int> kszt

    cdef ind_type i, j
    for j in range(k):
        for i in prange(nY, schedule='static', num_threads=nthreads):
            # for i in range(nY):
            exp_T_dot_B[i] += G_sh[ix_u[i] * k + j] / G_rt[ix_u[i] * k + j] * L_sh[ix_i[i] * k + j] / L_rt[
                ix_i[i] * k + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void calc_T_dot_B_par(
        real_t * T_dot_B,
        real_t * T, real_t * B,
        ind_type * ix_u, ind_type * ix_i, ind_type nY, ind_type k, int nthreads
) nogil:
    cdef ind_type i, j
    for j in range(k):
        for i in prange(nY, schedule='static', num_threads=nthreads):
            # for i in range(nY):
            T_dot_B[i] += T[ix_u[i] * k + j] * B[ix_i[i] * k + j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void initialize_parameters_Latent_par(real_t * T, real_t * B, ind_type kszt,
                                           real_t * Y, real_t * Latent,
                                           ind_type * ix_u, ind_type * ix_i,
                                           ind_type nY, int nthreads) nogil:
    cdef ind_type uid, iid
    cdef ind_type uid_st, iid_st, Latent_st
    cdef real_t sumTB, maxval
    cdef ind_type i, j
    cdef int k = <int> kszt

    for i in prange(nY, schedule='static', num_threads=nthreads):
        # for i in range(nY):
        uid = ix_u[i]
        iid = ix_i[i]
        sumTB = 0.
        # maxval = - HUGE_VAL_T
        uid_st = k * uid
        iid_st = k * iid
        Latent_st = i * k
        for j in range(k):
            sumTB = sumTB + T[uid_st + j] * B[iid_st + j]

        for j in range(k):
            Latent[Latent_st + j] = Y[i] * T[uid_st + j] * B[iid_st + j] / sumTB

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_phi_par(real_t * G_sh, real_t * G_rt, real_t * L_sh, real_t * L_rt,
                         real_t * phi, real_t * Y, ind_type k, int sum_exp_trick,
                         ind_type * ix_u, ind_type * ix_i, ind_type nY, int nthreads) nogil:
    cdef ind_type uid, iid
    cdef ind_type uid_st, iid_st, phi_st
    cdef real_t sumphi, maxval
    cdef ind_type i, j

    for i in prange(nY, schedule='static', num_threads=nthreads):
        # for i in range(nY):
        uid = ix_u[i]
        iid = ix_i[i]
        sumphi = 0
        uid_st = k * uid
        iid_st = k * iid
        phi_st = i * k
        for j in range(k):
            phi[phi_st + j] = exp(
                psi(G_sh[uid_st + j]) - log(G_rt[uid_st + j]) + psi(L_sh[iid_st + j]) - log(L_rt[iid_st + j]))
            sumphi += phi[phi_st + j]
        for j in range(k):
            phi[phi_st + j] *= 1. / sumphi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_sh_par(real_t * G_sh, real_t * L_sh,
                              real_t * phi, ind_type k,
                              ind_type * ix_u, ind_type * ix_i, ind_type nY, int nthreads,
                              real_t * Y) nogil:
    cdef ind_type i, j
    for i in prange(nY, schedule='static', num_threads=nthreads):
        for j in range(k):
            G_sh[ix_u[i] * k + j] += phi[i * k + j] * Y[i]
            L_sh[ix_i[i] * k + j] += phi[i * k + j] * Y[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_rt_par(
        real_t * exp_T_dot_B,
        real_t * G_sh, real_t * G_rt,
        real_t * L_sh, real_t * L_rt,
        real_t k_sh, real_t * k_rt,
        real_t t_sh, real_t * t_rt,
        ind_type * ix_u, ind_type * ix_i,
        ind_type nY, ind_type k, ind_type nU, ind_type nI,
        real_t * Y, int nthreads
) nogil:
    cdef ind_type i, j, tmp_ix
    cdef real_t tmp_G_rt
    cdef real_t tmp_L_rt
    cdef int count = 1

    for j in range(k):
        for tmp_ix in range(nU):

            tmp_G_rt = G_rt[tmp_ix * k + j]
            G_rt[tmp_ix * k + j] = k_sh / k_rt[tmp_ix]
            for i in prange(nY, schedule='dynamic', num_threads=nthreads):
                if ix_u[i] == <ind_type> tmp_ix:
                    G_rt[tmp_ix * k + j] += Y[i] * L_sh[ix_i[i] * k + j] / L_rt[ix_i[i] * k + j] / exp_T_dot_B[i]

            for i in prange(nY, schedule='dynamic', num_threads=nthreads):
                if ix_u[i] == <ind_type> tmp_ix:
                    exp_T_dot_B[i] -= G_sh[tmp_ix * k + j] / tmp_G_rt * L_sh[ix_i[i] * k + j] / L_rt[ix_i[i] * k + j]
                    exp_T_dot_B[i] += G_sh[tmp_ix * k + j] / G_rt[tmp_ix * k + j] * L_sh[ix_i[i] * k + j] / L_rt[
                        ix_i[i] * k + j]
            with gil:
                print('update_G_n_L_rt (' + str(count) + ' / ' + str(k * (nU + nI)) + ')', end = "\r")
                count += 1
        for tmp_ix in range(nI):

            tmp_L_rt = L_rt[tmp_ix * k + j]
            L_rt[tmp_ix * k + j] = t_sh / t_rt[tmp_ix]

            for i in prange(nY, schedule='dynamic', num_threads=nthreads):
                if ix_i[i] == <ind_type> tmp_ix:
                    L_rt[tmp_ix * k + j] += Y[i] * G_sh[ix_u[i] * k + j] / G_rt[ix_u[i] * k + j] / exp_T_dot_B[i]

            for i in prange(nY, schedule='dynamic', num_threads=nthreads):
                if ix_i[i] == <ind_type> tmp_ix:
                    exp_T_dot_B[i] -= G_sh[ix_u[i] * k + j] / G_rt[ix_u[i] * k + j] * L_sh[tmp_ix * k + j] / tmp_L_rt
                    exp_T_dot_B[i] += G_sh[ix_u[i] * k + j] / G_rt[ix_u[i] * k + j] * L_sh[tmp_ix * k + j] / L_rt[
                        tmp_ix * k + j]
            with gil:
                print('update_G_n_L_rt (' + str(count) + ' / ' + str(k * (nU + nI)) + ')', end = "\r")
                count += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void update_G_n_L_rt_approx(
        real_t * G_rt_new, real_t * L_rt_new,
        real_t * exp_T_dot_B,
        real_t * G_sh, real_t * G_rt,
        real_t * L_sh, real_t * L_rt,
        real_t k_sh, real_t * k_rt,
        real_t t_sh, real_t * t_rt,
        ind_type * ix_u, ind_type * ix_i,
        ind_type nY, ind_type k, ind_type nU, ind_type nI,
        real_t * Y, int nthreads
) nogil:
    cdef ind_type i, j
    for j in range(k):
        for i in prange(nY, schedule='static', num_threads=nthreads):
            G_rt_new[ix_u[i] * k + j] += Y[i] * L_sh[ix_i[i] * k + j] / L_rt[ix_i[i] * k + j] / exp_T_dot_B[i]
            L_rt_new[ix_i[i] * k + j] += Y[i] * G_sh[ix_u[i] * k + j] / G_rt[ix_u[i] * k + j] / exp_T_dot_B[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void llk_plus_rmse(real_t * T, real_t * B, real_t * Y,
                        # Theta, Beta, Y
                        ind_type * ix_u, ind_type * ix_i, ind_type nY, ind_type kszt,
                        # ix_u, ix_i, nY, k
                        real_t * out, int nthreads, int add_mse, int full_llk, int Y_max) nogil:
    cdef ind_type i
    cdef int one = 1
    cdef real_t yhat
    cdef real_t out1 = 0
    cdef real_t out2 = 0
    cdef int k = <int> kszt

    if add_mse:
        for i in prange(nY, schedule='static', num_threads=nthreads):
            # for i in range(nY):
            yhat = tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
            # out1 += Y[i] * log(yhat) - loggamma(Y[i] + 1.)
            out1 += - yhat + Y[i] * log(yhat) - loggamma(Y[i] + 1.)
            if yhat > Y_max: yhat = Y_max
            out2 += (Y[i] - yhat) ** 2

        out[0] = out1
        out[1] = out2
    else:

        # for i in prange(nY, schedule='static', num_threads=nthreads):
        for i in range(nY):
            yhat = tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)
            # out1 += Y[i] * log(tdot(&k, &T[ix_u[i] * kszt], &one, &B[ix_i[i] * kszt], &one)) - loggamma(Y[i] + 1.)
            out1 += - yhat + Y[i] * log(yhat) - loggamma(Y[i] + 1.)
        out[0] = out1

### Printing output
###################
def print_norm_diff(int it, int check_every, real_t normdiff):
    print("Iteration %d | Norm(Theta_{%d} - Theta_{%d}): %.5f" % (it, it, it - check_every, normdiff))

def print_llk_iter(int it, long long llk, double rmse, int has_valset):
    cdef str dataset_type
    # dataset_type.encode("UTF-8")
    if has_valset:
        dataset_type = "val"
    else:
        dataset_type = "train"
    msg = "Iteration %d | " + dataset_type + " llk: %d | " + dataset_type + " rmse: %.4f"
    print(msg % (it, llk, rmse))

def print_final_msg(int it, long long llk, double rmse, double end_tm):
    print("\n\nOptimization finished")
    print("Final log-likelihood: %d" % llk)
    print("Final RMSE: %.4f" % rmse)
    print("Minutes taken (optimization part): %.1f" % end_tm)
    print("")
