# PURE LBA_2A
import numpy as np
import pandas as pd


def random_lba_2A(k, A, tau, cor_drift, inc_drift, sd_drift):
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    b = k + A
    one_pose = True
    v_cor = np.random.normal(cor_drift, sd_drift, shape)
    v_inc = np.random.normal(inc_drift, sd_drift, shape)

    # this while loop might be wrong
    while one_pose:
        ind = np.logical_and(v_cor < 0, v_inc < 0)
        v_cor[ind] = np.random.normal(cor_drift[ind], np.ones(cor_drift[ind].shape))
        v_inc[ind] = np.random.normal(inc_drift[ind], np.ones(inc_drift[ind].shape))
        one_pose = np.sum(ind) > 0

    start_cor = np.random.uniform(np.zeros(A.shape), A)
    start_inc = np.random.uniform(np.zeros(A.shape), A)

    ttf_cor = (b - start_cor) / v_cor
    ttf_inc = (b - start_inc) / v_inc

    ind = np.logical_and(ttf_cor <= ttf_inc, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + tau[ind]

    ind = np.logical_and(ttf_inc < 0, 0 < ttf_cor)
    acc[ind] = 1
    rt[ind] = ttf_cor[ind] + tau[ind]

    ind = np.logical_and(ttf_inc < ttf_cor, 0 < ttf_inc)
    acc[ind] = 0
    rt[ind] = ttf_inc[ind] + tau[ind]

    ind = np.logical_and(ttf_cor < 0, 0 < ttf_inc)
    acc[ind] = 0
    rt[ind] = ttf_inc[ind] + tau[ind]

    return rt, acc


def simulate_lba_2A(n_trials,
                    gen_cor_drift,
                    gen_inc_drift,
                    gen_A,  # Threshold should be A in function random_lba_2A
                    gen_tau,  # gen_tau is gen_ndt
                    gen_k,  # gen_k is gen_rel_sp
                    participant_label=1):
    None


def simulate_hier_lba(n_trials, n_participants,
                      gen_mu_drift, gen_sd_drift,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                      **kwargs):
    None
