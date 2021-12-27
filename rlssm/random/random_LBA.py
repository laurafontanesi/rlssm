# PURE LBA_2A
import numpy as np


def random_lba_2A(k, A, tau, cor_drift, inc_drift):
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    b = k + A
    one_pose = True
    v_cor = np.random.normal(cor_drift, np.ones(cor_drift.shape))
    v_inc = np.random.normal(inc_drift, np.ones(inc_drift.shape))

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
