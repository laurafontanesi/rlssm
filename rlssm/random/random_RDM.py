import numpy as np


def random_rdm_2A(cor_drift, inc_drift, threshold, ndt, noise_constant=1, dt=0.001, max_rt=10):
    shape = cor_drift.shape
    acc = np.empty(shape)
    rt = np.empty(shape)
    acc[:] = np.nan
    rt[:] = np.nan

    max_tsteps = max_rt / dt

    x_cor = np.zeros(shape)
    x_inc = np.zeros(shape)

    tstep = 0
    ongoing = np.array(np.ones(shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x_cor[ongoing] += np.random.normal(cor_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        x_inc[ongoing] += np.random.normal(inc_drift[ongoing] * dt,
                                           noise_constant * np.sqrt(dt),
                                           np.sum(ongoing))
        tstep += 1
        ended_correct = (x_cor >= threshold)
        ended_incorrect = (x_inc >= threshold)

        # store results and filter out ended trials
        if np.sum(ended_correct) > 0:
            acc[np.logical_and(ended_correct, ongoing)] = 1
            rt[np.logical_and(ended_correct, ongoing)] = dt * tstep + ndt[np.logical_and(ended_correct, ongoing)]
            ongoing[ended_correct] = False

        if np.sum(ended_incorrect) > 0:
            acc[np.logical_and(ended_incorrect, ongoing)] = 0
            rt[np.logical_and(ended_incorrect, ongoing)] = dt * tstep + ndt[np.logical_and(ended_incorrect, ongoing)]
            ongoing[ended_incorrect] = False
    return rt, acc


def simulate_rdm_2A(n_trials,
                    gen_cor_drift,
                    gen_inc_drift,
                    gen_threshold,  # Threshold should be A in function random_lba_2A
                    gen_ndt,
                    gen_rel_sp=.5,
                    participant_label=1,
                    gen_drift_trialsd=None,
                    gen_rel_sp_trialsd=None,
                    **kwargs):
    None


def simulate_hier_rdm(n_trials, n_participants,
                      gen_mu_drift, gen_sd_drift,
                      gen_mu_threshold, gen_sd_threshold,
                      gen_mu_ndt, gen_sd_ndt,
                      gen_mu_rel_sp=.5, gen_sd_rel_sp=None,
                      **kwargs):
    None
