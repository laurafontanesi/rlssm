import numpy as np


# PURE RDM_2A
def rdm_trial(I, threshold, non_decision_time, noise_constant=1, dt=0.001, max_rt=10):
    n_choice = len(I)
    x = np.zeros(n_choice)
    stop_race = False
    rt = 0

    while not stop_race:
        for i in range(n_choice):
            x[i] += np.random.normal(I[i] * dt, noise_constant * (dt ** (1 / 2)), 1)[0]
        rt += dt
        not_reached = np.sum(x < threshold)
        if not_reached == n_choice:
            stop_race = False
            if rt > max_rt:
                x = np.zeros(n_choice)
                rt = 0
        elif not_reached == n_choice - 1:
            stop_race = True
        else:
            stop_race = False
            x = np.zeros(n_choice)
            rt = 0

    return rt + non_decision_time, np.where(x >= threshold)[0][0] + 1


def random_rdm_nA(drift, threshold, ndt, noise_constant=1, dt=0.001, max_rt=10):
    shape = ndt.shape
    n_options = drift.shape[1]
    choice = np.empty(shape)
    choice[:] = np.nan
    rt = np.empty(shape)
    rt[:] = np.nan

    max_tsteps = max_rt / dt

    x = np.zeros(drift.shape)
    tstep = 0
    ongoing = np.array(np.ones(drift.shape), dtype=bool)
    ended = np.array(np.ones(drift.shape), dtype=bool)

    stop_race = False

    while np.sum(ongoing) > 0 and tstep < max_tsteps:
        x[ongoing] += np.random.normal(drift[ongoing] * dt,
                                       noise_constant * np.sqrt(dt),
                                       np.sum(ongoing))
        tstep += 1

        for i in range(n_options):
            ended[:, i, :] = (x[:, i, :] >= threshold)

        # store results and filter out ended trials
        for i in range(n_options):
            if np.sum(ended[:, i, :]) > 0:
                choice[np.logical_and(ended[:, i, :], ongoing[:, i, :])] = i + 1
                rt[np.logical_and(ended[:, i, :], ongoing[:, i, :])] = dt * tstep + ndt[
                    np.logical_and(ended[:, i, :], ongoing[:, i, :])]
                ongoing[:, i, :][ended[:, i, :]] = False

    return rt, choice


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
