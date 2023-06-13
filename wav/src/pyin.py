import numpy as np
from src.distributions import get_triangular_distribution, get_beta_distribution
from src.yin import cumulative_mean_normalized_difference_function, parabolic_interpolation


def pyin(audio_signal,
         sampling_rate=22050,
         window_size=2048,
         hop_size=256,
         min_frequency=55.0,
         max_frequency=1760.0,
         frequency_resolution=10,
         thresholds=np.arange(0.01, 1, 0.01),
         alpha=1,
         beta=18,
         absolute_min_prob=0.01,
         voicing_prob=0.5):

    min_period = max(int(np.ceil(sampling_rate / max_frequency)), 1)
    max_period = int(np.ceil(sampling_rate / min_frequency))

    x_pad = np.concatenate((np.zeros(window_size // 2), audio_signal, np.zeros(window_size // 2)))  # Add zeros for centered estimates

    beta_distribution = get_beta_distribution(alpha, beta, thresholds)

    B = int(np.log2(max_frequency / min_frequency) * (1200 / frequency_resolution))
    frequency_axis = min_frequency * np.power(2, np.arange(B) * frequency_resolution / 1200)
    observations, rms, _, _ = yin_multi_thr(x_pad, sampling_rate, window_size, hop_size, min_period,
                                                        max_period, thresholds, beta_distribution, absolute_min_prob,
                                                        frequency_axis, voicing_prob)

    triangular_distribution = get_triangular_distribution(frequency_resolution)
    transition_matrix = compute_transition_matrix(B, triangular_distribution)

    # HMM smoothing
    C = np.ones((2 * B, 1)) / (2 * B)  # uniform initialization
    f0_idxs = viterbi_log_likelihood(transition_matrix, C.flatten(), observations)  # libfmp Viterbi implementation

    # Obtain F0-trajectory
    F_axis_extended = np.concatenate((frequency_axis, np.zeros(len(frequency_axis))))
    f0 = F_axis_extended[f0_idxs]

    # Suppress low power estimates
    f0[0] = 0  # due to algorithmic reasons, we set the first value unvoiced
    f0[rms < 0.01] = 0

    # confidence
    O_norm = observations[:, np.arange(observations.shape[1])] / np.max(observations, axis=0)
    conf = O_norm[f0_idxs, np.arange(observations.shape[1])]

    times = np.arange(observations.shape[1]) * hop_size / sampling_rate

    return f0, times, conf


def probabilistic_thresholding(cmndf, thresholds, min_period, max_period, absolute_min_prob, F_axis, Fs, beta_distr,
                               parabolic_interp=True):
    cmndf[:min_period] = np.inf
    cmndf[max_period:] = np.inf

    min_indexes = (np.argwhere((cmndf[1:-1] < cmndf[0:-2]) & (cmndf[1:-1] < cmndf[2:]))).flatten().astype(np.int64) + 1

    observations = np.zeros(2 * len(F_axis))

    if min_indexes.size == 0:
        return observations, np.ones_like(thresholds) * min_period, np.ones_like(thresholds)

    # Optional: Parabolic Interpolation of local minima
    if parabolic_interp:
        # do not interpolate at the boarders, Numba compatible workaround for np.delete()
        min_idxs_interp = delete_numba(min_indexes, np.argwhere(min_indexes == min_period))
        min_idxs_interp = delete_numba(min_idxs_interp, np.argwhere(min_idxs_interp == max_period - 1))
        p_corr, cmndf[min_idxs_interp] = parabolic_interpolation(cmndf[min_idxs_interp - 1],
                                                                 cmndf[min_idxs_interp],
                                                                 cmndf[min_idxs_interp + 1])
    else:
        p_corr = np.zeros_like(min_indexes).astype(np.float64)

    # set p_corr=0 at the boarders (no correction done later)
    if min_indexes[0] == min_period:
        p_corr = np.concatenate((np.array([0.0]), p_corr))

    if min_indexes[-1] == max_period - 1:
        p_corr = np.concatenate((p_corr, np.array([0.0])))

    yin_estimates = np.zeros_like(thresholds)
    val_thr = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        min_idxs_thr = min_indexes[cmndf[min_indexes] < threshold]

        if not min_idxs_thr.size:
            lag = np.argmin(cmndf)
            am_prob = absolute_min_prob
            val = np.min(cmndf)
        else:
            am_prob = 1
            lag = np.min(min_idxs_thr)
            val = cmndf[lag]

            if parabolic_interp:
                lag += p_corr[np.argmin(min_idxs_thr)]

        if lag < min_period:
            lag = min_period
        elif lag >= max_period:
            lag = max_period - 1

        yin_estimates[i] = lag
        val_thr[i] = val

        idx = np.argmin(np.abs(1200 * np.log2(F_axis / (Fs / lag))))
        observations[idx] += am_prob * beta_distr[i]

    return observations, yin_estimates, val_thr


def yin_multi_thr(audio_signal, sampling_rate, window_size, hop_size, min_period, max_period, thresholds,
                  beta_distribution, absolute_min_prob, frequency_axis, voicing_prob, parabolic_interp=True):
    estimates_count = int(np.floor((len(audio_signal) - window_size) / hop_size)) + 1
    B = len(frequency_axis)

    rms = np.zeros(estimates_count)  # RMS Power
    O = np.zeros((2 * B, estimates_count))  # every voiced state has an unvoiced state (important for later HMM modeling)
    p_orig = np.zeros((len(thresholds), estimates_count))
    val_orig = np.zeros((len(thresholds), estimates_count))

    for m in range(estimates_count):
        frame = audio_signal[m * hop_size:m * hop_size + window_size]
        cmndf = cumulative_mean_normalized_difference_function(frame, max_period)
        rms[m] = np.sqrt(np.mean(frame ** 2))

        observations, _, _ = probabilistic_thresholding(
            cmndf, thresholds, min_period, max_period, absolute_min_prob, frequency_axis,
            sampling_rate, beta_distribution, parabolic_interp)

        O[:, m] = observations

    O[0:B, :] *= voicing_prob
    b = (1 - voicing_prob) * (1 - np.sum(O[0:B, :], axis=0)) / B
    O[B:2 * B, :] = b

    return O, rms, p_orig, val_orig


def compute_transition_matrix(M, triang_distr):
    prob_self = 0.99
    matrix = np.zeros((2 * M, 2 * M))
    max_step = len(triang_distr) // 2

    for i in range(M):
        if i < max_step:
            probability = prob_self * triang_distr[max_step - i:-1] / np.sum(triang_distr[max_step - i:-1])
            matrix[i, 0:i + max_step] = probability
            matrix[i + M, M:i + M + max_step] = probability

        if max_step <= i < M - max_step:
            probability = prob_self * triang_distr
            matrix[i, i - max_step:i + max_step + 1] = probability
            matrix[i + M, (i + M) - max_step:(i + M) + max_step + 1] = probability

        if M - max_step <= i:
            probability = prob_self * triang_distr[0:max_step - (i - M)] / np.sum(triang_distr[0:max_step - (i - M)])
            matrix[i, i - max_step:M] = probability
            matrix[i + M, i + M - max_step:2 * M] = probability

        matrix[i, i + M] = 1 - prob_self
        matrix[i + M, i] = 1 - prob_self

    return matrix


def viterbi_pyin(transition_matrix, initial_state_probabilities, emission_matrix):
    B = emission_matrix.shape[0] // 2
    M = emission_matrix.shape[1]
    D = np.zeros((B * 2, M))
    E = np.zeros((B * 2, M - 1))

    idxs = np.zeros(M)

    for i in range(B * 2):
        D[i, 0] = initial_state_probabilities[i, 0] * emission_matrix[i, 0]  # D matrix Intial state setting

    D[:, 0] = D[:, 0] / np.sum(D[:, 0])  # Normalization (using pYIN source code as a basis)

    for n in range(1, M):
        for i in range(B * 2):
            abyd = np.multiply(transition_matrix[:, i], D[:, n - 1])
            D[i, n] = np.max(abyd) * emission_matrix[i, n]
            E[i, n - 1] = np.argmax(abyd)

        D[:, n] = D[:, n] / np.sum(D[:, n])  # Row normalization to avoid underflow (pYIN source code sparseHMM)

    idxs[M - 1] = np.argmax(D[:, M - 1])

    for n in range(M - 2, 0, -1):
        bkd = int(idxs[n + 1])  # Intermediate variable to be compatible with Numba
        idxs[n] = E[bkd, n]

    return idxs.astype(np.int32)


def viterbi_log_likelihood(transition_matrix, initial_state_probabilities, emission_matrix):
    number_of_states = transition_matrix.shape[0]
    observation_seq_len = emission_matrix.shape[1]
    tiny = np.finfo(0.).tiny
    transition_matrix_log = np.log(transition_matrix + tiny)
    initial_state_probabilities_log = np.log(initial_state_probabilities + tiny)
    emission_matrix_log = np.log(emission_matrix + tiny)

    d_log = np.zeros((number_of_states, observation_seq_len))
    e = np.zeros((number_of_states, observation_seq_len - 1)).astype(np.int32)
    d_log[:, 0] = initial_state_probabilities_log + emission_matrix_log[:, 0]

    for n in range(1, observation_seq_len):
        for i in range(number_of_states):
            temp_sum = transition_matrix_log[:, i] + d_log[:, n - 1]
            d_log[i, n] = np.max(temp_sum) + emission_matrix_log[i, n]
            e[i, n - 1] = np.argmax(temp_sum)

    s_opt = np.zeros(observation_seq_len).astype(np.int32)
    s_opt[-1] = np.argmax(d_log[:, -1])
    for n in range(observation_seq_len - 2, -1, -1):
        s_opt[n] = e[int(s_opt[n + 1]), n]

    return s_opt


def delete_numba(arr, num):
    """Delete number from array, Numba compatible. Inspired by:
        https://stackoverflow.com/questions/53602663/delete-a-row-in-numpy-array-in-numba
    """
    mask = np.zeros(len(arr), dtype=np.int64) == 0
    mask[np.where(arr == num)[0]] = False
    return arr[mask]
