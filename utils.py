# This code is created by Shyam Marjit. Some functions and codes are taken from 
# https://github.com/nikdon/pyEntropy and https://github.com/raphaelvallat/antropy.
# I give them credit for all these taken parts of the code.

# Import python library
import numpy as np
import scipy.stats as st
from pywt import wavedec
from itertools import chain
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.neighbors import KDTree
from hurst import compute_Hc, random_walk
from scipy.signal import periodogram, welch
from math import factorial, floor, log, sqrt
from scipy.stats import kurtosis, mode, skew
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# global variable
eeg_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 
                'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

def first_difference(input_data):
    # First order difference
    fd, temp = 0, 0
    for i in range(0, input_data.shape[0]-1):
        temp = abs(input_data[i+1]-input_data[i])
        fd += temp
    return fd/(input_data.shape[0]-1)

def second_difference(input_data):
    # second order difference
    sd, temp = 0, 0
    for i in range(0, input_data.shape[0] - 2):
        temp = abs(input_data[i+2]-input_data[i])
        sd += temp
    return sd/(input_data.shape[0]-2)

def avg_and_rms_power(input_data):
    # average power and root mean square of a signal
    mean_data, avg_power = np.mean(input_data), 0
    for i in range(input_data.shape[0]):
        temp = (mean_data - input_data[i])**2
        avg_power += temp
    return avg_power/(input_data.shape[0]), np.sqrt(avg_power/(input_data.shape[0]))

def katz_fractal_dimension(input_data):
    # Katz Fractal Dimension
    axis = -1
    x = np.asarray(input_data)
    dists = np.abs(np.diff(x, axis = axis))
    ll = dists.sum(axis = axis)
    ln = np.log10(ll / dists.mean(axis = axis))
    aux_d = x - np.take(x, indices = [0], axis = axis)
    d = np.max(np.abs(aux_d), axis = axis)
    kfd = np.squeeze(ln / (ln + np.log10(d / ll)))
    if not kfd.ndim:
        kfd = kfd.item()
    return kfd

def non_linear_energy(input_data):
    # Nonlinear Energy
    nle_value = 0
    for i in range(1, input_data.shape[0]-1):
        nle_value += (input_data[i]**2)-(input_data[i+1]*input_data[i-1])
    return nle_value

def num_zerocross(x, normalize=False, axis=-1):
    # Number of zero-crossings.
    x = np.asarray(x)
    nzc = np.diff(np.signbit(x), axis=axis).sum(axis=axis)
    if normalize:
        nzc = nzc / x.shape[axis]
    return nzc

def petrosian_fd(input_data):
    # Petrosian fractal dimension
    axis = -1
    x = np.asarray(input_data)
    N = x.shape[axis]
    # Number of sign changes in the first derivative of the signal
    nzc_deriv = num_zerocross(np.diff(x, axis=axis), axis=axis)
    pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc_deriv)))
    return pfd

def extrema(arr):
    # function to find number of local extremum
    n, count = arr.shape[0], 0
    a = arr.tolist()
    for i in range(1, n - 1): # start loop from position 1 till n-1
        count += (a[i]>a[i-1] and a[i]>a[i+1]);
        count += (a[i] < a[i - 1] and a[i] < a[i + 1]);
    return count

def shannon_entropy(time_series):
    if not isinstance(time_series, str): # Check if string
        time_series = list(time_series)
    data_set = list(set(time_series)) # Create a frequency data
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))
    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    return -ent

def _embed(x, order=3, delay=1):
    # Time-delay embedding
    x = np.asarray(x)
    N = x.shape[-1]
    if x.ndim == 1: # 1D array (n_times)
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
        return Y.T
    else: # 2D array (signal_indice, n_times)
        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)
        embed_signal_length = N - (order - 1) * delay
        # define the new signal length
        indice = [[(i * delay), (i * delay + embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal
        for i in range(order):
            # loop with the order
            temp = x[:, indice[i][0] : indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)
            Y.append(temp)
            # append the sliced signal to list
        return np.concatenate(Y, axis=-1)

def app_entropy(x, order=2, metric="chebyshev", approximate=True):
    _all_metrics = KDTree.valid_metrics
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = (KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64))
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64))
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return np.subtract(phi[0], phi[1])

def AntroPy_xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx

def perm_entropy(x, order=3, delay=1, normalize=False):
    # Permutation Entropy
    # If multiple delay are passed, return the average across all d
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = - AntroPy_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe

def pyentrp_embed(x, order=3, delay=1):
    # Time-delay embedding.
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def weighted_permutation_entropy(time_series, order=2, delay=1, normalize=False):
    x = pyentrp_embed(time_series, order=order, delay=delay)
    weights = np.var(x, axis=1)
    sorted_idx = x.argsort(kind='quicksort', axis=1)
    motifs, c = np.unique(sorted_idx, return_counts=True, axis=0)
    pw = np.zeros(len(motifs))
    # TODO hashmap
    for i, j in zip(weights, sorted_idx):
        idx = int(np.where((j == motifs).sum(1) == order)[0])
        pw[idx] += i
    pw /= weights.sum()
    b = np.log2(pw)
    wpe = -np.dot(pw, b)
    if normalize:
        wpe /= np.log2(factorial(order))
    return wpe

def _linear_regression(x, y):
    # Fast linear regression using Numba.
    epsilon = 10e-9
    n_times = x.size
    sx2, sx, sy, sxy = 0, 0, 0, 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den, num = n_times * sx2 - (sx**2), n_times * sxy - sx * sy
    slope = num / (den + epsilon)
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def higuchi_fd(x, kmax=10):
    # Higuchi Fractal Dimension.
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1.0 / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi

def _numba_sampen(x, order, r):
    # Fast evaluation of the sample entropy using Numba.
    n, n1 = x.size, n - 1
    order += 1
    order_dbld = 2 * order
    # initialize the lists
    run, run1 = [0] * n, run[:]
    r1 = [0] * (n * order_dbld)
    a = [0] * order
    b, p = a[:], a[:]
    for i in range(n1):
        nj = n1 - i
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = order if order < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1: b[m] += 1
            else:
                run[jj] = 0
        for j in range(order_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > order_dbld - 1:
            for j in range(order_dbld, nj):
                run1[j] = run[j]
    m = order - 1
    while m > 0:
        b[m] = b[m - 1]
        m -= 1
    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])

def _app_samp_entropy(x, order, metric="chebyshev", approximate=True):
    # Utility function for app_entropy and sample_entropy.
    _all_metrics = KDTree.valid_metrics
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)
    _emb_data1 = _embed(x, order, 1)
    if approximate: emb_data1 = _emb_data1
    else: emb_data1 = _emb_data1[:-1]
    count1 = (KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64))
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64))
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi

def sample_entropy(x, order=2, metric="chebyshev"):
    # Calculates the sample entropy of degree m of a time_series.
    x = np.asarray(x, dtype=np.float64)
    if metric == "chebyshev" and x.size < 5000:
        return _numba_sampen(x, order=order, r=(0.2 * x.std(ddof=0)))
    else:
        phi = _app_samp_entropy(x, order=order, metric=metric, approximate=False)
        return -np.log(np.divide(phi[1], phi[0]))
    
    return sampen

def _log_n(min_n, max_n, factor):
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor**i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

def detrended_fluctuation(x):
    '''
    Note: This function is adopted from AntroPy.
    Please refer https://github.com/raphaelvallat/antropy
    I give full credit for this function to AntroPy developers.
    '''
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[: N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope, intercept = _linear_regression(ran_n, d[i])
            trend[i, :] = intercept + slope * ran_n
        # Calculate root mean squares of walks in d around trend
        # Note that np.mean on specific axis is not supported by Numba
        flucs = np.sum((d - trend) ** 2, axis=1) / n
        # https://github.com/neuropsychology/NeuroKit/issues/206
        fluctuations[i_n] = np.sqrt(np.mean(flucs))

    # Filter zero
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        dfa = np.nan
    else:
        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
    return dfa

def hjorth_params(x, axis=-1):
    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    return x_var, mob, com

def svd_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= sum(W)
    svd_e = -AntroPy_xlogx(W).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e

def statistical_features(input_data, advanced = True):
    # Statistical features
    # Mean, Variance, Mode, Skew, Standard Deviation, Kurtosis
    mean, var, mode_ = np.mean(input_data), np.var(input_data), float(mode(input_data)[0]),
    median, skew_, std = np.median(input_data), skew(input_data), np.std(input_data)
    kurtosis_ = kurtosis(input_data)
    if(advanced == True):
        # First Difference, Second Difference, Normalized, First Difference, Normalized Second Difference
        first_diff = first_difference(input_data)
        norm_first_diff = first_diff/std
        sec_diff = second_difference(input_data)
        norm_sec_diff = sec_diff/std
        return [mean, var, mode_, median, skew_, std, kurtosis_, first_diff, norm_first_diff, sec_diff, norm_sec_diff]
    return [mean, var, mode_, median, skew_, std, kurtosis_]

def IWMF(psd, frqs):
    iwmf = 0
    temp = 0
    for i in range(psd.shape[0]):
        temp = psd[i]*frqs[i]
        iwmf+= temp
    return iwmf

def IWBW(psd, frqs):
    iwbw_1 = 0
    iwmf = IWMF(psd, frqs)
    for i in range(psd.shape[0]):
        temp_1 = (frqs[i]-iwmf)**2
        temp_2 = temp_1*psd[i]
        iwbw_1 = temp_2 + iwbw_1
    return sqrt(iwbw_1)

def calcNormalizedFFT(epoch,lvl,nt,fs=128):
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1]))
    D /= D.sum()
    return D

def SpectralEdgeFreq(epoch, lvl):
    # find the spectral edge frequency
    nt = 18
    fs = 512
    percent = 0.5
    sfreq = fs
    tfreq = 40
    ppow = percent
    topfreq = int(round(nt/sfreq*tfreq)) + 1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq])
    B = A - (A.max()*ppow)
    spedge = np.min(np.abs(B))
    spedge = (spedge - 1)/(topfreq - 1)*tfreq
    return spedge

def calcNormalizedFFT(epoch,lvl,nt,fs=128):
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1]))
    D /= D.sum()
    return D

def SpectralEdgeFreq(epoch, lvl):
    # find the spectral edge frequency
    nt = 18
    fs = 512
    percent = 0.5
    sfreq = fs
    tfreq = 40
    ppow = percent
    topfreq = int(round(nt/sfreq*tfreq)) + 1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq])
    B = A - (A.max()*ppow)
    spedge = np.min(np.abs(B))
    spedge = (spedge - 1)/(topfreq - 1)*tfreq
    return spedge

def spectral_entropy(x, sf, method="fft", nperseg=None, normalize=False, axis=-1):
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == "fft":
        _, psd = periodogram(x, sf, axis=axis)
    elif method == "welch":
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -AntroPy_xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se

def left_or_right(query):
    left = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1']
    right = ['Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    center = ['Oz', 'Pz', 'Fz', 'Cz']
    for i in left:
        if(query==i):
            return 'left'
            break
    for i in right:
        if(query==i):
            return 'right'
            break
    for i in center:
        if(query==i):
            return 'center'
            break

def get_channel_no(input_ch):
    if(len(input_ch)==1):
        for i in range(0, len(eeg_channels)):
            if(input_ch[0]==eeg_channels[i]):
                return i
    else:
        channel_no = []
        for ich in input_ch:
            channel_no.append(get_channel_no([ich]))
        return channel_no

def time_domain_features(data):
    features = []
    '''
    --------------------------------------------------------------------------------------------
    Extracted Time domain features:
    --------------------------------------------------------------------------------------------
    Mean, Variance, Mode, Median, Skew, Standard Deviation, Kurtosis, First Difference, Second 
    Difference, Normalized, First Difference, Normalized Second Difference, Energy, Average
    Power, RMS, Katz fractal dimension, Nonlinear Energy, Approximate Entropy, Shanon Entropy,
    Permutation Entropy, Sample Entropy, Weighted Permutation Entropy, Singular Value 
    Decomposition, Hurst Exponent, Higuchi fractal, dimension, Hjorth activity, mobility,
    complexity, Detrended Fluctuation Analysis, Number of local extrema, Number of 
    zero-crossings, Petrosian fractal dimension
    '''
    # Statistical features
    features = statistical_features(data)
    # Energy
    energy = sum(abs(data)**2)
    # Average Power and RMS
    avg_power, rms = avg_and_rms_power(data)
    # Katz fractal dimension
    kfd = katz_fractal_dimension(data)
    # Nonlinear Energy
    nonlinear_energy = non_linear_energy(data)
    # Approximate Entropy
    aentropy = app_entropy(data)
    # Shanon Entropy
    shannon_entp= shannon_entropy(data)
    # Permutation Entropy
    perm_entp = perm_entropy(data)
    # Sample Entropy
    sample_entp = sample_entropy(data)
    # Weighted Permutation Entropy
    WPE = weighted_permutation_entropy(data, order=3, normalize=False)
    # Singular Value Decomposition
    svd_entropy_val = svd_entropy(data, normalize=True)
    # Hurst Exponent
    H, c, data_HC = compute_Hc(data, kind='change', simplified=True)
    # Higuchi fractal dimension
    higuchi_val = higuchi_fd(data)
    # Petrosian fractal dimension
    petrosian_val = petrosian_fd(data)
    # Hjorth activity, mobility, and complexity
    hjorth_avability, hjorth_mobilty, hjorth_complexity = hjorth_params(data)
    # Detrended Fluctuation Analysis
    DFA = detrended_fluctuation(data)
    # Number of local extrema
    local_extrema = extrema(data)
    
    # Number of zero-crossings
    num_zerocross_val = num_zerocross(data)   # Number of zero-crossings
    
    features_ = [energy, avg_power, rms, kfd, nonlinear_energy, aentropy, shannon_entp, perm_entp, sample_entp, WPE,\
                 svd_entropy_val, svd_entropy_val, H, c, higuchi_val, petrosian_val, hjorth_avability, hjorth_mobilty,\
                 hjorth_complexity, DFA, local_extrema]
    features = features + features_
    return features

def Rational_Differential_Asymmetry(input_data, optimal_channels):
    # one can also find pair wise calculation for computing DASM and RASM. To do so 
    # please refer https://ieeexplore.ieee.org/document/6695876

    left_channels, right_channels = [], []
    for ith_channel in optimal_channels:
        if(left_or_right(ith_channel)=='left'):
            left_channels.append(ith_channel)
        elif(left_or_right(ith_channel)=='right'):
            right_channels.append(ith_channel)
        elif(left_or_right(ith_channel)=='center'):
            pass
        else:
            print('error')
            break

    # get the channel no
    left_ch_no = get_channel_no(left_channels)
    right_ch_no = get_channel_no(right_channels)
    
    # left power
    theta_left, alpha_left, beta_left, gamma_left = 0, 0, 0, 0
    for left_ch in left_ch_no:
        psd, freqs = plt.psd(input_data[left_ch], Fs = 128)
        # subbands
        theta_left+= np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
        alpha_left+= np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
        beta_left+= np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
        gamma_left+= np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])

    # right power
    theta_right, alpha_right, beta_right, gamma_right = 0, 0, 0, 0
    for right_ch in right_ch_no:
        psd, freqs = plt.psd(input_data[right_ch], Fs = 128)
        # subbands
        theta_right+= np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
        alpha_right+= np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
        beta_right+= np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
        gamma_right+= np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])

    DASM_theta = theta_left - theta_right
    DASM_alpha = alpha_left - alpha_right
    DASM_beta  = beta_left - beta_right
    DASM_gamma = gamma_left - gamma_right

    RASM_theta = theta_left/theta_right
    RASM_alpha = alpha_left/alpha_right
    RASM_beta  = beta_left/beta_right
    RASM_gamma = gamma_left/gamma_right

    features = [DASM_theta, DASM_alpha, DASM_beta, DASM_gamma, RASM_theta, RASM_alpha, RASM_beta, RASM_gamma]
    return features

def frequency_domain_features(data):
    '''
    Mean, Variance, Mode, Median, Skew, Standard
    Deviation, Kurtosis, Energy, Average Power, RMS for 5
    frequency bands( 0.5-4 Hz, 4-7 Hz, 8-13 Hz, 13-30 Hz,
    30-40 Hz) after applying PSD on raw data, [13]
    First Difference, Normalized First Difference, Second [14]
    Difference, Normalized Second Difference after applying PSD on raw data
    Intensity weighted mean frequency, Intensity weighted
    bandwidth, Spectral Edge Frequency, Spectral Entropy, [13]
    Mean of Peak Frequency after applying PSD on the raw data.
    Rational Asymmetry, Differential Asymmetry 
    '''
    # power spectral density
    # please refer: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.psd.html
    psd, freqs = plt.psd(data, Fs = 128)
    # get frequency bands mean power

    theta = psd[np.logical_and(freqs >= 4, freqs <= 7)]
    alpha = psd[np.logical_and(freqs >= 8, freqs <= 13)]
    beta  = psd[np.logical_and(freqs >= 13, freqs <= 30)]
    gamma = psd[np.logical_and(freqs >= 30, freqs <= 40)]

    # Statistical feature, computed from the DWT-based decomposed subbands
    theta_stat = statistical_features(theta, advanced = False)
    alpha_stat = statistical_features(alpha, advanced = False)
    beta_stat  = statistical_features(beta, advanced = False)
    gamma_stat = statistical_features(gamma, advanced = False)
    
    # First Difference, Normalized First Difference, Second Difference, Normalized 
    # Second Difference after applying PSD on raw data
    first_diff = first_difference(psd)
    norm_first_diff = first_diff/np.std(psd)
    sec_diff = second_difference(psd)
    norm_sec_diff = sec_diff/np.std(psd)


    # Energy calculation for each band
    theta_energy, alpha_energy = sum(abs(theta)**2), sum(abs(alpha)**2)
    beta_energy, gamma_energy = sum(abs(beta)**2), sum(abs(gamma)**2)
    
    # Average power and RMS for each band
    theta_avg_power, theta_rms = avg_and_rms_power(theta)
    alpha_avg_power, alpha_rms = avg_and_rms_power(alpha)
    beta_avg_power, beta_rms = avg_and_rms_power(beta)
    gamma_avg_power, gamma_rms = avg_and_rms_power(gamma)
    
    
    # Intensity weighted mean frequency
    iwmf = IWMF(psd, freqs)

    # Intensity weighted bandwidth
    iwbw = IWBW(psd, freqs)

    # Spectral Edge Frequency
    sef = SpectralEdgeFreq(psd, freqs)

    # Spectral Entropy
    spectral_entropy_val = spectral_entropy(data, sf=128, method='welch', normalize=True)

    # Mean of Peak Frequency after applying PSD on the raw data
    peaks, _ = find_peaks(psd, height = 0)
    peak_values = psd[peaks]
    avg_peak_value = np.mean(psd[peaks])
    
    features = theta_stat + alpha_stat + beta_stat + gamma_stat 
    temp = [first_diff, norm_first_diff, sec_diff, norm_sec_diff, theta_energy, alpha_energy, beta_energy, gamma_energy,
    theta_avg_power, theta_rms, alpha_avg_power, alpha_rms, beta_avg_power, beta_rms, gamma_avg_power, gamma_rms, iwmf,
    iwbw, sef, spectral_entropy_val, avg_peak_value]
    features = features + temp
    return features



def dwt_features(data):
    '''
    --------------------------------------------------------------------------------------------
    Extracted Discreate Wavelet domain features:
    --------------------------------------------------------------------------------------------
    Mean, Variance, Mode, Median, Skew, Standard deviation, Kurtosis,
    Energy, Average Power, RMS, Shannon Entropy, Approximate Entropy
    Permutation Entropy, Weighted Permutation Entropy, Hurst Exponent,
    Higuchi Fractal Dimension, Petrosian Fractal Dimension, Spectral
    Entropy, Mean of Peak Frequency, Auto Regressive and Auto Regressive
    moving Average model parameters computed on decomposition coefficients
    '''
    coeffs = wavedec(data, 'db1', level = 4)
    delta, theta, alpha, beta, gamma = coeffs
    
    # Statistical feature, computed from the DWT-based decomposed subbands
    theta_stat = statistical_features(theta, advanced = True)
    alpha_stat = statistical_features(alpha, advanced = True) 
    beta_stat  = statistical_features(beta, advanced = True)
    gamma_stat = statistical_features(gamma, advanced = True)
    
    # Energy calculation for each band
    theta_energy, alpha_energy  = sum(abs(theta)**2), sum(abs(alpha)**2)
    beta_energy, gamma_energy = sum(abs(beta)**2), sum(abs(gamma)**2)
    
    # Average power and RMS for each band
    theta_avg_power, theta_rms = avg_and_rms_power(theta)
    alpha_avg_power, alpha_rms = avg_and_rms_power(alpha)
    beta_avg_power, beta_rms = avg_and_rms_power(beta)
    gamma_avg_power, gamma_rms = avg_and_rms_power(gamma)
    
    # Shannon entropy (shEn)
    theta_ShEn, alpha_ShEn = shannon_entropy(theta), shannon_entropy(alpha)
    beta_ShEn, gamma_ShEn = shannon_entropy(beta), shannon_entropy(gamma)
    
    # Approximate entropy
    theta_aentropy, alpha_aentropy = app_entropy(theta), app_entropy(alpha)
    beta_aentropy, gamma_aentropy = app_entropy(beta), app_entropy(gamma)
    
    # Permutation entropy
    theta_pentropy, alpha_pentropy = perm_entropy(theta, normalize = True), perm_entropy(alpha, normalize = True)
    beta_pentropy, gamma_pentropy = perm_entropy(beta, normalize = True), perm_entropy(gamma, normalize = True)

    # Weigheted Permutation Entropy
    theta_wpe = weighted_permutation_entropy(theta, order = 3, normalize = False)
    alpha_wpe = weighted_permutation_entropy(alpha, order = 3, normalize = False)
    beta_wpe = weighted_permutation_entropy(beta, order = 3, normalize = False)
    gamma_wpe = weighted_permutation_entropy(gamma, order = 3, normalize = False)

    # Hurst Exponent(HE): Here we have two paramaters of HE i.e. H and c
    H_theta, c_theta, data_HC_theta = compute_Hc(theta, kind = 'change', simplified = True)
    H_alpha, c_alpha, data_HC_alpha = compute_Hc(alpha, kind = 'change', simplified = True)
    H_beta,  c_beta,  data_HC_beta  = compute_Hc(beta,  kind = 'change', simplified = True)
    H_gamma, c_gamma, data_HC_gamma = compute_Hc(gamma, kind = 'change', simplified = True)

    # Higuchi Fractal dimention
    higuchi_theta = higuchi_fd(theta) # Higuchi fractal dimension for theta
    higuchi_alpha = higuchi_fd(alpha) # Higuchi fractal dimension for alpha
    higuchi_beta  = higuchi_fd(beta)  # Higuchi fractal dimension for beta
    higuchi_gamma = higuchi_fd(gamma) # Higuchi fractal dimension for gamma
    
    # Petrosian fractal dimension
    petrosian_theta = petrosian_fd(theta) # Petrosian fractal dimension for theta
    petrosian_alpha = petrosian_fd(alpha) # Petrosian fractal dimension for alpha
    petrosian_beta  = petrosian_fd(beta)  # Petrosian fractal dimension for beta
    petrosian_gamma = petrosian_fd(gamma) # Petrosian fractal dimension for gamma
        
    # Auto regressive (AR)
    res_theta = AutoReg(theta,lags = 128).fit()
    res_alpha = AutoReg(alpha,lags = 128).fit()
    res_beta  = AutoReg(beta,lags = 128).fit()
    res_gamma = AutoReg(gamma,lags = 128).fit()
    aic_theta_ar, hqic_theta_ar, bic_theta_ar, llf_theta_ar = res_theta.aic, res_theta.hqic, res_theta.bic, res_theta.llf
    aic_alpha_ar, hqic_alpha_ar, bic_alpha_ar, llf_alpha_ar = res_alpha.aic, res_alpha.hqic, res_alpha.bic, res_alpha.llf
    aic_beta_ar,  hqic_beta_ar,  bic_beta_ar,  llf_beta_ar  = res_beta.aic,  res_beta.hqic,  res_beta.bic,  res_beta.llf
    aic_gamma_ar, hqic_gamma_ar, bic_gamma_ar, llf_gamma_ar = res_gamma.aic, res_gamma.hqic, res_gamma.bic, res_gamma.llf

    # Autoregressive moving Average (ARMA)
    try: arma_theta = ARIMA(theta, order=(5,1,0)).fit()
    except: arma_theta = ARIMA(theta, order=(3, 1,0)).fit()
    try: arma_alpha = ARIMA(alpha, order=(5,1,0)).fit()
    except: arma_alpha = ARIMA(alpha, order=(3,1,0)).fit()
    try: arma_beta = ARIMA(beta, order=(5,1,0)).fit()
    except: arma_beta = ARIMA(beta, order=(3,1,0)).fit()
    try: arma_gamma = ARIMA(gamma, order=(5,1,0)).fit()
    except: arma_gamma = ARIMA(gamma, order=(3,1,0)).fit()
    aic_theta_arma, hqic_theta_arma = arma_theta.aic, arma_theta.hqic
    bic_theta_arma, llf_theta_arma  = arma_theta.bic, arma_theta.llf
    aic_alpha_arma, hqic_alpha_arma = arma_alpha.aic, arma_alpha.hqic
    bic_alpha_arma, llf_alpha_arma = arma_alpha.bic, arma_alpha.llf
    aic_beta_arma,  hqic_beta_arma = arma_beta.aic, arma_beta.hqic
    bic_beta_arma,  llf_beta_arma  = arma_beta.bic, arma_beta.llf
    aic_gamma_arma, hqic_gamma_arma = arma_gamma.aic, arma_gamma.hqic
    bic_gamma_arma, llf_gamma_arma = arma_gamma.bic, arma_gamma.llf
    
    theta_vector = [theta_energy, theta_avg_power, theta_rms, theta_ShEn, theta_aentropy, theta_pentropy,
                    theta_wpe, H_theta, c_theta, higuchi_theta, petrosian_theta, aic_theta_ar, hqic_theta_ar, bic_theta_ar,
                    llf_theta_ar, aic_theta_arma, hqic_theta_arma, bic_theta_arma, llf_theta_arma]
    theta_vector = theta_stat + theta_vector
    
    alpha_vector = [alpha_energy, alpha_avg_power, alpha_rms, alpha_ShEn, alpha_aentropy, alpha_pentropy,
                    alpha_wpe, H_alpha, c_alpha, higuchi_alpha, petrosian_alpha, aic_alpha_ar, hqic_alpha_ar, bic_alpha_ar,
                    llf_alpha_ar, aic_alpha_arma, hqic_alpha_arma, bic_alpha_arma, llf_alpha_arma]
    alpha_vector = alpha_stat + alpha_vector
    
    beta_vector = [beta_energy, beta_avg_power, beta_rms, beta_ShEn, beta_aentropy, beta_pentropy,
                    beta_wpe, H_beta, c_beta, higuchi_beta, petrosian_beta, aic_beta_ar, hqic_beta_ar, bic_beta_ar,
                    llf_beta_ar, aic_beta_arma, hqic_beta_arma, bic_beta_arma, llf_beta_arma]
    beta_vector = beta_stat + beta_vector
    
    gamma_vector = [gamma_energy, gamma_avg_power, gamma_rms, gamma_ShEn, gamma_aentropy, gamma_pentropy,
                    gamma_wpe, H_gamma, c_gamma, higuchi_gamma, petrosian_gamma, aic_gamma_ar, hqic_gamma_ar, bic_gamma_ar,
                    llf_gamma_ar, aic_gamma_arma, hqic_gamma_arma, bic_gamma_arma, llf_gamma_arma]
    gamma_vector = gamma_stat + gamma_vector
    
    feature = [theta_vector, alpha_vector, beta_vector, gamma_vector]
    feature = list(chain.from_iterable(list(feature)))
    return feature