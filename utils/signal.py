import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import torch
import math
import torch.nn.functional as F
def cal_hr(output, Fs=30):
    """Compute heart rate for a single signal.
    Returns:
        hr: estimated heart rate (bpm) in range [40, 180]
        psd_x, psd_y: power spectral density outputs
    """
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    elif isinstance(output, torch.Tensor):
        pass

    def compute_complex_absolute_given_k(output, k, N):
        # return PSD (power spectral density)
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float, device=output.device) / N
        hanning = torch.tensor(np.hanning(N), device=output.device).view(1, -1) # 1, N -> process single signal

        output = output * hanning
        output = output.view(1, 1, -1)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute

    output = output.view(1, -1) # this line makes the function handle a single signal

    N = output.size()[1] # number of samples
    bpm_range = torch.arange(40, 180, dtype=torch.float, device=output.device) # heart rate range (bpm)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz # frequency-domain coordinates

    # compute PSD over feasible heart-rate band
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute # normalize so sum == 1
    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max returns (value, index)

    whole_max_idx = whole_max_idx.type(torch.float).to(output.device) # peak index corresponds to HR

    return whole_max_idx + 40, bpm_range.view(-1), complex_absolute	# analogous to a softmax over PSD

def cal_hr2(x, zero_pad=20, high_pass=180, low_pass=40, Fs=30):
    """Compute heart rate for a batch of signals.

    Args:
        x: tensor or numpy array of shape (bs, signal_length)
        zero_pad: zero-padding multiplier for frequency interpolation
        high_pass: upper bound for HR (bpm)
        low_pass: lower bound for HR (bpm)
        Fs: sampling frequency (Hz)

    Returns (CPU tensors or numpy arrays if input was numpy):
        HR: (bs, 1) estimated HR (bpm) corresponding to peak PSD in [40, 180]
        f: (bs, single-sided length) frequencies (bpm)
        x: (bs, single-sided length) single-sided PSD
    """
    if isinstance(x, np.ndarray):
        origin_type = "numpy"
        x = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        origin_type ="tensor"
        pass

    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise ValueError("x must be 1D or 2D tensor")

    x = x - torch.mean(x, dim=-1, keepdim=True)
    if zero_pad > 0:
        L = x.shape[-1]
        x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

    # Get PSD
    x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward')) # (bs, single-sided length, 2): real+imag
    x = torch.add(x[:,:, 0] ** 2, x[:,:, 1] ** 2) # magnitude squared -> power
    x = x / (Fs * x.shape[-1]) # divide by sampling freq and length to obtain single-sided PSD
    # Filter PSD for relevant parts
    Fn = Fs / 2
    freqs = torch.linspace(0, Fn, x.shape[-1], device=x.device)
    use_freqs = torch.logical_and(freqs >= low_pass / 60, freqs <= high_pass / 60)
    x = x[:, use_freqs]
    # Normalize PSD
    x[:,1:] = 2 * x[:,1:] # remove DC component factor; double the rest for single-sided PSD
    pxx = x / torch.sum(x, dim=-1, keepdim=True) # normalize PSD
    f = freqs[use_freqs] * 60
    HR = f[pxx.argmax(dim=-1, keepdim=True)]
    if origin_type == "numpy":
        HR = HR.numpy()
        f = f.numpy()
        pxx = pxx.numpy()
    return HR, f, pxx

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []

    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)

def hr_fft(sig, fs, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    try:
        peak_idx2 = peak_idx[sort_idx[1]]   # fallback to second peak if available
    except:
        peak_idx2 = peak_idx1

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr

def hr_fft_batch(sig_list, fs, harmonics_removal=True):
    # get heart rate by FFT (batch version)
    # return both heart rate and PSD

    hr_list = []
    for sig in sig_list:
        sig = sig.reshape(-1)
        sig = sig * signal.windows.hann(sig.shape[0])
        sig_f = np.abs(fft(sig))
        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
        sig_f_original = sig_f.copy()

        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])
        sort_idx = sort_idx[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        f_hr2 = peak_idx2 / sig.shape[0] * fs
        hr2 = f_hr2 * 60
        if harmonics_removal:
            if np.abs(hr1-2*hr2)<10:
                hr = hr2
            else:
                hr = hr1
        else:
            hr = hr1

        # x_hr = np.arange(len(sig))/len(sig)*fs*60
        hr_list.append(hr)
    return np.array(hr_list)

def normalize(x):
    return (x-x.mean())/x.std()

# import neurokit2 as nk
def get_hr_psd(signal, method="neurokit2"):
    hr, psd_y, psd_x = hr_fft(signal, 30)
    return hr, psd_y, psd_x


from typing import no_type_check
import numpy as np
import heartpy as hp

from scipy.interpolate import Akima1DInterpolator


def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)            #25 * 1000              25*1000
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    try:
        freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    except ZeroDivisionError:
        # unexpected sampling freq (Fs==0)
        return 123, 123
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=180., method='fast_ideal'):
                      #ppg     30HZ
    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)

        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        pp = 1
        if(type(freqs) == type(pp)):
            return 88
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

    elif method == 'fast_ideal_bimodal_filter':
        """ Same as above but check for secondary peak around 1/2 of first
        (to break the tie in case of occasional bimodal PS)
        Note - this may make metrics worse if the power spectrum is relatively flat
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        max_freq = freqs[max_ind]
        max_ps = ps[max_ind]

        # check for a second lower peak at 0.45-0.55f and >50% power
        freqs_valid = np.logical_and(freqs >= max_freq * 0.45, freqs <= max_freq * 0.55)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        if len(freqs) > 0:
            max_ind_lower = np.argmax(ps)
            max_freq_lower = freqs[max_ind_lower]
            max_ps_lower = ps[max_ind_lower]
        else:
            max_ps_lower = 0

        if max_ps_lower / max_ps > 0.50:
            return max_freq_lower
        else:
            return max_freq
    else:
        raise NotImplementedError


def predict_instantaneous_heart_rate(signal, Fs, window_size, step_size, min_hr=40., max_hr=180., method='fast_ideal'):
    N = len(signal)
    hr_signal = np.zeros((N,))
    hr_count = np.zeros((N,))
    if window_size > N:
        window_size = N
    final_index = N - window_size
    start_indices = list(range(0, final_index, step_size))
    if final_index not in start_indices:
        start_indices.append(final_index)
    window_func = np.hamming(window_size)
    for start_index in start_indices:
        segment_hr = predict_heart_rate(signal[start_index:start_index+window_size], Fs, min_hr, max_hr, method)
        hr_signal[start_index:start_index+window_size] += window_func * segment_hr
        hr_count[start_index:start_index+window_size] += window_func
    hr_signal /= hr_count
    return hr_signal
