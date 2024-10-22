import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Union, Any, Tuple, Optional
from scipy.special import erfc
from scipy.signal import fftconvolve, stft
from scipy.signal.windows import hann


def db(x: NDArray[float],
       is_squared: bool = False,
       min_value: float = -200) -> NDArray[float]:
    """Converts values to decibels.

    Args:
        x (NDArray):
            value(s) to be converted to dB.
        is_squared (bool):
            Indicates whether `x` represents some power-like quantity (True) or some root-power-like quantity (False).
            Defaults to False, i.e. `x` is a root-power-like auqntity (e.g. Voltage, pressure, ...).
        min_value (float): cap the decibels to this value, cannot be lower

    Returns:
        An array with the converted values, in dB.
    """
    x = np.abs(x)
    factor = 10.0 if is_squared else 20.0
    y = factor * np.log10(x + np.finfo(np.float64).eps)
    return y.clip(min=min_value)


def db2lin(x: ArrayLike) -> NDArray:
    """Convert from decibels to linear

    Args:
        x (ArrayLike): value(s) to be converted

    Returns:
        (ArrayLike): values converted to linear
    """
    return np.power(10.0, x * 0.05)


def ms_to_samps(ms: Union[float, ArrayLike],
                fs: float) -> Union[int, ArrayLike]:
    samp = ms * 1e-3 * fs
    if isinstance(samp, np.ndarray):
        return samp.astype(np.int32)
    else:
        return int(samp)


def estimate_echo_density(ir: NDArray,
                          fs: float,
                          win_ms: float = 20.,
                          end_early: bool = False) -> NDArray:
    """Estimate the normalised echo density in an impulse response.

    The echo density signal is normalised such that a value of 1 corresponds to the expected echo density of a Gaussian
    noise sequence.

    Abel & Huang (2006) "A simple, robust measure of reverberation echo density", 121st AES Convention

    Args:
        ir (np.ndarray): The impulse response (time in axis 0).
            The array should start at the arrival of the direct sound.
        fs (float): Sample rate (Hertz)
        win_ms (float, optional): Determines the window length in milliseconds for echo density analysis.
            Abel and Huang suggest window duration of 20-30 ms, by default we use 20 ms.
        end_early (bool, optional): Exit iteration loop once normalized echo density exceeds 1, by default False.
            This can be used to speed up calculation when we are only interested in the first point it exceeds one.

    Returns:
        np.ndarray: The normalised echo density function
    """
    echo_d = np.zeros_like(ir)
    half_win_len = int(np.round(win_ms * 5e-4 * fs))
    win_len = 2 * half_win_len  # ensures even window length
    num = ir.shape[0]
    assert num > win_len, "Impulse response must be longer than analysis window"
    norm_val = 1 / (win_len * erfc(1 / np.sqrt(2)))
    for n in np.arange(num):
        if n <= half_win_len:
            ir_seg = ir[:(n + half_win_len - 1)]
        elif half_win_len < n <= num - half_win_len:
            ir_seg = ir[(n - half_win_len - 1):(n + half_win_len - 1)]
        else:
            ir_seg = ir[(n - half_win_len - 1):]
        echo_d[n] = np.sum(np.abs(ir_seg) > np.std(ir_seg, axis=0),
                           axis=0) * norm_val
        if end_early and echo_d[n] > 1.0:
            echo_d = echo_d[:(n + 1)]
            break
    return echo_d


def conv_binaural(brir: NDArray[Any],
                  sig: NDArray[Any],
                  ear_axis: int = 0,
                  mode="full") -> NDArray[Any]:
    """Easy binaural convolution with single channel signal
    Args:
        brir (ndarray): binaural impulse responses
        sig (ndarray): mono/stereo channel signal
        ear_axis (optional, int): axis of brir corresponding to ear selection, 0 by default
        mode (str - full or same): if full, then len(brir) + len(sig) is returned, if 'same',
                                    len(brir) is returned

    Returns:
        ndarray: time domain output
    """
    # convert to dual mono signal
    if sig.ndim == 1:
        sig = np.vstack((sig, sig))
        sig = sig.T if ear_axis > 0 else sig
    assert len(brir.shape) == 2

    if ear_axis != 0:
        brir = brir.T
        sig = sig.T

    left_chan = brir[0, :]
    right_chan = brir[1, :]
    sig_left = sig[0, :]
    sig_right = sig[1, :]
    time_axis = 1

    sig_len = sig.shape[time_axis]
    sb = np.real(fftconvolve(left_chan, sig_left, mode))
    sb = np.tile(sb[np.newaxis, :], (2, 1))
    sb[1] = np.real(fftconvolve(right_chan, sig_right, mode))

    if mode == "same":
        sb = sb[:, sig_len]

    if ear_axis != 0:
        sb = sb.T

    return sb


def interaural_coherence_menzer(
    signal: NDArray,
    fs: float,
    time_axis: int = -1,
    ear_axis: int = -2,
    tf_win_len: int = 256,
    use_real: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Estimate interaural coherence for a binaural signal.

    Interaural coherence (IC) is calculated according to "Investigations on an early reflection-free
    model for BRIRs" by Menzer et al., Journal of the Audio Engineering Society, 2010.

    Args:
        signal (np.ndarray): Binaural signal (at least two-dimensional)
        fs (float): Sample rate (Hertz)
        time_axis (int, optional): Axis for time series in binaural_signal. Defaults to -1.
        ear_axis (int, optional): Axis for ear indices in binaural_signal (0: left, 1: right). Defaults to -2.
        tf_win_len (int, optional): The temporal window length for time-frequency analysis. Defaults to 256.
        use_real (bool, optional): Is the numerator real or absolute for IAC calculation. Defaults to False.

    Returns:
        np.ndarray: Sub-band IAC estimate.
        np.ndarray: Frequencies of sub-bands.
    """
    if signal.ndim < 2:
        raise ValueError("`signal` must be at least two-dimensional")
    if signal.shape[ear_axis] != 2:
        raise ValueError("The binaural signal must have size 2 in ear axis")
    if ear_axis == time_axis:
        raise ValueError("ear_axis and time_axis cannot be the same")

    if time_axis == -2:
        signal = np.moveaxis(signal, 1, 0)
        time_axis = -1
        ear_axis = -2

    sig_len = signal.shape[-1]
    if not np.log2(tf_win_len).is_integer():
        tf_win_len = np.int32(2**np.round(np.log2(tf_win_len)))

    # perform an STFT on the signal
    tf_overlap = tf_win_len // 2  # 50% overlap
    tf_hop = tf_win_len - tf_overlap  # calc hop (if overlap is not 50%)
    tf_nfft = tf_win_len * 2
    num_f = tf_nfft // 2 + 1
    f = np.linspace(0, fs / 2, num_f)
    t = np.arange(0, sig_len + tf_hop, tf_hop) / fs
    num_t = t.shape[0]

    tf_win = hann(tf_win_len)
    # time-freq noise signal
    f, t, tf_signal = stft(
        signal,
        fs,
        window=tf_win,
        nperseg=tf_win_len,
        noverlap=tf_hop,
        nfft=tf_nfft,
        axis=time_axis,
        return_onesided=True,
    )
    tf_signal = np.transpose(tf_signal, (1, 2, 0))  # move time to axis
    assert tf_signal.shape[0] == num_f
    assert tf_signal.shape[1] == num_t

    mod_func = np.real if use_real else np.abs
    tf_left = tf_signal[..., 0]
    tf_right = tf_signal[..., 1]
    numerator = mod_func(np.sum(tf_left * np.conj(tf_right), axis=1))
    denominator = np.sqrt(
        np.sum(np.abs(tf_left)**2, axis=1) *
        np.sum(np.abs(tf_right)**2, axis=1))
    iac = numerator / denominator
    assert iac.shape[0] == num_f
    return iac, f