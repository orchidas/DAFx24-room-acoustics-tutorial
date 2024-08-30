import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Union
from scipy.special import erfc


def db(x: NDArray[float],
       is_squared: bool = False,
       allow_inf: bool = True) -> NDArray[float]:
    """Converts values to decibels.

    Args:
        x (NDArray):
            value(s) to be converted to dB.
        is_squared (bool):
            Indicates whether `x` represents some power-like quantity (True) or some root-power-like quantity (False).
            Defaults to False, i.e. `x` is a root-power-like auqntity (e.g. Voltage, pressure, ...).
        allow_inf (bool):
            Whether an output value of -inf dB should be allowed (True) or raise an exception (False). Defaults to True.

    Returns:
        An array with the converted values, in dB.
    """
    x = np.abs(x)
    factor = 10.0 if is_squared else 20.0
    divide_mode = "ignore" if allow_inf is True else "raise"
    with np.errstate(divide=divide_mode):
        y = factor * np.log10(x)
    return y


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


def tf2minphase(tf: NDArray,
                axis: int = -1,
                is_even_fft: bool = True) -> NDArray:
    """Converts a transfer function to a minimum phase transfer function.

    Args:
        tf (NDArray): The original transfer function (single-sided spectrum, 0<=f<=fs/2)
        axis (int): The axis of H along which to calculate the minimum phase filters.
        is_even_fft (bool): Whether the two-sided conjugate-symmetric spectrum is even or odd in length,
            this affects the way we replicate the spectral coefficients.
            If not given (or None), we assume an even-length FFT was used,
            i.e. n_fft = 2 * (N - 1) where N is the number of frequency bins.

    Returns:
        NDArray: Spectral coefficients of the minimum phase filter (single-sided spectrum)
    """
    num_bins = tf.shape[axis]
    tf = one_to_two_sided_spectrum(tf, is_even_fft, axis)
    mag = np.abs(tf)
    phu = np.imag(sig.hilbert(-np.log(mag + EPS), axis=axis))
    ph = wrap_phase(phu)
    tf_mp: npt.NDArray
    tf_mp = mag * np.exp(1j * ph)
    # make one-sided spectrum again
    tf_mp = np.take(tf_mp, np.arange(num_bins), axis=axis)
    return tf_mp


def one_to_two_sided_spectrum(one_sided: NDArray,
                              is_even: bool = True,
                              freq_axis: int = -1) -> NDArray:
    """Convert a one-sided spectrum to a conjugate-symmetric two-sided spectrum (assuming real time-domain data)

    Only the real-parts of the 0 Hz and Nyquist frequency bins are used.

    Args:
        one_sided (NDArray): The single-sided spectral coefficients (complex)
            The frequency axis must correspond to frequencies in the range
            0 <= f <= fs/2 (Nyquist)
        is_even (bool, optional): Whether the two-sided spectrum had an even number of samples.
            Defaults to True.
        freq_axis (int, optional): The axis of single_sided that contains frequency bins. Defaults to -1.

    Returns:
        NDArray: The two-sided spectral coefficients
    """
    # mirror the spectrum
    two_sided: npt.NDArray
    two_sided = np.concatenate((one_sided, np.conj(one_sided[-2:0:-1])),
                               axis=0)
    if freq_axis != 0:
        one_sided = np.moveaxis(one_sided, source=freq_axis, destination=0)
    if is_even:
        # ensure that the Nyquist bin is real, ignoring phase
        one_sided[-1] = np.real(one_sided[-1])
        # mirror the spectrum
        two_sided = np.concatenate((one_sided, np.conj(one_sided[-2:0:-1])),
                                   axis=0)
    else:
        # mirror the spectrum
        two_sided = np.concatenate((one_sided, np.conj(one_sided[-1:0:-1])),
                                   axis=0)
    # ensure that the 0 Hz bin is real, ignoring phase
    two_sided[0] = np.real(two_sided[0])
    if freq_axis != 0:
        two_sided = np.moveaxis(two_sided, source=0, destination=freq_axis)
    return two_sided


def wrap_phase(ph_uw: NDArray, positive: bool = True) -> NDArray:
    """Wrap phase values into two pi range.

    Args:
        ph_uw (NDArray): Unwrapped phase values.
        positive (bool, optional): Values are position: 0->2pi. If False, values range: -pi->pi. Defaults to True.

    Returns:
        NDArray: Wrapped phase values.
    """
    twopi = 2 * np.pi
    if positive:
        ph_wrapped = np.remainder(ph_uw, twopi)
    else:
        ph_wrapped = ph_uw - np.round(ph_uw / twopi) * twopi
    return ph_wrapped


def estimate_echo_density(ir: np.ndarray,
                          fs: float,
                          win_ms: float = 20.,
                          end_early: bool = False) -> np.ndarray:
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