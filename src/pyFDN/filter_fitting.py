"""IIR Filter design with Prony's method and warped Prony's method,
adapted from Jatin Chowdhury's audio_dspy toolbox avaiable at
https://github.com/jatinchowdhury18/audio_dspy/

Author : Orchisama Das
Date : 27/07/2022
"""
from typing import Tuple, Optional
from numpy.typing import NDArray, ArrayLike

import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from scipy.linalg import toeplitz
from scipy.signal import tf2zpk, zpk2tf, hilbert
from scipy.fft import fftfreq, rfftfreq, fft, ifft, irfft

from .utils import db2lin, db

# pylint: disable=invalid-name


def one_to_two_sided_spectrum(one_sided: ArrayLike,
                              is_even: bool = True) -> ArrayLike:
    """Convert a one-sided spectrum to a conjugate-symmetric two-sided spectrum (assuming real time-domain data)

    Only the real-parts of the 0 Hz and Nyquist frequency bins are used.

    Args:
        one_sided (NDArray): The single-sided spectral coefficients (complex)
            The frequency axis must correspond to frequencies in the range
            0 <= f <= fs/2 (Nyquist)
        is_even (bool, optional): Whether the two-sided spectrum had an even number of samples.
            Defaults to True.

    Returns:
        NDArray: The two-sided spectral coefficients
    """
    two_sided: ArrayLike

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
    return two_sided


def interpolate_magnitude_spectrum(
        mag_spec: NDArray,
        freqs: NDArray,
        fs: int,
        n_fft: int,
        cutoff: Tuple[float] = (63, 16000),
        rolloff_dc_db: Optional[float] = None,
        rolloff_nyq_db: Optional[float] = None,
        return_one_sided: bool = True,
        bands_per_octave: int = 1) -> Tuple[NDArray, NDArray]:
    """Interpolate magnitude spectrum on a finer axis before feeding it into Trueplay fitter

    Args:
        mag_spec (np.ndarray): The spectrum to be interpolated (magnitude only)
        freqs (np.ndarray) : The frequencies at which spectrum has been calculated
        fs (float): The sampling rate in Hertz
        n_fft (int): Number of points in the FFT (bins in new spectrum two-sided spectrum)
        cutoff (Tuple[float]): Frequencies below and above which magnitude spectrum is discarded
        rolloff_dc_db (float) : roll-off rate at db/zeta octave from the low cutoff to DC
        rolloff_nyq_db (float) : roll-off rate at db/octave from high cutoff to nyquist
        return_one_sided (bool) : return only the positive spectrum
        bands_per_octave (optional, int): The octave band resolution of the T60

    Returns:
        np.ndarray: The interpolated magnitude spectrum
        np.ndarray: The new frequency axis
    """
    # calculate roll off value by looking at neighbours in magnitude spectrum
    if rolloff_nyq_db is None:
        rolloff_nyq_db = db(mag_spec[-1] - mag_spec[-2]) * bands_per_octave
    if rolloff_dc_db is None:
        rolloff_dc_db = -db(
            np.abs(mag_spec[1] - mag_spec[0])) * bands_per_octave

    # interpolate magnitude response onto FFT frequency grid
    new_freqs = rfftfreq(n_fft, d=1 / fs)
    n_bins_one_sided = new_freqs.size
    end_at = np.where(new_freqs >= cutoff[1])[0][0]
    mag_des = mag_spec.copy()
    spline_interp = splrep(freqs, mag_des, k=2)
    mag_interp = splev(new_freqs, spline_interp)

    # ignore frequencies below low cutoff and roll-off gently
    start_from = np.where(new_freqs >= cutoff[0])[0][0]
    mag_interp[:start_from] = np.flip(mag_interp[start_from] -
                                      db2lin(rolloff_dc_db) *
                                      np.arange(start_from))
    # also roll off gently above high cutoff
    max_freq = np.max(freqs)
    frac_decay_at_nyq = (fs / 2) / (2 * max_freq)
    mag_interp[
        end_at:] = mag_interp[end_at] - db2lin(rolloff_nyq_db) * np.linspace(
            0, frac_decay_at_nyq, n_bins_one_sided - end_at)
    if not return_one_sided:
        is_even = n_fft % 2 == 0
        mag_interp = one_to_two_sided_spectrum(mag_interp, is_even)
        new_freqs = fftfreq(n_fft, d=1 / fs)
    return mag_interp, new_freqs


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


def tf2minphase(tf: ArrayLike,
                axis: int = 0,
                is_even_fft: bool = True,
                is_time_domain: bool = False) -> ArrayLike:
    """Converts a transfer function to a minimum phase transfer function.

    Args:
        tf (NDArray): The original transfer function (single-sided spectrum, 0<=f<=fs/2)
        axis (int): The axis of H along which to calculate the minimum phase filters.
        is_even_fft (bool): Whether the two-sided conjugate-symmetric spectrum is even or odd in length,
            this affects the way we replicate the spectral coefficients.
            If not given (or None), we assume an even-length FFT was used,
            i.e. n_fft = 2 * (N - 1) where N is the number of frequency bins.
        is_time_domain (bool): is the result to be returned in time domain or as spectral coefficients

    Returns:
        NDArray: Spectral coefficients of the minimum phase filter (single-sided spectrum),
                 or the minimum phase impulse response
    """
    num_bins = tf.shape[axis]
    tf = one_to_two_sided_spectrum(tf, is_even_fft)
    mag = np.abs(tf)
    phu = np.imag(hilbert(-np.log(mag + np.finfo(np.float64).eps), axis=axis))
    ph = wrap_phase(phu)
    tf_mp: npt.NDArray
    tf_mp = mag * np.exp(1j * ph)
    # make one-sided spectrum again
    tf_mp = np.take(tf_mp, np.arange(num_bins), axis=axis)
    if not is_time_domain:
        return tf_mp
    else:
        return irfft(tf_mp)


def prony(h: np.ndarray, nb: int, na: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uses Prony's method to generate IIR filter coefficients
    that optimally match a given transfer function

    Args:
        h (np.ndarray): Numpy array containing the minimum phase response
        nb (int): Number of feedforward coefficients in the resulting filter
        na (int): Number of feedback coefficients in the resulting filter

    Returns:
        tuple: of arrays of feedforward coefficients and feedback coefficients
    """
    k = len(h) - 1
    H = np.asmatrix(toeplitz(np.array(h), np.append([1], np.zeros(k))))
    H = H[:, 0:(na + 1)]
    H1 = H[0:(nb + 1), :]
    h1 = H[(nb + 1):(k + 1), 0]
    H2 = H[(nb + 1):(k + 1), 1:(na + 1)]
    a = np.vstack((np.asmatrix(1), -H2.I * h1))
    aT = a.T
    b = aT * H1.T

    return b.getA()[0], aT.getA()[0]


def allpass_warp(ir: np.ndarray, rho: float) -> np.ndarray:
    """Performs allpass warping on a transfer function

    Args:
        rho (float): Amount of warping to perform. On the range (-1, 1). Positive warping
            "expands" the spectrum, negative warping "shrinks"
        ir (np.ndarray): The impulse response to warp

    Returns:
        np.ndarray: The warped impulse response
    """

    # using JAbel's implementation, not jatinchowdhury's
    nsamp = len(ir)  # length of impulse response
    nbinsmax = 65536  # maximum fft half bandwidth, bins

    # form linear and warped frequency axes
    stretch = (1 + np.abs(rho)) / (1 - np.abs(rho))
    nbins = min(
        nbinsmax,
        np.power(2, int(np.ceil((np.log(nsamp * stretch) / np.log(2))))))
    w = np.pi * np.arange(nbins) / nbins
    z = np.exp(1j * w)
    zeta = np.divide((z - rho), (1 - rho * z))
    ww = np.angle(zeta)

    # form, interpolate transfer function
    temp = fft(ir, 2 * nbins)
    tf = temp[:nbins]

    interpf = interp1d(w, tf, kind="cubic", fill_value="extrapolate")
    var = interpf(ww)
    tfw = np.r_[var, np.conj(np.flip(var[1:nbins]))]

    # compute warped impulse response
    temp = np.real(ifft(tfw, 2 * nbins))
    irw = temp[:nsamp]

    return irw


def allpass_warp_roots(rho: float, b: np.ndarray,
                       a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs allpass warping on a set of filter coefficients

    Args:
        rho (float): Amount of warping to perform. On the range (-1, 1).
            Positive warping "expands" the spectrum, negative warping "shrinks"
        b (np.ndarray): The filter coefficients (numerator)
        a (np.ndarray): The filter coefficients (denominator)

    Returns:
        tuple: array containing the warped filter coefficients
            and the warped filter coefficients
    """
    # find poles and zeros from filter coefficients

    # using JAbel's implementation, not jatinchowdhury's
    [z, p, k] = tf2zpk(b, a)
    # warp zeros, poles
    zw = np.divide((z + rho), (1 + rho * z))
    pw = np.divide((p + rho), (1 + rho * p))
    npz = len(p) - len(z)
    if npz > 0:
        zw = np.r_[zw, np.ones(npz) * rho]
    elif npz < 0:
        pw = np.r_[pw, np.ones(-npz) * rho]

    kw = k * np.prod(1 + z * rho) / np.prod(1 + p * rho)

    [bw, aw] = zpk2tf(zw, pw, kw)
    bw = np.real(bw)
    aw = np.real(aw)

    return bw, aw


def prony_warped(h: np.ndarray, nb: int, na: int,
                 rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uses Prony's method with frequency warping to generate IIR filter coefficients
    that optimally match a given transfer function

    Args:
        h (np.ndarray): Numpy array containing the minimum phase response
        nb (int): Number of feedforward coefficients in the resulting filter
        na (int): Number of feedback coefficients in the resulting filter
        rho (float): Amount of warping to perform. On the range (-1, 1).
            Positive warping "expands" the spectrum, negative warping "shrinks"

    Returns:
        tuple: of arrays of feedforward coefficients and feedback coefficients
    """
    h_warp = allpass_warp(h, rho)
    b_warped, a_warped = prony(h_warp, nb, na)
    b, a = allpass_warp_roots(-rho, b_warped, a_warped)
    return b, a
