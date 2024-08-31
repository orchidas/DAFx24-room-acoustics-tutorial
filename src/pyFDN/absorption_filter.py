import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray, ArrayLike
from typing import Optional, Union
from abc import ABC
from scipy.signal import freqz, lfilter
from scipy.fft import rfftfreq
from .utils import db, db2lin
from .filter_fitting import tf2minphase, interpolate_magnitude_spectrum, prony_warped


class Absorption(ABC):
    """Abstract base class for absorption filters / gains in the FDN"""

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: ArrayLike,
        sample_rate: float,
        frame_size: int,
        des_t60_ms: Union[float, ArrayLike],
    ):
        """
        Absorption filters in feedback delay networks
        Args:
            num_delay_lines (int): number of delay lines in FDN
            delay_line_lengths (ArrayLike): delay line lengths in samples
            sample_rate (float): sample rate of FDN
            frame_size (int): buffer size in each time step
            des_t60_ms (float, ArrayLike): desired T60 of FDN in ms. Can be frequency dependent 
        """
        self.num_delay_lines = num_delay_lines
        self.delay_line_lengths = delay_line_lengths.copy()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.des_t60_ms = des_t60_ms

    def create_gain_child_instance(self, abs_gain_instance):
        return abs_gain_instance(self.num_delay_lines, self.delay_line_lengths,
                                 self.sample_rate, self.frame_size,
                                 self.des_t60_ms)

    def create_filter_child_instance(self, abs_filter_instance,
                                     freqs: ArrayLike, filter_order: int,
                                     warp_factor: float):
        return abs_filter_instance(self.num_delay_lines,
                                   self.delay_line_lengths, self.sample_rate,
                                   self.frame_size, self.des_t60_ms, freqs,
                                   filter_order, warp_factor)

    def process(self, input_data: NDArray) -> NDArray:
        """Method implemented in child class"""
        pass


class AbsorptionGains(Absorption):

    def __init__(self, num_delay_lines: int, delay_line_lengths: ArrayLike,
                 sample_rate: float, frame_size: int, des_t60_ms: float):
        """
        Absorption gains in feedback delay networks with homogenous decay
        Args:
            num_delay_lines (int): number of delay lines in FDN
            delay_line_lengths (ArrayLike): delay line lengths in samples
            sample_rate (float): sample rate of FDN
            frame_size (int): buffer size in each time step
            des_t60_ms (float): desired T60 of FDN in ms. 
                             Only homogeneous decay is supported
            freqs (optiona, list): this is just a placeholder, and is not
                            used by this child class
        """

        super().__init__(num_delay_lines, delay_line_lengths, sample_rate,
                         frame_size, des_t60_ms)
        self.calculate_delay_line_gains()

    def calculate_delay_line_gains(self):
        self.absorption_gains = db2lin(
            (-60 * self.delay_line_lengths) /
            (self.sample_rate * self.des_t60_ms * 1e-3))

    def process(self, input_data: NDArray) -> NDArray:
        assert input_data.shape == (self.num_delay_lines, self.frame_size)
        return input_data * np.repeat(
            self.absorption_gains[:, np.newaxis], self.frame_size, axis=-1)


class AbsorptionFilters(Absorption):

    def __init__(self, num_delay_lines: int, delay_line_lengths: ArrayLike,
                 sample_rate: float, frame_size: int, des_t60_ms: ArrayLike,
                 freqs: ArrayLike, filter_order: int, warp_factor: float):
        """
        Absorption gains in feedback delay networks with frequency dependent decay
        Args:
            num_delay_lines (int): number of delay lines in FDN
            delay_line_lengths (ArrayLike): delay line lengths in samples
            sample_rate (float): sample rate of FDN
            frame_size (int): buffer size in each time step
            des_t60_ms (ArrayLike): desired T60 of FDN in ms. 
                             Only frequency-dependent decay is supported
            freqs (ArrayLike): frequencies where T60s are specified
            filter_order (int): order of the IIR filter to be fit
            warp_factor( float): warping coefficient between (-1,1). Negative coefficient
                                 squeezes the spectrum, positive coefficient expands it.
        """
        super().__init__(num_delay_lines, delay_line_lengths, sample_rate,
                         frame_size, des_t60_ms)
        assert len(des_t60_ms) == len(
            freqs
        ), "T60 and the frequencies at which it is specified must be the same"

        self.freqs = freqs
        self.filter_order = filter_order
        self.rho = warp_factor
        self.fit_t60_filters()

    def fit_t60_filters(self, num_freq_bins: int = 2**12):
        """
        Use frequency-warped Prony's method to fit an IIR filter to get the desired T60
        """
        # the T60s for each delay line need to be attenuated
        self.num_coeffs = np.zeros(
            (self.num_delay_lines, self.filter_order + 1))
        self.den_coeffs = np.zeros_like(self.num_coeffs)
        self.delay_line_filters = np.zeros(
            (self.num_delay_lines, len(self.freqs)))
        self.interp_delay_line_filter = np.zeros(
            (self.num_delay_lines, num_freq_bins // 2 + 1))

        for i in range(self.num_delay_lines):
            self.delay_line_filters[i, :] = db2lin(
                (-60 * (self.delay_line_lengths[i] + self.filter_order)) /
                (self.sample_rate * self.des_t60_ms * 1e-3))

            self.interp_delay_line_filter[
                i, :], _ = interpolate_magnitude_spectrum(
                    self.delay_line_filters[i, :],
                    self.freqs,
                    self.sample_rate,
                    n_fft=num_freq_bins,
                    cutoff=(100, 16e3),
                    rolloff_dc_db=-60,
                    return_one_sided=True)

            interp_min_phase_ir = tf2minphase(
                self.interp_delay_line_filter[i, :],
                axis=0,
                is_even_fft=True,
                is_time_domain=True)
            self.num_coeffs[i, :], self.den_coeffs[i, :] = prony_warped(
                interp_min_phase_ir, self.filter_order, self.filter_order,
                self.rho)

    def process(self, input_data: NDArray) -> NDArray:
        """
        Filter the incoming data with the absorotion filters
        """
        assert input_data.shape == (self.num_delay_lines, self.frame_size)
        output_data = np.zeros_like(input_data)
        for i in range(self.num_delay_lines):
            output_data[i, :] = lfilter(self.num_coeffs[i, :],
                                        self.den_coeffs[i, :],
                                        input_data[i, :])
        return output_data

    def plot_t60_filter_response(self, num_freq_bins: int = 2**12):
        """
        Plot the fitted T60 filters to see how well they match to the specified spectrum
        """
        total_response = np.zeros((self.num_delay_lines, num_freq_bins),
                                  dtype=complex)
        freq_axis_one_sided = rfftfreq(num_freq_bins, d=1.0 / self.sample_rate)

        for k in range(self.num_delay_lines):
            freq_axis, total_response[k, :] = freqz(self.num_coeffs[k, :],
                                                    self.den_coeffs[k, :],
                                                    worN=num_freq_bins,
                                                    fs=self.sample_rate)

        plt.figure()
        line0 = plt.semilogx(self.freqs,
                             db(self.delay_line_filters.T),
                             marker="o")
        line1 = plt.semilogx(freq_axis_one_sided,
                             db(np.abs(self.interp_delay_line_filter.T)),
                             linestyle='--')
        line2 = plt.semilogx(freq_axis, db(np.abs(total_response.T)))
        plt.legend([line0[0], line1[0], line2[0]],
                   ["Measured", "Interpolated", "Warped prony fit"])
        plt.xlabel('Frequency(Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.tight_layout()
