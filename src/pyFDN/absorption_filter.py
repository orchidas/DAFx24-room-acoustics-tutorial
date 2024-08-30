import numpy as np
from numpy.typing import NDArray, ArrayLike
from .utils import db2lin


class AbsorptionGains:

    def __init__(self, num_delay_lines: int, delay_line_lengths: ArrayLike,
                 sample_rate: float, des_t60_ms: float, frame_size: int):
        """
		Absorption filters in feedback delay networks
		Args:
			num_delay_lines (int): number of delay lines in FDN
			delay_line_lengths (ArrayLike): delay line lengths in samples
			sample_rate (float): sample rate of FDN
			des_t60_ms (float): desired T60 of FDN in ms. 
							 Only homogeneous decay is supported
			frame_size (int): buffer size in each time step
		"""

        self.num_delay_lines = num_delay_lines
        self.delay_line_lengths = delay_line_lengths.copy()
        self.des_t60_ms = des_t60_ms
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.calculate_delay_line_gains()

    def calculate_delay_line_gains(self):
        self.absorption_gains = db2lin(
            (-60 * self.delay_line_lengths) /
            (self.sample_rate * self.des_t60_ms * 1e-3))

    def process(self, input_data: NDArray) -> ArrayLike:
        assert input_data.shape == self.num_delay_lines, self.frame_size
        return input_data * np.repeat(
            self.absorption_gains[:, np.newaxis], self.frame_size, axis=-1)