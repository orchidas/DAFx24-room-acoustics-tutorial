import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike, NDArray
from .feedback_matrix import FeedbackMatrix, FilterMatrix, FeedbackMatrixType
from .delay_line import DelayLine
from .absorption_filter import Absorption, AbsorptionGains, AbsorptionFilters


class FDN:

    def __init__(self,
                 sample_rate: float,
                 num_inputs: int,
                 num_outputs: int,
                 num_delay_lines: int,
                 frame_size: int,
                 seed: Optional[int] = None):
        """Feedback delay network in the time domain"""
        self.num_delay_lines = num_delay_lines
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.postproc_filters = None
        self.seed = seed

    def init_io_gains(self, b: NDArray, c: NDArray):
        assert b.shape == (self.num_delay_lines, self.num_inputs)
        assert c.shape == (self.num_outputs, self.num_delay_lines)

        self.input_gains = b
        self.output_gains = c

    def init_direct_gain(self, d: NDArray):
        assert d.shape == (self.num_outputs, self.num_inputs)
        self.direct_gain = d

    def init_delay_line_lengths(self, delay_length_samps: ArrayLike):
        assert len(delay_length_samps) == self.num_delay_lines
        self.delay_length_samps = delay_length_samps

    def init_absorption_gains(self,
                              des_t60_ms: Union[float, ArrayLike],
                              t60_freqs: Optional[ArrayLike] = None,
                              filter_order: Optional[int] = None,
                              warp_factor: Optional[float] = None):
        """
        Initialise absorption gains / filters in FDN. T60 can be homogenous
        or frequency dependent. If the latter, then frequencies and IIR filter 
        order must be specified
        Args:
            des_t60_ms: broadband or frequency dependent T60 in ms
            t60_freqs (optional, ArrayLike): frequencies at which T60 is specified
            filter_order (optional, int): IIR filter order
            warp_factor (optional, float): warping coefficient for Prony's 
                                           method of IIR fitting
        """

        absorption_parent = Absorption(
            self.num_delay_lines,
            self.delay_length_samps + self.feedback.filter_order // 2,
            self.sample_rate, self.frame_size, des_t60_ms)

        if not isinstance(des_t60_ms, np.ndarray):
            self.absorption = absorption_parent.create_gain_child_instance(
                AbsorptionGains)
        else:
            self.absorption = absorption_parent.create_filter_child_instance(
                AbsorptionFilters, t60_freqs, filter_order, warp_factor)

    def init_feedback_matrix(
            self,
            feedback_matrix_type: Optional[
                FeedbackMatrixType] = FeedbackMatrixType.SCALAR_RANDOM,
            sparsity: Optional[float] = None,
            num_mixing_stages: Optional[int] = None):
        self.feedback = FeedbackMatrix(self.num_delay_lines,
                                       feedback_matrix_type, self.frame_size,
                                       sparsity, num_mixing_stages, self.seed)

    def init_delay_lines(self):
        self.delay_lines = []
        for i in range(self.num_delay_lines):
            # extra delay introduced by filter feedback matrices need to be accounted for in the gains
            self.delay_lines.append(
                DelayLine(self.delay_length_samps[i], self.frame_size))

    def init_postprocessing_filters(self, filters: NDArray):
        """
        Post processing filter matrix that adapts the FDN output
        Args: filters of size num_out x num_out x order
        """
        self.postproc_filters = FilterMatrix(self.num_outputs,
                                             self.num_outputs, self.frame_size)
        self.postproc_filters.initialise_matrix(filters)

    def process(self, input_data: NDArray) -> NDArray:
        """Process an incoming signal frame wise through the FDN and return the output"""

        assert input_data.shape[0] == self.num_inputs
        ir_len = input_data.shape[-1]

        if self.postproc_filters is None:
            output_data = np.zeros((self.num_outputs, ir_len))
        else:
            output_data = np.zeros(
                (self.num_outputs,
                 ir_len + self.postproc_filters.filter_matrix.shape[-1] - 1))

        self.feedback_output = np.zeros(
            (self.num_delay_lines,
             self.frame_size + self.feedback.filter_order - 1))

        delay_line_output = np.zeros((self.num_delay_lines, self.frame_size))

        sample_number = 0
        while sample_number < ir_len - self.frame_size:
            cur_idx_range = np.arange(sample_number,
                                      sample_number + self.frame_size)
            cur_input_buffer = input_data[:, cur_idx_range]

            # multiply with input gains
            temp_data = self.input_gains @ cur_input_buffer
            # add output from feedback matrix
            delay_line_input = temp_data + self.feedback_output[:, :self.
                                                                frame_size]
            if self.feedback.filter_order > 1:
                delay_line_input = np.hstack(
                    (delay_line_input, self.feedback_output[:,
                                                            self.frame_size:]))
            # update delay lines
            for i in range(self.num_delay_lines):
                # read from delay lines
                delay_line_output[i, :] = self.delay_lines[i].readBuffer()

                # push into delay lines
                self.delay_lines[i].writeBuffer(delay_line_input[i, :])

            # add absorption filtering
            delay_line_output = self.absorption.process(delay_line_output)

            # multiply output of delay line with feedback matrix
            self.feedback_output = self.feedback.process(delay_line_output)

            # multiply delay line output with output gains
            cur_output_buffer = self.output_gains @ delay_line_output

            # carry out post-processing filtering
            if self.postproc_filters is not None:
                cur_output_buffer = self.postproc_filters.process_input(
                    cur_output_buffer)
                # latter half of cur_output_buffer goes to output directly
                # without modification
                output_data[:, cur_idx_range[-1]:cur_idx_range[-1] +
                            self.postproc_filters.filter_order -
                            1] += cur_output_buffer[:, self.frame_size:]

            # add to direct gain
            output_data[:,
                        cur_idx_range] += cur_output_buffer[:, :self.
                                                            frame_size] + self.direct_gain @ cur_input_buffer
            sample_number += self.frame_size

        return output_data
