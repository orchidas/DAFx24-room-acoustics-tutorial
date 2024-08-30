import numpy as np
from typing import Optional
from numpy.typing import ArrayLike, NDArray
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .delay_line import DelayLine
from .absorption_filter import AbsorptionGains


class FDN:

    def __init__(self, sample_rate: float, num_inputs: int, num_outputs: int,
                 num_delay_lines: int, frame_size: int):
        """Feedback delay network in the time domain"""
        self.num_delay_lines = num_delay_lines
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

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

    def init_absorption_gains(self, des_t60_ms: float):
        self.absorption = AbsorptionGains(
            self.num_delay_lines,
            self.delay_length_samps + self.feedback.filter_order // 2,
            self.sample_rate, des_t60_ms, self.frame_size)

    def init_feedback_matrix(
            self,
            feedback_matrix_type: Optional[
                FeedbackMatrixType] = FeedbackMatrixType.SCALAR_RANDOM,
            sparsity: Optional[float] = None,
            num_mixing_stages: Optional[int] = None):
        self.feedback = FeedbackMatrix(self.num_delay_lines,
                                       feedback_matrix_type, self.frame_size,
                                       sparsity, num_mixing_stages)

    def init_delay_lines(self):
        self.delay_lines = []
        for i in range(self.num_delay_lines):
            # extra delay introduced by filter feedback matrices need to be accounted for in the gains
            self.delay_lines.append(
                DelayLine(self.delay_length_samps[i], self.frame_size))

    def process(self, input_data: NDArray) -> NDArray:
        """Process an incoming signal frame wise through the FDN and return the output"""

        assert input_data.shape[0] == self.num_inputs
        ir_len = input_data.shape[-1]
        output_data = np.zeros((self.num_outputs, ir_len))
        self.feedback_output = np.zeros(
            (self.num_delay_lines,
             self.frame_size + self.feedback.filter_order - 1))

        delay_line_output = np.zeros((self.num_delay_lines, self.frame_size))

        sample_number = 0
        while sample_number < ir_len - self.frame_size:
            cur_input_buffer = input_data[:, sample_number:sample_number +
                                          self.frame_size]

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
                # push into delay lines
                self.delay_lines[i].writeBuffer(delay_line_input[i, :])
                # read from delay lines and adapt absorption
                delay_line_output[i, :] = self.delay_lines[i].readBuffer(
                ) * self.absorption.absorption_gains[i]

            # multipily output of delay line with feedback matrix
            self.feedback_output = self.feedback.process(delay_line_output)

            # multiply delay line output with output gains
            cur_output_buffer = self.output_gains @ delay_line_output
            output_data[:, sample_number:sample_number + self.
                        frame_size] = cur_output_buffer + self.direct_gain @ cur_input_buffer

            sample_number += self.frame_size

        return output_data
