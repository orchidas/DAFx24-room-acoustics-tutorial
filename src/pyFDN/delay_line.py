import numpy as np
from numpy.typing import ArrayLike


class DelayLine:

    def __init__(self, length: int, frame_size: int):
        """
        Class implementing a delay line of 'length' samples. Pushes and pops a buffer
        at a time.
        Args:
            length (int) : length of delay line in samples
            frameSize (int): buffer size in samples 

        """
        # integer delay line length
        self.length = length
        self.data = np.zeros(self.length, dtype=float)
        self.frame_size = frame_size

        if (self.length < self.frame_size):
            print("Length of delay line is ", self.length)
            raise Exception("Try smaller buffer size")

    def readBuffer(self) -> ArrayLike:
        """Read the delay line buffer"""

        # extract output buffer
        output_data = self.data[-self.frame_size:]

        # return output buffer
        # the flip is needed so that oldest samples go out first
        # this is basically a FIFO queue
        return np.flip(output_data)

    def writeBuffer(self, input_data: ArrayLike):
        """Write new data into the delay line buffer"""

        # if (input_data.size != self.frame_size):
        #     print(" Input data size is ", input_data.size,
        #           ' but frame size is ', self.frame_size)
        #     raise Exception("Buffer Sizes do not match")

        # shift data to the right
        self.data = np.roll(self.data, len(input_data))

        # add buffer to start of delayLine
        # the flip is needed so that latest samples go in first
        # this is basically a FIFO queue
        self.data[:len(input_data)] = np.flip(input_data)
