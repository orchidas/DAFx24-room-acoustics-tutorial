import numpy as np
from scipy.linalg import qr, hadamard
from typing import Optional
from numpy.typing import NDArray, ArrayLike
from enum import Enum
from loguru import logger
from scipy.signal import fftconvolve


class FeedbackMatrixType(str, Enum):
    """
    The interpolation method to use for HRIRs.

    Options:
        - scalar_random - random orthonormal matrix
        - scalar_hadamard - hadamard matrix
        - filter_velvet - velvet feedback matrix
    """
    SCALAR_RANDOM = 'scalar_random'
    SCALAR_HADAMARD = 'scalar_hadamard'
    FILTER_VELVET = 'filter_velvet'

    def __repr__(self) -> str:
        return str(self.value)


class FilterMatrix:

    def __init__(self, num_rows: int, num_cols: int, frame_size: int):
        """
        Generic filter matrix class
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.frame_size = frame_size

    def initialise_matrix(self, matrix: NDArray):
        """Copy a 2D or 3D matrix to the filter matrix"""
        self.filter_matrix = matrix.copy()

        if matrix.ndim == 2:
            self.filter_order = 1
        else:
            self.filter_order = matrix.shape[-1]

    def process_input(self, input_data: NDArray):
        """
        Process a buffer of data through the filter matrix
        Args:
            input_data (NDArray): num_delay_lines x frame_size real array at matrix input
        Returns:
            NDArray, NDArray: output matrix of size 
                            num_delay_lines x frame_size + filter_order - 1,
        """

        assert input_data.shape == (self.num_rows, self.frame_size)
        input_data_copy = input_data.copy()

        if self.filter_order == 1:
            return self.filter_matrix @ input_data
        else:
            temp_matrix = np.zeros(
                (self.num_rows, self.filter_order + self.frame_size - 1))
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    temp_matrix[i, :] += fftconvolve(
                        self.filter_matrix[i, j, :], input_data_copy[j, :])

            output_data = temp_matrix.copy()

            return output_data


class FeedbackMatrix(FilterMatrix):

    def __init__(self,
                 num_rows: int,
                 feedback_matrix_type: FeedbackMatrixType,
                 frame_size: int,
                 sparsity: Optional[float] = None,
                 num_mixing_stages: Optional[int] = None,
                 seed: Optional[int] = None):
        """
        Feedback matrix of a feedback delay network. Can be scalar or filter
        Args:
            num_rows (int): size of matrix
            feedback_matrix_type: type of the feedback matrix (scalar, or filter)
            frame_size (int): buffer size
            sparsity (float) : a number between 0 and 1, denoting the sparsity of the VFM
            num_mixing_stages (int): number of mixing stages in constructing the VFM
            seed (int): seed for random number generation
        """

        super().__init__(num_rows, num_rows, frame_size)
        self.filter_order = 1
        self.feedback_matrix_type = feedback_matrix_type

        if seed is not None:
            np.random.seed(seed)

        if feedback_matrix_type == FeedbackMatrixType.SCALAR_RANDOM:
            self.feedback_matrix = FeedbackMatrix.get_random_unitary_matrix(
                self.num_rows)

        elif feedback_matrix_type == FeedbackMatrixType.SCALAR_HADAMARD:
            if np.log2(self.num_rows).is_integer():
                self.feedback_matrix = FeedbackMatrix.get_hadamard_matrix(
                    self.num_rows)
            else:
                raise RuntimeError(
                    "Hadamard matrix size can only be a power of 2")

        elif feedback_matrix_type == FeedbackMatrixType.FILTER_VELVET:
            if sparsity is None or num_mixing_stages is None:
                raise RuntimeError(
                    "Sparsity and num_mixing_stages must be specified for VFM")
            else:
                (self.feedback_matrix,
                 self.filter_order) = self.populate_velvet_feedback_matrix(
                     sparsity, num_mixing_stages)
        else:
            raise NotImplementedError(
                "Other types of feedback matrices are not supported")

        # initialise the super class' filter matrix
        super().initialise_matrix(self.feedback_matrix)

    @staticmethod
    def matrix_convolution(A: NDArray, B: NDArray) -> NDArray:
        """Multiply 3D matrices with filter coefficients along the third dimension
        Inputs:
        A - matrix polynomial of size [m,n,order]
        B - matrix polynomial of size [n,k,order]

        Outputs:
          C - matrix polynomial of size [m,k,order] with C(z) = A(z)+B(z)
        """

        szA = np.shape(A)
        szB = np.shape(B)

        assert (szA[1] == szB[0]), "Matrix dimensions should match"

        # the last dimension stores the result of convolution
        C = np.zeros((szA[0], szB[1], szA[2] + szB[2] - 1))

        for row in range(szA[0]):
            for col in range(szB[1]):
                for it in range(szA[1]):
                    C[row, col, :] += np.convolve(A[row, it, :], B[it, col, :])

        return C

    @staticmethod
    def is_unitary(test_matrix: NDArray, tol=1e-9) -> bool:
        """Checks if the feedback matrix is unitary"""
        dim = test_matrix.shape[0]
        test_matrix_prod = np.matmul(test_matrix.T, test_matrix)
        test_matrix_prod -= np.eye(dim)
        is_unitary_flag = np.all(test_matrix_prod < tol)
        return is_unitary_flag

    @staticmethod
    def is_paraunitary(test_matrix: NDArray, tol=1e-6) -> bool:
        """
        Helper function to check if the feedback matrix
        is paraunitary
        """
        dim = test_matrix.shape[0]
        order = test_matrix.shape[2] - 1
        matrix_conjugate = np.transpose(np.flip(test_matrix, axis=2),
                                        (1, 0, 2))
        test_matrix_prod = FeedbackMatrix.matrix_convolution(
            test_matrix, matrix_conjugate)
        # should be close to zero
        test_matrix_prod[..., order] -= np.eye(dim)

        max_off_diag_val = np.amax(np.abs(test_matrix_prod))
        is_paraunitary_flag = max_off_diag_val < tol

        return is_paraunitary_flag

    @staticmethod
    def get_random_unitary_matrix(num_rows):
        # start with a random matrix and get its QR decomposition
        (Q, R) = qr(np.random.randn(num_rows, num_rows))
        return np.matmul(Q, np.diag(np.sign(np.diag(R))))

    @staticmethod
    def get_hadamard_matrix(num_rows):
        hadamard_matrix = hadamard(num_rows) / np.sqrt(num_rows)
        if not FeedbackMatrix.is_unitary(hadamard_matrix):
            raise ValueError("Hadamard matrix is not scaled properly!")
        return hadamard_matrix

    def get_delay_feedback_matrix(self, delays: ArrayLike) -> NDArray:
        """
        Delay feedback matrix has delays along the diagonals
        """
        max_delay = np.max(delays)
        delay_matrix = np.zeros((self.num_rows, self.num_rows, max_delay + 1))

        for k in range(self.num_rows):
            delay_matrix[k, k, delays[k]] = 1.0

        return delay_matrix

    def populate_velvet_feedback_matrix(self,
                                        sparsity: float = 1.0,
                                        num_mixing_stages: int = 1):
        """
        Populate polynomial feedback matrix recursively
        """
        if sparsity < 1.0:
            init_delays = np.floor(
                (np.arange(self.num_rows) +
                 0.99 * np.random.rand(self.num_rows)) / sparsity).astype(int)
        else:
            init_delays = np.arange(self.num_rows)

        cur_delay_matrix = self.get_delay_feedback_matrix(init_delays)
        if np.log2(self.num_rows).is_integer():
            cur_dense_matrix = FeedbackMatrix.get_hadamard_matrix(
                self.num_rows)[..., np.newaxis]
        else:
            cur_dense_matrix = FeedbackMatrix.self.get_random_unitary_matrix(
                self.num_rows)[..., np.newaxis]

        for k in range(num_mixing_stages):
            unitary_matrix = FeedbackMatrix.get_hadamard_matrix(
                self.num_rows) if np.log2(self.num_rows).is_integer(
                ) else FeedbackMatrix.get_random_unitary_matrix(self.num_rows)
            tmp = self.matrix_convolution(unitary_matrix[..., np.newaxis],
                                          cur_delay_matrix)
            next_dense_matrix = self.matrix_convolution(tmp, cur_dense_matrix)
            cur_delays = init_delays * (np.floor(
                self.num_rows / sparsity).astype(int)**k)
            cur_delay_matrix = self.get_delay_feedback_matrix(cur_delays)
            cur_dense_matrix = next_dense_matrix

        # ensure that the feedback matrix is energy preserving
        assert self.is_paraunitary(
            cur_dense_matrix), "Feedback matrix must be energy preserving!"
        logger.info(f'The order of the VFM is {cur_dense_matrix.shape[2]}')
        # cur_dense_matrix is a NxNxN^K feedback matrix
        return (cur_dense_matrix, cur_dense_matrix.shape[2])

    def process(self, input_data: NDArray) -> NDArray:
        """
        Process a buffer of data through the feedback matrix
        Args:
            input_data (NDArray): num_delay_lines x frame_size real array at matrix input
        Returns:
            NDArray, NDArray: output matrix of size 
                            num_delay_lines x frame_size + filter_order - 1,
        """

        return super().process_input(input_data)
