import numpy as np
import sofar
from loguru import logger
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Tuple, Union
from pathlib import Path
from scipy.fft import rfft, irfft, rfftfreq


def cart2sph(x: Union[NDArray, float],
             y: Union[NDArray, float],
             z: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    x, y, z = np.broadcast_arrays(x, y, z)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if degrees:
        azimuth = np.degrees(azimuth)
        elevation = np.degrees(elevation)

    return np.stack((azimuth, elevation, r), axis=axis)


def sph2cart(azimuth: Union[NDArray, float],
             elevation: Union[NDArray, float],
             r: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    azimuth, elevation, r = np.broadcast_arrays(azimuth, elevation, r)
    if degrees:
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.stack((x, y, z), axis=axis)


def unpack_coordinates(
        coord_matrix: NDArray,
        axis: int = -1) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Unpack a coordinate matrix into separate arrays each containing
    one of 3 dimensions
    """
    if axis != -1:
        coord_matrix = coord_matrix.T

    dim1 = coord_matrix[:, 0]
    dim2 = coord_matrix[:, 1]
    dim3 = coord_matrix[:, 2]

    return dim1, dim2, dim3


class HRIRSOFAReader:

    def __init__(self, sofa_path: Path):
        try:
            self.sofa_reader = sofar.read_sofa(str(sofa_path),
                                               verify=True,
                                               verbose=True)
        except:
            raise FileNotFoundError(
                "SOFA file not found in specified location!")

        self.fs = self.sofa_reader.Data_SamplingRate
        self.ir_data = self.sofa_reader.Data_IR
        self.num_dims = self.ir_data.ndim
        logger.info(self.sofa_reader.list_dimensions)
        logger.info(f'Shape of the IRs is {self.ir_data.shape}')

        if self.num_dims > 3:
            self.num_meas, self.num_emitter, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape
        else:
            self.num_meas, self.num_receivers, self.ir_length = self.sofa_reader.Data_IR.shape

        self.listener_view = self.get_listener_view(coord_type="spherical")

    def get_listener_view(self, coord_type: str = "cartesian") -> NDArray:
        """Get the listener view array. An array of vectors corresponding to the view direction of the listener.
        This can be in spherical or cartesian coordinates.

        Args:
            coord_type (str): Required coordinate system, by default "cartesian".
                Options: cartesian or spherical

        Returns:
            NDArray:
                Array of listener view vectors in HuRRAh coordinate convention. Dims: [M/I, C]
                M is number of measurements. I is 1. C is 3 (for 3D coordinates).
                Spherical coordinates have angles in degrees.

        Raises:
            ValueError: If the given coord_type is not one of the supported options.
            ValueError: If the SOFA listener view is not in degree,
                                      degree, metre units
        """
        coord_type = coord_type.lower()
        is_source_cart = self.sofa_reader.ListenerView_Type.lower(
        ) == "cartesian"

        if is_source_cart:
            list_view_cart = self.sofa_reader.ListenerView
        else:
            # check that we've got angles in degrees
            if self.file.ListenerView_Units != "degree, degree, metre":
                raise ValueError(
                    f"Incompatible units for type of ListenerView in SOFA file. "
                    f"Type: {self.sofa_reader.ListenerView_Type}, Units: {self.sofa_reader.ListenerView_Units} "
                    "Should be: degree, degree, metre")
            list_view_sph = self.sofa_reader.ListenerView
            # if radius is set to zero in file, set to 1
            list_view_sph[list_view_sph[:, 2] == 0.0, 2] = 1.0
            az, el, r = unpack_coordinates(list_view_sph, axis=-1)
            list_view_cart = sph2cart(az, el, r, axis=-1, degrees=True)

        # now convert to spherical if needed
        if coord_type == "cartesian":
            return list_view_cart
        else:
            x, y, z = unpack_coordinates(list_view_cart, axis=-1)
            return cart2sph(x, y, z, axis=-1, degrees=True)

    def get_ir_corresponding_to_listener_view(self,
                                              des_listener_view: NDArray,
                                              axis: int = -1,
                                              coord_type: str = "spherical",
                                              degrees: bool = True) -> NDArray:
        """
        Get IR corresponding to a particular listener view
        Args:
            des_listener_view: P x 3 array of desired listener views
            axis (int): axis of coordinates
            coord_type (str): coordinate system type when specifying listener view
            degrees (bool): whether the listener view is specified in degrees
        Returns:
            P x E x R x N IR corresponding to the particular listener views 
        """

        if axis != -1:
            des_listener_view = des_listener_view.T

        num_views = des_listener_view.shape[0]
        assert num_views < self.num_meas

        # euclidean distance between desired and available views
        dist = np.zeros((self.num_meas, num_views))
        des_ir_matrix = np.zeros((num_views, self.ir_data.shape[1:]),
                                 dtype=float)

        if coord_type == "spherical":
            az, el, r = unpack_coordinates(des_listener_view.copy(), axis=axis)
            des_listener_view = sph2cart(az, el, r, axis=axis, degrees=degrees)

        # find index of view that minimuses the error from the desied view
        for k in range(num_views):
            dist[:, k] = np.sqrt(
                np.sum((self.listener_view - self.des_listener_view[k, :])**2,
                       axis=axis))
            closest_idx = np.argmin(dist[:, k])
            des_ir_matrix[k, ...] = self.ir_data[closest_idx, ...]

        return des_ir_matrix

    def get_all_irs_corresponding_to_receiver(self,
                                              receiver_idx: int) -> NDArray:
        """
        Returns all HRIRs corresponding to a particular receiver. (0, 1) for binaural
        data
        Args:
            received_idx (int) : index of the receiver
        Returns:
            NDArray: array of size M x E x N containing the IRs
        """
        assert receiver_idx < self.num_receivers
        return self.sofa_reader.Data_IR[:, receiver_idx, ...]

    def get_interaural_coherence(self, domain: str = "frequency") -> ArrayLike:
        """
        Get interaural coherence by averaging the HRIRs specified
        Args:
            domain (str): whether to return the IAC in time, or frequency domain
        Returns:
            N x 1 IAC in the time or frequency domain
        """
        # all HRIRs corresponding to left and right ears
        left_hrirs = self.get_all_irs_corresponding_to_receiver(0)
        right_hrirs = self.get_all_irs_corresponding_to_receiver(1)

        # take FFT of HRIRs to get HRTFs
        num_fft_bins = self.ir_length
        left_hrtfs = rfft(left_hrirs, num_fft_bins, axis=-1)
        right_hrtfs = rfft(right_hrirs, num_fft_bins, axis=-1)

        left_norm_factor = np.sum(np.abs(left_hrtfs)**2, axis=0)
        right_norm_factor = np.sum(np.abs(right_hrtfs)**2, axis=0)

        numerator = np.abs(np.sum(left_hrtfs * np.conj(right_hrtfs), axis=0))
        denominator = np.sqrt(left_norm_factor * right_norm_factor)

        iac = numerator / denominator

        if domain == "frequency":
            return iac
        else:
            return irfft(iac, self.ir_length)
