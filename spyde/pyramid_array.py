# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict, Callable, Any


def bin_ndarray(ndarray, factors:Tuple[int]):
    """
    Bins a ndarray in all axes based on a list of factors.es.
    """
    shape = np.array(ndarray.shape)

    new_shape = shape/factors
    for d in new_shape:
        if not d.is_integer():
            raise ValueError("Binning factors must divide the original shape evenly.")

    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [int(l) for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        ndarray = ndarray.sum(-1*(i+1))
    return ndarray


class BinnedArray:
    """
    A class to represent a binned array. This is a multidimensional array
    which has been re-binned along certain dimensions.
    """

    def __init__(self,
                 array: npt.ArrayLike,
                 binning_factors: Tuple[int, ...],
                 offsets: Tuple[int, ...] = None,
                 parent_array: 'PyramidArray' = None
                 ) -> None:
        self.array = array
        self.binning_factors = binning_factors
        self.parent_array = parent_array
        if offsets is None:
            self.offsets = tuple(0 for _ in range(len(binning_factors)))
        else:
            self.offsets = offsets

    def __array__(self):
        return self.array

    def sum(self, axis=None):
        return self.array.sum(axis=axis)

    def __getitem__(self, item):
        """Get an item from the array.

        9 --> 9-6(offset) --> ceil(3/ 10(bin_factor)) --> 1.0
        new_offset = bf - offset % bf
        """
        new_offsets = []
        new_binning_factors = []
        if isinstance(item, tuple):
            items = []
            if Ellipsis in item:
                el_ind = item.index(Ellipsis)
                before_ellipsis = item[:el_ind]
                after_ellipsis = item[el_ind + 1:]
                item = [slice(None) for i in range(self.array.dim)]
                item[:len(before_ellipsis)] = before_ellipsis
                item[-len(after_ellipsis):] = after_ellipsis
                item = tuple(item)
            for s, bf, offset in zip(item,
                                     self.binning_factors,
                                     self.offsets):
                if isinstance(s, slice):
                    start = s.start or 0
                    start, off = np.divmod((start - offset), bf)
                    new_offsets.append(off)

                    stop = s.stop or self.array.shape[item.index(s)] * bf + offset
                    stop = np.floor((stop - offset) / bf).astype(int)
                    sl2 = slice(start, stop)
                    items.append(sl2)
                    new_binning_factors.append(bf)
                elif isinstance(s, int):
                    if bf != 1:
                        raise NotImplementedError("Integer indexing is not supported for binned arrays"
                                                  " and dimension greater than 1.")
                    items.append(s)
                elif isinstance(s, np.ndarray):
                    raise NotImplementedError("Array indexing is not supported for binned arrays.")
            new_array = self.array[tuple(items)]
        elif isinstance(item, int):
            if len(self.binning_factors) != 1:
                raise NotImplementedError("Integer indexing is not supported for binned arrays "
                                          "with more than one dimension.")
            new_array = self.array[item]
            new_offsets = self.offsets[1:]
            new_binning_factors = self.binning_factors[1:]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or self.array.shape[0] * self.binning_factors[0] + self.offsets[0]
            start, off_start = np.divmod((start - self.offsets[0]), self.binning_factors[0])
            stop = np.floor((stop - self.offsets[0]) / self.binning_factors[0]).astype(int)
            new_array = self.array[start:stop]
            new_offsets = [off_start,] + list(self.offsets[1:])
            new_binning_factors = self.binning_factors
        elif isinstance(item, np.ndarray):
            raise NotImplementedError("Array indexing is not supported for binned arrays.")
        else:
            raise TypeError("Unsupported index type for binned array.")

        return BinnedArray(new_array,
                           tuple(new_binning_factors),
                           tuple(new_offsets))


class PyramidArray:
    """A class to represent a pyramid of arrays.  This is a multidimensional array
    which has been re-binned along certain dimensions.

    Parameters
    ----------
    array : npt.ArrayLike
        The input array to be converted into a pyramid.
    pyramid : Dict, optional
        A dictionary representing the pyramid structure. If None, a default pyramid structure is created.
    pyramid_levels : List[Tuple], optional
        A list of tuples representing the pyramid levels. Each tuple contains the dimensions of the pyramid level.
        If None, default levels are used.


    Examples
    --------

    >>> summed_pa = pa[:10, 10:20].sum()
    >>> summed_pa = pa[mask].sum()
    >>>
    """

    def __init__(self,
                 array: npt.NDArray,
                 binned_arrays: List[BinnedArray] = None,
                 ) -> None:

        self.array = array
        self.binned_arrays = binned_arrays if binned_arrays is not None else []

    @property
    def bin_factors(self) -> np.ndarray:
        """
        Return the binning factors for all binned arrays.
        """
        return np.array([binned_array.binning_factors for binned_array in self.binned_arrays])

    @property
    def bin_offsets(self)-> np.ndarray:
        return np.array([binned_array.offsets for binned_array in self.binned_arrays])

    def __array__(self):
        return self.array

    @property
    def shape(self):
        return self.array.shape

    def add_pyramid_level(self, binning_factors: Tuple[int]):
        """Create a pyramid of arrays. This is a placeholder for the actual implementation."""
        shape =self.shape
        if len(binning_factors) != len(shape):
            raise ValueError("Binning factors must match the number of dimensions in the array.")

        # we could also deal with factors which are not integer divisors of the shape,
        # but for now we will assume they are and that the offsets are zero.
        offsets = tuple(0 for _ in range(len(shape)))

        bin_array = bin_ndarray(self.array, binning_factors)
        self.binned_arrays.append(BinnedArray(bin_array, binning_factors, offsets))
        return

    def __getitem__(self, item):
        """Get an item from the array. """
        new_array = self.array[item]
        binned_arrays = [binned_array[item] for binned_array in self.binned_arrays]
        return PyramidArray(new_array,
                            binned_arrays)

    def sum(self, axis=(2, 3)):
        """Sum the array along all dimensions.

        To do this we can just take the pyramid with the maximum along each specified axis.
        """
        if isinstance(axis, int):
            axis = (axis,)

        # all other bin factors must be 1??
        other_axes = tuple(set(np.arange(self.array.ndim)) - set(axis))

        use_axis = np.all((self.bin_factors == 1)[:, (0, 1)], axis=1)
        factors = self.bin_factors.copy()
        factors[~use_axis, :] = 1

        # we want to find the maximum product to make sure we are summing over the binned array with
        # the largest chunks.
        factors = np.prod(factors, axis=1)
        ind = np.argmax(factors)
        bin_arr = self.binned_arrays[ind]
        sum_middle = bin_arr.array.sum(axis=axis)
        pre_offsets = bin_arr.offsets[axis]
        num, post_offsets = np.divmod(self.array.shape[axis] - pre_offsets, self.bin_factors[ind][axis])

        if np.all(post_offsets == 0) and np.all(pre_offsets == 0):
            return sum_middle
        starting_slice = [slice(None), ] * self.array.ndim
        slices = []
        running_pre = starting_slice
        running_post = starting_slice
        for pre, post, ax in zip(pre_offsets, post_offsets, axis):
            sl = slice(None, pre)
            sl_end = slice(-post, None)
            running_pre[ax] = sl
            running_post[ax] = sl_end
            running_post.append(sl_end)
            slices.append(running_pre)
            slices.append(running_post)

        for sl in slices:
            # For more efficiency it might be better to slice the PyramidArray and return
            # a PyramidArray. In that case you should be able to "step down" the pyramid levels,
            # which should be more efficient that just hopping from the top level to the bottom most.
            sum_middle = sum_middle + self.array[sl].sum(axis=axis)

        return sum_middle


    def __add__(self, other: np.typing.ArrayLike):
        """Sum the array along all dimensions."""
        pass

    def __sub__(self, other):
        """Subtract the array along all dimensions."""
        pass

    def __mul__(self, other):
        """Multiply the array along all dimensions."""
        pass


