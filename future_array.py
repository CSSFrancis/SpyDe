from typing import Tuple

import numpy as np


class FutureArray:
    """
    A class to represent an array of future values.  Most likely this is populated from an incoming
    stream of data.
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.frame_indexes = []  # a list of frame indexes which map to create a full array.
        self.futures = []  # a list of futures which map to create a full array based on the frame indexes.

    def spawn_futures(self):
        """
        Create a list of futures which will pull data from the incoming stream/queue. Each future
        will be associated with some

        """
        pass

    def map(self, func,
            signal_dimensions: int = 2,
            output_type: np.dtype = None,
            output_shape: Tuple = None,
            ) -> 'FutureArray':
        """
        Apply a function to each element in the array and return a new FutureArray.

        Parameters
        ----------
        func : callable
            A function to apply to each element in the array.
        signal_dimensions : int, optional
            The number of dimensions to map some function over. Default is 2.
        output_type : type, optional
            The type of the output array. If None, the output type will be the same as the input type.
        output_shape : tuple, optional
            The shape of the output array. If None, the output shape will be the same as the input shape.

        Returns
        -------
        FutureArray
            A new FutureArray with the results of applying func to each element.
        """
