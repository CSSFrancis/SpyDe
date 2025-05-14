import numpy as np


class TestPyramidArray

    def test_initialization(self):
        """Test the initialization of the PyramidArray class."""
        from spyde.pyramid_array import PyramidArray

        PyramidArray = PyramidArray(np.arrange(4*5*6).reshape(4, 5, 6), (2, 2, 2))