import numpy as np
from spyde.pyramid_array import PyramidArray, bin_ndarray
import pytest


class TestPyramidArray:

    @pytest.fixture
    def pyramid_3d(self):
        """Fixture to create a 3D PyramidArray for testing."""
        data = np.arange(4*5*6).reshape(4, 5, 6)
        p = PyramidArray(data)
        p.add_pyramid_level((1, 1, 3))
        return p
    def test_initialization(self, pyramid_3d):
        """Test the initialization of the PyramidArray class."""
        p = pyramid_3d
        assert p.shape == (4, 5, 6)
        assert p.binned_arrays[0].array.shape == (4, 5, 2)
        assert p.binned_arrays[0].binning_factors == (1, 1, 3)
        assert p.binned_arrays[0].offsets == (0, 0, 0)

    def test_getitem(self, pyramid_3d):
        """Test the __getitem__ method of the PyramidArray class."""
        p = pyramid_3d
        sub_array = p[1:3, 2:4, 1:]
        assert sub_array.shape == (2, 2, 5)
        assert sub_array.binned_arrays[0].array.shape == (2, 2, 1)
        assert sub_array.binned_arrays[0].binning_factors == (1, 1, 3)
        assert sub_array.binned_arrays[0].offsets == (0, 0, 1)
