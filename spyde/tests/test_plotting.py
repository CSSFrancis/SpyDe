import pytest
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

# Assuming your main window class is named `MainWindow`
from spyde.main_window import MainWindow
from unittest.mock import patch
import hyperspy.api as hs
import numpy as np


class TestMainWindow:
    """Test class for the MainWindow."""

    def test_main_window_initialization(self, qtbot):
        """Test to ensure the main window initializes properly."""
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        qtbot.addWidget(window)
        assert window is not None
        assert isinstance(window, QMainWindow)
        assert window.windowTitle() == "SpyDE"  # Adjust based on your actual title
        assert window.client is not None



    @pytest.mark.parametrize("dimensions", [2,3,4,5])
    def test_load_data_and_plot_images(self, qtbot, tmpdir, dimensions):
        """Test to ensure data can be loaded and plotted."""
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        qtbot.addWidget(window)
        shape = np.arange(5,5+dimensions)
        np.arange(np.prod(shape)).reshape(shape)
        s = hs.signals.Signal2D(np.arange(np.prod(shape)).reshape(shape))
        file = tmpdir.join("test.hspy")
        s.save(tmpdir.join("test.hspy"))
        window._create_signals([str(file),])
        if dimensions ==2:
            assert len(window.plot_subwindows)==1
        else:
            assert len(window.plot_subwindows)==2
            if dimensions == 3:
                assert window.plot_subwindows[0].ndim == 1
                assert window.plot_subwindows[1].ndim == 2
            elif dimensions == 4:
                assert window.plot_subwindows[0].ndim == 2
                assert window.plot_subwindows[1].ndim == 2
            elif dimensions == 5:
                assert window.plot_subwindows[0].ndim == 1
                assert window.plot_subwindows[1].ndim == 2

    @pytest.mark.parametrize("dimensions", [1,2,3,4])
    def test_load_data_and_plot_lines(self, qtbot, tmpdir, dimensions):
        """Test to ensure data can be loaded and plotted."""
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        qtbot.addWidget(window)
        shape = np.arange(5,5+dimensions)
        np.arange(np.prod(shape)).reshape(shape)
        s = hs.signals.Signal1D(np.arange(np.prod(shape)).reshape(shape))
        file = tmpdir.join("test.hspy")
        s.save(tmpdir.join("test.hspy"))

        window._create_signals([str(file),])
        if dimensions == 1:
            assert len(window.plot_subwindows)==1
        else:
            assert len(window.plot_subwindows)==2
            if dimensions == 2:
                assert window.plot_subwindows[0].ndim == 1
                assert window.plot_subwindows[1].ndim == 1
            elif dimensions == 3:
                assert window.plot_subwindows[0].ndim == 2
                assert window.plot_subwindows[1].ndim == 1
            elif dimensions == 4:
                assert window.plot_subwindows[0].ndim == 1
                assert window.plot_subwindows[1].ndim == 2

    def test_plot_update(self, qtbot):
        """Test to ensure the plot updates correctly when the selectors move."""
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        qtbot.addWidget(window)
        # Assuming you have a method to update the plot
        window.update_plot()
        # Check if the plot was updated correctly
        assert window.plot_subwindows[0].isVisible()
