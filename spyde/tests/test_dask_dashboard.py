import pytest
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

# Assuming your main window class is named `MainWindow`
from spyde.main_window import MainWindow
from unittest.mock import patch
import hyperspy.api as hs
import numpy as np

