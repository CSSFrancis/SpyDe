from functools import partial

import numpy as np
from PyQt6 import QtWidgets
import hyperspy.api as hs


class DatasetSizeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, filename=None):
        print("Creating Dialog")
        super().__init__(parent)

        self.filename = filename

        kwargs = {}
        # try to load the dataset
        print(f"loading: {filename}")
        if ".mrc" in filename:
            kwargs["distributed"] = True
        try:
            data = hs.load(filename, lazy=True, **kwargs)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.reject()
            return
        nav_shape = [a.size for a in data.axes_manager.navigation_axes]
        x, y, t = nav_shape + ([0,] * (3-len(nav_shape)))


        self.total_frames = np.prod(nav_shape)
        print(self.total_frames)

        sig_shape = [a.size for a in data.axes_manager.signal_axes]
        kx, ky, kz = sig_shape + ([0,] * (3-len(sig_shape)))
        print("setting_size")

        self.setWindowTitle("Dataset Size Configuration")

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Input fields for x, y, and time sizes
        self.x_input = QtWidgets.QSpinBox()
        self.x_input.setRange(1, 100000)
        self.x_input.setValue(x)
        set_x = partial(self.update_image_size, 0)
        self.x_input.valueChanged.connect(set_x)

        set_y = partial(self.update_image_size, 1)
        self.y_input = QtWidgets.QSpinBox()
        self.y_input.setRange(1, 100000)
        self.y_input.setValue(y)
        self.y_input.valueChanged.connect(set_y)

        set_t = partial(self.update_image_size, 2)
        self.time_input = QtWidgets.QSpinBox()
        self.time_input.setRange(1, 10000)
        self.time_input.setValue(t)
        self.time_input.valueChanged.connect(set_t)

        # Labels and inputs
        layout.addWidget(QtWidgets.QLabel("X Size:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QtWidgets.QLabel("Y Size:"))
        layout.addWidget(self.y_input)
        layout.addWidget(QtWidgets.QLabel("Time Size:"))
        layout.addWidget(self.time_input)

        # Display for image size in pixels
        self.image_size_label = QtWidgets.QLabel(f"Image Size (Pixels):( {kx}, {ky})")
        layout.addWidget(self.image_size_label)
        # Add a button to enable/disable the time input
        self.toggle_time_button = QtWidgets.QPushButton("Enable Time Input")
        self.toggle_time_button.setCheckable(True)
        self.toggle_time_button.toggled.connect(self.toggle_time_input)
        layout.addWidget(self.toggle_time_button)
        self.time_input.setEnabled(False)  # Initially disable the time input
        # OK and Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.update_image_size()

    def update_image_size(self, index=None):
        """Update the image size in pixels based on x and y inputs."""
        if self.time_input.isEnabled():
            x = self.x_input.value()
            y = self.y_input.value()
            t = self.time_input.value()
            if index == 0 or index == 1:
                t = self.total_frames // (x * y)
                self.time_input.setValue(t)
        else:
            if index ==0:
                x_size = self.x_input.value()
                self.y_input.setValue(self.total_frames//x_size)
            else:
                y_size = self.y_input.value()
                self.x_input.setValue(self.total_frames//y_size)

    def toggle_time_input(self, checked):
        """Enable or disable the time input box."""
        self.time_input.setEnabled(checked)