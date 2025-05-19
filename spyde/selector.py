from spyde.external.rectangle_selector import RectangleSelector, CircleSelector
from PyQt6 import QtCore, QtWidgets
from fastplotlib import LinearSelector
import numpy as np


class Selector:
    def __init__(self,
                 on_nav=True,
                 integration_order=None,
                 type="RectangleSelector",
                 *args,
                 **kwargs):
        if type == "RectangleSelector":
            self.selector = RectangleSelector(*args, **kwargs, edge_thickness=4)
        elif type == "CircleSelector" or type == "RingSelector":
            self.selector = CircleSelector(*args, **kwargs, edge_thickness=4)
        elif type == "LineSelector":
            self.selector = LinearSelector(*args, **kwargs)
        else:
            raise ValueError("Invalid Selector Type")
        self.is_live = not on_nav  # if on signal is_live is always False
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.widget)
        self.plots = []
        self.is_integrating = False

        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(20)  # Every 20ms we will check to update the plots??
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.delayed_update_data)

        if on_nav:
            self.integrate_button = QtWidgets.QPushButton("Integrate")
            self.integrate_button.setCheckable(True)
            self.live_button = QtWidgets.QPushButton("Live")
            self.live_button.setCheckable(True)
            self.live_button.setChecked(True)
            self.live_button.toggled.connect(self.on_live_toggled)
            self.layout.addWidget(self.live_button)
        else:
            self.integrate_button = QtWidgets.QPushButton("Compute")
            self.integrate_button.setCheckable(False)

        self.integrate_button.setChecked(False)
        self.integrate_button.update()
        self.integrate_button.toggled.connect(self.on_integrate_toggled)
        self.integrate_button.pressed.connect(self.on_integrate_pressed)
        self.layout.addWidget(self.integrate_button)
        self.widget.setLayout(self.layout)
        self.integration_order = integration_order
        self.last_indices = [[0,0],]

    def __getattr__(self, item):
        try:
            return getattr(self.selector, item)
        except AttributeError:
            raise AttributeError(f"'Selector' object has no attribute '{item}'")

    def on_integrate_toggled(self, checked):
        print("Integrate Toggled")
        print(self.is_live)
        if self.is_live:
            self.is_integrating = checked
            self.update_data()
            for p in self.plots:
                p.update_plot(get_result=True)

    def on_integrate_pressed(self):
        if not self.is_live:
            # fire off the integration
            print("Computing!")
            for p in self.plots:
                p.compute_data()

    def on_live_toggled(self, checked):
        self.is_live = checked
        if checked:
            self.integrate_button.setText("Integrate")
            self.integrate_button.setCheckable(True)
            self.integrate_button.setChecked(self.is_integrating)
            self.selection = (self.selection[0], self.selection[0] + 15, self.selection[2], self.selection[2] + 15)
            self.size_limits = (1, 15, 1, 15)
            # update the plot
            for p in self.plots:
                p.update_data()
        else:
            self.integrate_button.setText("Compute")
            self.is_integrating = True
            self.integrate_button.setCheckable(False)
            self.size_limits = (1, self.limits[1], 1, self.limits[3])

    def get_selected_indices(self):
        if isinstance(self.selector, LinearSelector):
            self.selector.get_selected_index()
            print("getting selected inds")
            return np.array([self.selector.get_selected_index(),]).astype(int)
        else:
            indices = self.selector.get_selected_indices()
            if isinstance(self.selector, CircleSelector):
                if not self.is_integrating:
                    indices = np.array([np.round(np.mean(indices, axis=1)).astype(int),])
            elif isinstance(self.selector, RectangleSelector):
                if not self.is_integrating:
                    try:
                        indices = np.round([np.mean(indices[0]), np.mean(indices[1])], 0).astype(int)
                        indices = np.array([indices,])
                    except ValueError:
                        print("Failed to get mean indices")
                        return
                else:
                    indices = np.reshape(np.array(np.meshgrid(indices[0], indices[1])).T, (-1, 2))
            elif isinstance(self.selector, LinearSelector):
                indices = self.selector.selection
            return indices

    def update_data(self, ev=None):
        """
        Start the timer to delay the update.
        """
        if ev is None:
            self.delayed_update_data()
        elif self.is_live:
            print("Restarting Timer")
            self.update_timer.start()

    def delayed_update_data(self):
        """
        Perform the actual update if the indices have not changed.
        """
        print("Time out")
        indices = self.get_selected_indices()
        print("Indices", indices)
        print("Last Indices", self.last_indices)
        if not np.array_equal(indices, self.last_indices):
            print("Updating Data")
            self.last_indices = indices
            for p in self.plots:
                p.current_indexes[self.integration_order] = indices
                print("Updating Plot")
                p.update_plot()