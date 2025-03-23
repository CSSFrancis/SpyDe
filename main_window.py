import numpy as np
import pyxem.data
from hyperspy.roi import CircleROI
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWebEngineWidgets import QWebEngineView

import sys
import fastplotlib as fpl
import hyperspy.api as hs
import dask.array as da
from dask.distributed import Client, Future
from external.rectangle_selector import RectangleSelector, CircleSelector, GraphicFeature
from functools import partial
import time

class PlotUpdateWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):
        """Long-running task."""
        print("Running Plot Update Loop")
        waiting_plots = True
        while waiting_plots:
            waiting_plots = False
            for p in self.main_window.plot_subwindows:
                if isinstance(p.data, Future) and p.data.done():
                    print("Updating Plot")
                    p.data = p.data.result()
                    p.update()
                elif isinstance(p.data, Future):
                    print("Waiting for computation")
                    waiting_plots = True
                    time.sleep(0.1)
        print("All plots updated")
        self.finished.emit()

class MainWindow(QtWidgets.QMainWindow):
    """
    A class to manage the main window of the application.
    """
    def __init__(self):
        super().__init__()
        self.client = Client()  # Start a local Dask client (this should be settable eventually)
        print(self.client.dashboard_link)
        self.setWindowTitle("Hyperspy Plot")
        # get screen size and set window size to 3/4 of the screen size
        # get screen size and set subwindow size to 1/4 of the screen size
        screen = QtWidgets.QApplication.primaryScreen()
        self.screen_size = screen.size()
        self.resize(self.screen_size.width() * 3 // 4, self.screen_size.height() * 3 // 4)

        # center the main window on the screen
        self.move(
            (self.screen_size.width() - self.width()) // 2,
            (self.screen_size.height() - self.height()) // 2
        )
        # create an MDI area
        self.mdi_area = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi_area)

        self.plot_subwindows = []

        self.mdi_area.subWindowActivated.connect(self.on_subwindow_activated)
        self.add_dask_dashboard()
        self.create_menu()

        self.signal_tree = None
        self.selector_list = None
        self.s_list_widget = None
        self.add_plot_control_widget()

        self.plot_update_event_thread = QtCore.QThread()
        self.plot_update_event_worker = PlotUpdateWorker(self)
        self.plot_update_event_worker.moveToThread(self.plot_update_event_thread)

    def start_plot_update_loop(self):
        print("Starting Plot Update Loop")
        self.plot_update_event_thread.started.connect(self.plot_update_event_worker.run)
        self.plot_update_event_worker.finished.connect(self.plot_update_event_thread.quit)
        #self.plot_update_event_worker.finished.connect(self.plot_update_event_worker.deleteLater)
        #self.plot_update_event_thread.finished.connect(self.plot_update_event_thread.deleteLater)
        # Step 6: Start the thread
        self.plot_update_event_thread.start()

    def create_menu(self):
        menubar = self.menuBar()

        # Add File Menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        open_create_data_dialog = QAction("Create Data...", self)
        open_create_data_dialog.triggered.connect(self.create_data)
        file_menu.addAction(open_create_data_dialog)



        example_data = file_menu.addMenu("Load Example Data...")

        names = ["mgo_nanocrystals", "small_ptychography", "zrnb_precipitate"]
        for n in names:
            action = example_data.addAction(n)
            action.triggered.connect(partial(self.load_example_data, n))



    def open_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Hyperspy Files (*.hspy)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
                signal = hs.load(file_path, lazy=True)
                hyper_signal = HyperSignal(signal, main_window=self, client=self.client)
                plot = Plot(hyper_signal, is_signal=False, key_navigator=True)
                plot.main_window = self
                self.add_plot(plot)
                plot.add_selector_and_new_plot()

    def create_data(self):
        """
        A dialog for creating data for testing purposes.
        """

        random_data_dialog = QtWidgets.QDialog(self)
        random_data_dialog.setWindowTitle("Create Random Data")

        # Create layout
        layout = QtWidgets.QVBoxLayout(random_data_dialog)
        buttons_and_sizes_layout = QtWidgets.QHBoxLayout()
        sizes_layout = QtWidgets.QVBoxLayout()

        button_layout = QtWidgets.QVBoxLayout()
        buttons_and_sizes_layout.addLayout(button_layout)
        buttons_and_sizes_layout.addLayout(sizes_layout)
        layout.addLayout(buttons_and_sizes_layout)

        # Create radio buttons for multiphase, strain, and random
        multiphase_radio = QtWidgets.QRadioButton("Multiphase")
        #strain_radio = QtWidgets.QRadioButton("Strain")
        random_radio = QtWidgets.QRadioButton("Random")
        random_radio.setChecked(True)  # Set default selection

        # Add radio buttons to layout
        button_layout.addWidget(multiphase_radio)
        #button_layout.addWidget(strain_radio)
        button_layout.addWidget(random_radio)

        # Create a button group to ensure only one radio button can be selected at a time
        button_group = QtWidgets.QButtonGroup(random_data_dialog)
        button_group.addButton(multiphase_radio)
        #button_group.addButton(strain_radio)
        button_group.addButton(random_radio)

        # Create input fields for x, y, kx, ky
        x_input = QtWidgets.QSpinBox()
        x_input.setRange(1, 10000)
        x_input.setValue(128)
        sizes_layout.addWidget(QtWidgets.QLabel("X Size:"))
        sizes_layout.addWidget(x_input)

        y_input = QtWidgets.QSpinBox()
        y_input.setRange(1, 10000)
        y_input.setValue(128)
        sizes_layout.addWidget(QtWidgets.QLabel("Y Size:"))
        sizes_layout.addWidget(y_input)

        kx_input = QtWidgets.QSpinBox()
        kx_input.setRange(1, 10000)
        kx_input.setValue(64)
        sizes_layout.addWidget(QtWidgets.QLabel("KX Size:"))
        sizes_layout.addWidget(kx_input)

        ky_input = QtWidgets.QSpinBox()
        ky_input.setRange(1, 10000)
        ky_input.setValue(64)
        sizes_layout.addWidget(QtWidgets.QLabel("KY Size:"))
        sizes_layout.addWidget(ky_input)

        # Add OK and Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)

        # Connect buttons
        button_box.accepted.connect(random_data_dialog.accept)
        button_box.rejected.connect(random_data_dialog.reject)

        # Show dialog and get result
        if random_data_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            x_size = x_input.value()
            y_size = y_input.value()
            kx_size = kx_input.value()
            ky_size = ky_input.value()

            if random_radio.isChecked():
                data = da.random.random((kx_size, ky_size, x_size, y_size), chunks=("auto", "auto", -1, -1))
                s = hs.signals.Signal2D(data).as_lazy()
            else:  # multiphase_radio.isChecked():
                s = pyxem.data.fe_multi_phase_grains(size=x_size,
                                                          recip_pixels=kx_size,
                                                          num_grains=4).as_lazy(chunks=("auto",
                                                                                        "auto",
                                                                                        -1,
                                                                                        -1))

            self.add_signal(s)

    def add_signal(self, s):
        """
        Add a signal to the main window.
        """
        hyper_signal = HyperSignal(s, main_window=self, client=self.client)
        plot = Plot(hyper_signal, is_signal=False, key_navigator=True)
        plot.main_window = self
        self.add_plot(plot)
        plot.add_selector_and_new_plot()

    def load_example_data(self, name):
        """
        Load example data for testing purposes.
        """
        signal = getattr(pyxem.data, name)(allow_download=True, lazy=True)
        self.add_signal(signal)

    def add_plot(self, plot):
        plot.resize(self.screen_size.height() // 2, self.screen_size.height() // 2)
        self.mdi_area.addSubWindow(plot)
        plot.show()
        self.plot_subwindows.append(plot)
        return

    def on_subwindow_activated(self, window):
        if hasattr(window, "show_selector_control_widget"):
            window.show_selector_control_widget()
        for plot in self.plot_subwindows:
            if window != plot and hasattr(plot, "hide_selector_control_widget"):
                plot.hide_selector_control_widget()

    def add_plot_control_widget(self):
        dock_widget = QtWidgets.QDockWidget("Plot Control", self)
        dock_widget.setBaseSize(self.width() // 6, self.height() // 6)

        # Create a main widget and layout
        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main_widget)

        # Create a tree widget for the signal tree
        self.signal_tree = QtWidgets.QTreeWidget()
        self.signal_tree.setHeaderLabel("Signal Tree")
        layout.addWidget(self.signal_tree)

        # Create a list widget for the selectors
        self.s_list_widget = QtWidgets.QWidget()
        self.selector_list = QtWidgets.QVBoxLayout(self.s_list_widget)

        layout.addWidget(self.s_list_widget)

        dock_widget.setWidget(main_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)

    def add_dask_dashboard(self):
        """
        just
        :return:
        """
        webview = QWebEngineView()
        task_stream = self.client.dashboard_link.split("status")[0] + "individual-task-stream"
        print(task_stream)
        webview.setUrl(QtCore.QUrl(task_stream))
        dock_widget = QtWidgets.QDockWidget("Dask Dashboard", self)
        dock_widget.setWidget(webview)
        dock_widget.setBaseSize(self.width() // 6, self.height() // 6)
        dock_widget.setMaximumHeight(self.height() // 4)
        dock_widget.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        dock_widget.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock_widget.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock_widget)
        dock_widget.setBaseSize(self.width() // 6, self.height() // 6)


class SignalTree:
    """
    A class to manage the signal tree. This class manages the tree of different signals
    after some transformation has been applied.  The idea is that you can toggle between
    the different signals to see the effects of transformations such as filtering, centering
    the direct beam, azimuthal integration, etc.

    For example, you might have a tree like this:
                                              -/-> [FEM Variance]
                                            /
               --> [denoise filter] --> [centered] --> [azimuthal integration]
             /
      [signal]
             \
              --> [centered] --> [get_diffraction_vectors] --> [strain matrix] -/-> [strain maps]
                        \
                         -/-> [get_virtual_image]

    -----------------------------------------------------------------------------------------------

                                           -/-> [Bright-field](toggle visible/not visible)
                                         /
       [navigator] --> [signal] --> [Centered]
                                         \
                                          -/-> [Dark-field] (toggle visible/not visible)

    -----------------------------------------------------------------------------------------------
    Then you can select the different steps in the tree to see the data computed at that point.  Calling the
    compute function will break the tree and create a new plot. In contrast, a non-breaking function will
    just update the current "signal" plot with the new data. Toggling back to the previous fork in the tree will
    allow you to see the data along the way.

    Certain "broken" branches which are

    """

class HyperSignal:
    """
    A class to manage the plotting of hyperspy signals. This class manages the
    different plots associated with a hyperspy signal.

    Because of the 1st class nature of lazy signals the navigation signal won't be "live"
    """

    def __init__(self, signal, main_window, client=None):
        self.signal = signal
        self.client = client
        self.main_window = main_window

        if signal.navigator is not None:
            self.key_navigation_signal = signal.navigator
        else:
            self.key_navigation_signal = signal.sum(signal.axes_manager.signal_axes)
            if self.key_navigation_signal._lazy:
                self.key_navigation_signal.compute()

        # A tree of signal transformations.  You can navigate the tree to see the
        # transformations that have been applied to the signal. For different
        # signal plots.  (Not implemented yet)
        self.signal_transformation_trees = {"signal": self.signal}  # --> only signals??
        self.navigation_plots = []
        self.signal_plots = []

    def apply_mapped_function(self, function, signal_key=None, *args, **kwargs):
        """
        Apply a function which calls the underlying `hyperspy` `BaseSignal.map` function.
        """
        if signal_key is None:
            signal_key = "signal"

        s = self.signal_transformation_trees[signal_key]["signal"]
        func = getattr(s, function)
        new_signal = func(inplace=True, *args, **kwargs)
        self.signal_transformation_trees[function] = {"signal": new_signal}

    def update_signal_transformations(self):
        """
        Update the signal transformations tree.
        """
        signal_tree = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(signal_tree)
        button = QtWidgets.QPushButton("Raw Signal")

        layout.addWidget(button)
        signal_tree.setLayout(layout)


class Selector:
    def __init__(self,
                 on_nav=True,
                 type="RectangleSelector",
                 *args,
                 **kwargs):
        if type == "RectangleSelector":
            self.selector = RectangleSelector(*args, **kwargs, edge_thickness=4)
        elif type == "CircleSelector":
            self.selector = CircleSelector(*args, **kwargs, edge_thickness=4)
        else:
            raise ValueError("Invalid Selector Type")
        self.is_live = not on_nav  # if on signal is_live is always False
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.widget)
        self.plot = None
        self.is_integrating = False

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

    def __getattr__(self, item):
        try:
            return getattr(self.selector, item)
        except AttributeError:
            raise AttributeError(f"'Selector' object has no attribute '{item}'")

    def on_integrate_toggled(self, checked):
        print("Integrate Toggled")
        if self.is_live:
            self.is_integrating = checked
            self.plot.update_data()

    def on_integrate_pressed(self):
        if not self.is_live:
            # fire off the integration
            print("Computing!")
            self.plot.compute_data()

    def on_live_toggled(self, checked):
        self.is_live = checked
        if checked:
            self.integrate_button.setText("Integrate")
            self.integrate_button.setCheckable(True)
            self.integrate_button.setChecked(self.is_integrating)
            self.selection = (self.selection[0], self.selection[0]+15, self.selection[2], self.selection[2]+15)
            self.size_limits = (1, 15, 1, 15)
            # update the plot
            self.plot.update_data()
        else:
            self.integrate_button.setText("Compute")
            self.is_integrating = True
            self.integrate_button.setCheckable(False)
            self.size_limits = (1, self.limits[1], 1, self.limits[3])


class Plot(QtWidgets.QMdiSubWindow):
    """
    A class to manage the plotting of hyperspy signals and images.

    This class is a subclass of fastplotlib.Figure and is used to manage the
    plotting of hyperspy signals and images.

    Each plot which is linked to a selector can be updated as the selector moves.
    """

    def __init__(self, signal, key_navigator=False, is_signal=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = None  # None --> Static plot points to selector on another Plot object
        self.is_signal = is_signal
        self.qt_widget = None
        self.main_window = None
        self.hyper_signal = signal
        self.selectors = []  # List of selectors associated on the plot
        self.fpl_fig = fpl.Figure()
        self.fpl_fig[0, 0].axes.visible = False
        self.data = None
        self.current_indexes = []

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.CustomizeWindowHint)
        qwidget = self.fpl_fig.show()
        self.setWidget(qwidget)

        if key_navigator:
            self.data = self.hyper_signal.key_navigation_signal.data
            self.fpl_image = self.fpl_fig[0, 0].add_image(self.data, name="navigator")
            self.setWindowTitle("Key Navigator Plot")
            self.fpl_image.title = "Key Navigator"

        elif is_signal:
            self.setWindowTitle("Signal Plot" if is_signal else "Navigation Plot")
            self.current_indexes = np.array([0, 0])
            current_img = self.hyper_signal.signal._get_cache_dask_chunk([0, 0], get_result=True)
            self.fpl_image = self.fpl_fig[0, 0].add_image(current_img, name="signal")
            self.fpl_image.title = "Signal"
        else:
            self.setWindowTitle("Virtual Image Plot")
            self.fpl_image = self.fpl_fig[0, 0].add_image(self.hyper_signal.key_navigation_signal.data,
            name="virtual_image")
            self.fpl_image.title = "Virtual Image"


        self.fpl_fig[0, 0].center_graphic(self.fpl_image)
        self.fpl_image.add_event_handler(self.get_context_menu, "pointer_up")

    @property
    def is_live(self):
        if self.selector:
            return self.selector.is_live
        else:
            return False

    @is_live.setter
    def is_live(self, value):
        if self.selector and self.is_signal:
            self.selector.is_live = value
        else:
            print("Selector is not set")

    @property
    def is_integrating(self):
        if self.selector:
            return self.selector.is_integrating
        else:
            return False

    def add_selector_and_new_plot(self, type="RectangleSelector"):
        limits = (0, self.fpl_image._data.value.shape[1], 0,
                  self.fpl_image._data.value.shape[0])

        if type == "RectangleSelector":
            kwargs = {"selection": (1, 4, 1, 4)}
            if self.is_signal:
                kwargs["size_limits"] = (1, 15, 1, 15)
            else:
                kwargs["size_limits"] = None
        else:
            kwargs = {"center": (0, 0),
                      "radius": 2,
                      "size_limits": None
                      }

        selector = Selector(type=type,
                              resizable=True,
                              edge_color="red",
                              vertex_color="red",
                              fill_color=(0, 0, 0.35, 0.2),
                              parent=self.fpl_image,
                              limits=limits,
                              on_nav=not self.is_signal,
                              **kwargs)

        selector.is_live = not self.is_signal
        self.fpl_image._plot_area.add_graphic(selector.selector, center=False)

        self.selectors.append(selector)
        new_plot = Plot(self.hyper_signal,
                        is_signal=not self.is_signal,)  # flip/flop for back and forth
        selector.plot = new_plot
        new_plot.selector = selector

        new_plot.main_window = self.main_window
        self.main_window.add_plot(new_plot)
        selector.add_event_handler(new_plot.update_data, "selection")
        return new_plot

    def adjust_contrast(self):
        histogram_figure = fpl.Figure(size=(300, 300))
        if self.data is None:
            data = self.hyper_signal.signal._get_cache_dask_chunk(self.current_indexes, get_result=True)
        else:
            data = self.data

        histogram_widget = fpl.tools.HistogramLUTTool(data,
                                                      self.fpl_image,
                                                      )
        histogram_figure[0, 0].add_graphic(histogram_widget,)
        histogram_figure[0, 0].axes.visible = False
        histogram_figure.show()

    def get_context_menu(self, ev):
        if ev.button == 2:  # Right click
            context_menu = QtWidgets.QMenu()
            selector_menu = context_menu.addMenu("Add Selector")
            action = selector_menu.addAction("Add Rectangle Selector and New Plot")
            partial_add_selector = partial(self.add_selector_and_new_plot, type="RectangleSelector")
            action.triggered.connect(partial_add_selector)
            action = selector_menu.addAction("Add Circle Selector and New Plot")
            partial_add_selector = partial(self.add_selector_and_new_plot, type="CircleSelector")
            action.triggered.connect(partial_add_selector)
            action = context_menu.addAction("Adjust Contrast")
            action.triggered.connect(self.adjust_contrast)
            point = QtCore.QPoint(ev.x, ev.y)
            context_menu.exec(self.mapToGlobal(point))

    def show_selector_control_widget(self):
        """This is maybe a little complicated but there are problems when you try to
        remove a widget where it doesn't properly render the widget again.

        Sometimes it's easier to just "hide them"  and then show them again when you need them.

        There is a slight problem that we aren't actually deleting the selector... We also
        have the same problem with fast plotlib too....
        """
        for plot in self.main_window.plot_subwindows:
            if plot != self:
                for selector in plot.selectors:
                    selector.widget.hide()
        for selector in self.selectors:
            if selector.visible and selector.widget not in self.main_window.selector_list.children():
                self.main_window.selector_list.addWidget(selector.widget)
                selector.widget.show()
            elif selector.visible:
                selector.widget.show()

    def compute_data(self):
        """
        Compute the virtual image from the "selection".
        """
        x1, x2, y1, y2 = np.array(self.selector.selection).astype(int)
        if not self.is_signal:
            if isinstance(self.selector.selector, RectangleSelector):
                print(f"{time.time()}:Computing Signal:", x1, y1, x2, y2)
                print(self.hyper_signal.signal.data)
                virtual_img = self.hyper_signal.signal.isig[x1:x2, y1:y2]
                print(virtual_img)
                virtual_img = virtual_img.sum(axis=(-1, -2))
            else:
                inds = self.selector.selection
                rad = inds[2] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale
                inner_rad = inds[3] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale
                cx = ((inds[0] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale)+
                       self.hyper_signal.signal.axes_manager.signal_axes[0].offset)
                cy = (inds[1] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale+
                        self.hyper_signal.signal.axes_manager.signal_axes[1].offset)
                roi = CircleROI(cx, cy, rad, inner_rad)
                virtual_img = self.hyper_signal.signal.get_virtual_image(roi)
        else:
            if isinstance(self.selector.selector, RectangleSelector):
                virtual_img = self.hyper_signal.signal.inav[x1:x2, y1:y2].sum(axis=(0, 1))
            else:
                inds = self.selector.selection
                rad = inds[2] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale
                inner_rad = inds[3] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale
                cx = ((inds[0] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale)+
                       self.hyper_signal.signal.axes_manager.signal_axes[0].offset)
                cy = (inds[1] / self.hyper_signal.signal.axes_manager.signal_axes[0].scale+
                        self.hyper_signal.signal.axes_manager.signal_axes[1].offset)
                roi = CircleROI(cx,cy, rad, inner_rad)
                virtual_img = self.hyper_signal.signal.get_virtual_image(roi)

        lazy_arr = virtual_img.data  # make non blocking
        print(f"{time.time()}:Computing Virtual Image")
        data = self.hyper_signal.client.compute(lazy_arr)
        self.data = data
        print(f"{time.time()}:Starting loop")
        self.main_window.start_plot_update_loop()

    def update(self):
        self.fpl_image.data = self.data
        self.fpl_image.reset_vmin_vmax()

    def update_data(self, ev=None):
        # get the new data
        if self.selector.is_live or ev is None:
            if ev is None:
                indices = self.selector.get_selected_indices()
                print(indices)
            else:
                indices = ev.get_selected_indices()
            # snap to the nearest pixel values

            if not self.selector.is_integrating:
                try:
                    indices = np.round([np.mean(indices[0]), np.mean(indices[1])], 0).astype(int)
                except ValueError:
                    print("Failed to get mean indices")
                    return
            else:
                indices = np.reshape(np.array(np.meshgrid(indices[0], indices[1])).T, (-1, 2))
            if self.current_indexes.shape != indices.shape or not np.array_equal(self.current_indexes, indices):
                self.current_indexes = indices
                current_img = self.hyper_signal.signal._get_cache_dask_chunk(indices, get_result=False)
                if current_img is not None:
                    self.fpl_image.data = current_img
                    self.fpl_image.reset_vmin_vmax()

    def closeEvent(self, event):
        # Add your custom close logic here
        super().closeEvent(event)
        if self.selector:
            self.selector.remove_event_handler("selection")
            self.selector.visible = False
            self.selector.widget.hide()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SpyDe")  # Set the application name

    main_window = MainWindow()
    main_window.setWindowTitle("SpyDe")  # Set the window title
    main_window.show()

    signal = hs.signals.Signal2D(da.random.random((32, 32, 128, 128),
                                                  chunks=(16, 16, -1, -1))).as_lazy()
    signal.set_signal_type("electron_diffraction")
    hyper_signal = HyperSignal(signal, main_window, client=main_window.client)

    plot = Plot(hyper_signal, is_signal=False, key_navigator=True)

    plot.main_window = main_window
    main_window.add_plot(plot)
    plot.add_selector_and_new_plot()
    app.exec()
