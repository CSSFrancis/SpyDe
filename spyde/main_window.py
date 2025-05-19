import distributed
import pyxem.data
import webbrowser

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap, QColor


import sys
import os
from functools import partial
import hyperspy.api as hs
import dask.array as da
from dask.distributed import Client, Future, LocalCluster


from spyde.camera_control import CameraControlDock
from spyde.misc.dialogs import DatasetSizeDialog
from spyde.plot import Plot
import qdarktheme
if not sys.platform == "win32":
    from PyQt6.QtWebEngineWidgets import QWebEngineView


class MainWindow(QtWidgets.QMainWindow):
    """
    A class to manage the main window of the application.
    """

    def __init__(self, app=None):
        super().__init__()
        self.app = app
        qdarktheme.setup_theme("light")
        qdarktheme.setup_theme(corner_shape="rounded")
        # Test if the theme is set correctly

        cpu_count = os.cpu_count()
        threads = (cpu_count//4) -1
        cluster = LocalCluster(n_workers=threads, threads_per_worker=4)
        self.client = Client(cluster)  # Start a local Dask client (this should be settable eventually)
        print(self.client.dashboard_link)
        self.setWindowTitle("SpyDE")
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
        self.create_menu()
        self.setMouseTracking(True)

        if not sys.platform == "win32":
            self.add_dask_dashboard()


        self.signal_tree = None
        self.selector_list = None
        self.s_list_widget = None
        self.add_plot_control_widget()
        self.file_dialog = None

        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)  # Every 10ms we will check to update the plots??
        self.timer.timeout.connect(self.update_plots_loop)
        self.timer.start()

        camera_control_dock = CameraControlDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, camera_control_dock)
        camera_control_dock.hide()

    def update_plots_loop(self):
        """This is a simple loop to check if the plots need to be updated. Currently, this
        is running on the main event loop but it could be moved to a separate thread if it
        starts to slow down the GUI.

        """
        for p in self.plot_subwindows:
            if isinstance(p.data, Future) and p.data.done():
                print("Updating Plot")
                p.data = p.data.result()
                p.update()

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

        names = ["mgo_nanocrystals", "small_ptychography", "zrnb_precipitate", "pdcusi_insitu"]
        for n in names:
            action = example_data.addAction(n)
            action.triggered.connect(partial(self.load_example_data, n))

        # Add View Menu
        view_menu = menubar.addMenu("View")
        if not sys.platform == "win32":
            toggle_dask_dashboard = QAction("Toggle Dask Dashboard", self)
            toggle_dask_dashboard.setChecked(True)
            toggle_dask_dashboard.triggered.connect(self.toggle_dask_dashboard_visibility)
            view_menu.addAction(toggle_dask_dashboard)

        # Add a view to open the dask dashboard
        view_dashboard_action = QAction("Open Dask Dashboard", self)
        view_dashboard_action.triggered.connect(self.open_dask_dashboard)
        view_menu.addAction(view_dashboard_action)

        # Add a view to toggle the camera control
        toggle_camera_control = QAction("Toggle Camera Control", self)
        toggle_camera_control.setChecked(True)
        toggle_camera_control.triggered.connect(self.toggle_camera_control_visibility)
        view_menu.addAction(toggle_camera_control)

        # Add Light/Dark Mode Toggle
        toggle_theme_action = QAction("Toggle Light/Dark Mode", self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(toggle_theme_action)

    def open_dask_dashboard(self):
        """
        Open the Dask dashboard in a new window.
        """
        if self.client:
            dashboard_url = self.client.dashboard_link
            webbrowser.open(dashboard_url)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Dask client is not initialized.")

    def toggle_dask_dashboard_visibility(self):
        """
        Toggle the visibility of the Dask dashboard.
        """
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if dock.windowTitle() == "Dask Dashboard":
                if dock.isVisible():
                    dock.hide()
                else:
                    dock.show()

    def toggle_theme(self):
        """
        Toggle between light and dark mode.
        """
        if qdarktheme.get_themes() == "dark":
            qdarktheme.setup_theme("light")
        else:
            qdarktheme.setup_theme("dark")

    def toggle_camera_control_visibility(self):
        """
        Toggle the visibility of the Camera Control dock.
        """
        for dock in self.findChildren(CameraControlDock):
            if dock.isVisible():
                dock.hide()
            else:
                dock.show()

    def _create_signals(self, file_paths):
        for file_path in file_paths:
            kwargs = {"lazy": True}
            if file_path.endswith(".mrc"):
                dialog = DatasetSizeDialog(self, filename=file_path)
                if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    x_size = dialog.x_input.value()
                    y_size = dialog.y_input.value()
                    time_size = dialog.time_input.value()
                    kwargs["navigation_shape"] = tuple([val for val in (x_size, y_size, time_size) if val > 1])
                    print(f"{kwargs['navigation_shape']}")
                else:
                    print("Dialog cancelled")
                    return
                # .mrc always have 2 signal axes.  Maybe needs changed for eels.
                if len(kwargs["navigation_shape"]) == 3:
                    kwargs["chunks"] = ((1,) + ("auto",) * (len(kwargs["navigation_shape"]) - 1)) + (-1, -1)
                else:
                    kwargs["chunks"] = (("auto",) * len(kwargs["navigation_shape"])) + (-1, -1)

                print(f"chunks: {kwargs['chunks']}")
                kwargs["distributed"] = True

            signal = hs.load(file_path, **kwargs)
            print(signal.data.chunks)
            hyper_signal = HyperSignal(signal, main_window=self, client=self.client)
            print(hyper_signal)
            plot = Plot(hyper_signal, is_signal=False, key_navigator=True)
            plot.main_window = self
            plot.titleColor = QColor("lightgray")
            self.add_plot(plot)
            print("Adding selector and plot")
            plot.add_selector_and_new_plot()

    def open_file(self):
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        self.file_dialog.setNameFilter("Hyperspy Files (*.hspy), mrc Files (*.mrc)")

        if self.file_dialog.exec():
            file_paths = self.file_dialog.selectedFiles()
            if file_paths:
                self._create_signals(file_paths)

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
        t_input = QtWidgets.QSpinBox()
        t_input.setRange(1, 10000)
        t_input.setValue(0)
        sizes_layout.addWidget(QtWidgets.QLabel("Time Size:"))
        sizes_layout.addWidget(t_input)

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
            t_size = t_input.value()
            x_size = x_input.value()
            y_size = y_input.value()
            kx_size = kx_input.value()
            ky_size = ky_input.value()
            size = (t_size, x_size, y_size, kx_size, ky_size)
            size = tuple([s for s in size if s>1])

            if random_radio.isChecked():
                chunks = ("auto",) * (len(size) - 2) + (-1, -1)
                data = da.random.random(size, chunks=chunks)
                s = hs.signals.Signal2D(data).as_lazy()
                s.cache_pad = 2
            else:  # multiphase_radio.isChecked():
                s = pyxem.data.fe_multi_phase_grains(size=x_size,
                                                     recip_pixels=kx_size,
                                                     num_grains=4).as_lazy(chunks=("auto",
                                                                                   "auto",
                                                                                   -1,
                                                                                   -1))
                s.cache_pad = 2

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

        plot.setWindowTitle("Test")
        plot.titleColor = QColor("green")
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
        Add a Dask dashboard to the main window on the bottom.

        This is useful for monitoring the progress of computations and debugging.
        """
        webview = QWebEngineView()
        task_stream = self.client.dashboard_link.split("status")[0] + "individual-task-stream"
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
        dock_widget.hide()

    def close(self):
        self.client.close()
        super().close()


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

    Because of the 1st class nature of lazy signals there are limits to how fast this class can
    be.  Hardware optimization is very, very important to get the most out of this class.  That being
    said dask task-scheduling is always going to be somewhat of a bottleneck.

    Parameters
    ----------
    signal : hs.signals.BaseSignal
        The hyperspy signal to plot.
    main_window : MainWindow
        The main window of the application.
    client : distributed.Client
        The Dask client to use for computations.
    """

    def __init__(self,
                 signal: hs.signals.BaseSignal,
                 main_window: MainWindow,
                 parent_signal=None,
                 client: distributed.Client = None):
        self.signal = signal
        self.client = client
        self.main_window = main_window
        self.parent_signal = parent_signal
        print(self.signal)
        print("navigator", self.signal.navigator)

        if len(signal.axes_manager.navigation_axes) > 0 and len(signal.axes_manager.signal_axes) != 0:
            if signal._lazy and signal.navigator is not None:
                nav_sig = signal.navigator
            else:
                nav_sig = signal.sum(signal.axes_manager.signal_axes)
                if nav_sig._lazy:
                    nav_sig.compute()
            if not isinstance(nav_sig, hs.signals.BaseSignal):
                nav_sig = hs.signals.BaseSignal(nav_sig).T
            if len(nav_sig.axes_manager.navigation_axes) > 2: #
                nav_sig = nav_sig.transpose(2)

            self.nav_sig = HyperSignal(nav_sig,
                                       main_window=self.main_window,
                                       parent_signal=self,
                                       client=self.client
                                       )  # recursive...
        else:
            self.nav_sig = None

        # A tree of signal transformations.  You can navigate the tree to see the
        # transformations that have been applied to the signal. For different
        # signal plots.  (Not implemented yet)
        self.signal_transformation_trees = {"signal": self.signal}  # --> only signals??
        self.navigation_plots = []
        self.signal_plots = []

    def apply_mapped_function(self, function, signal_key=None, *args, **kwargs):
        """
        Apply a function which calls the underlying `hyperspy` `BaseSignal.map` function.

        This operates lazily and will not compute the result but instead will create a new
        signal which shares the same navigator. You can toggle between the two signals.  Signals
        further down the tree will reuse cached data to reduce computation time. Otherwise
        it's recommended to not use the "live" option.

        Parameters
        ----------
        function : str
            The name of the function to apply.
        signal_key : str
            The key of the signal to apply the function to.
        args : list
            The positional arguments to pass to the function.
        kwargs : dict
            The keyword arguments to pass to the function
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SpyDe")  # Set the application name
    # Create and show the splash screen
    logo_path = "SpydeDark.png"  # Replace with the actual path to your logo
    pixmap = QPixmap(logo_path).scaled(300, 300,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)

    splash = QSplashScreen(pixmap,
                           Qt.WindowType.FramelessWindowHint)
    splash.show()
    splash.raise_()  # Bring the splash screen to the front
    app.processEvents()
    main_window = MainWindow(app=app)

    main_window.setWindowTitle("SpyDe")  # Set the window title

    if sys.platform == "darwin":
        logo_path = "Spyde.icns"
    else:
        logo_path = "SpydeDark.png"  # Replace with the actual path to your logo
    main_window.setWindowIcon(QIcon(logo_path))
    main_window.show()
    splash.finish(main_window)  # Close the splash screen when the main window is shown

    app.exec()

