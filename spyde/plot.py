from PyQt6 import QtCore, QtWidgets
import time
from functools import partial

import fastplotlib as fpl
import numpy as np
import dask.array as da
from spyde.selector import Selector
from dask.distributed import Future
from fastplotlib.utils import quick_min_max
from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from external.titlebar import CustomSubWindow

def fast_index_virtual(arr, indexes, method="sum", reverse=True):
    ranges = np.vstack([np.min(indexes, axis=0), np.max(indexes, axis=0)]).T
    slice_ranges = tuple([slice(r[0], r[1]+1) for r in ranges])
    shape = np.diff(ranges, axis=1).flatten().astype(int)+1
    mask = np.zeros(shape, dtype=bool)
    mask[tuple([r for r in (indexes-ranges[:,0]).T])]=1
    extend = (arr.ndim - mask.ndim) * (np.newaxis,)
    if reverse:
        if method == "sum":
            arr = arr[..., *slice_ranges] * mask[*extend]
            axes = tuple(np.arange(1, len(mask.shape)+1, dtype=int)*-1)
            print(axes)
            print(arr.shape)
            return arr.sum(axis=axes)
        elif method == "mean":
            arr[slice_ranges] = np.nan  # upcasting and maybe much slower??
            return arr.nanmean(axis=tuple(1,np.arange(len(mask.shape)+1, dtype=int)*-1))


class Plot(CustomSubWindow):
    """
    A class to manage the plotting of hyperspy signals and images.

    This class is a subclass of fastplotlib.Figure and is used to manage the
    plotting of hyperspy signals and images.

    Each plot which is linked to a selector can be updated as the selector moves.
    """

    def __init__(self,
                 signal,
                 key_navigator=False,
                 is_signal=False,
                 selector_list=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if selector_list is None:
            selector_list = []
        self.selector_list = selector_list  # Selectors which effect the data on the plot
        self.is_signal = is_signal
        self.qt_widget = None
        self.main_window = None
        self.hyper_signal = signal
        self.selectors = []  # List of selectors associated on the plot
        self.fpl_fig = fpl.Figure()
        self.fpl_fig[0, 0].axes.visible = False
        self.data = None
        self.current_indexes = []
        self.current_indexes_dense = []

        qwidget = self.fpl_fig.show()
        self.layout().addWidget(qwidget)
        qwidget.setContentsMargins(0, 0, 0, 0)
        #self.setWidget(qwidget)
        self.plot_widget = qwidget

        if key_navigator:  # first initialization get to root
            if self.hyper_signal.nav_sig is not None:
                key_hypersignal = self.hyper_signal.nav_sig
                while key_hypersignal.nav_sig is not None:
                    key_hypersignal = key_hypersignal.nav_sig

                self.hyper_signal = key_hypersignal
                self.data = self.hyper_signal.signal.data
                name = "Key Navigator Plot"
        elif is_signal:
            self.setWindowTitle("Signal Plot" if is_signal else "Navigation Plot")
            self.current_indexes = [s.get_selected_indices() for s in self.selector_list]
            self.current_indexes_dense = self.get_dense_indexes()
            print("Current Indexes", self.current_indexes_dense)
            if self.hyper_signal.signal._lazy:
                print(f"getting current image from {self.hyper_signal.signal}")
                current_img = self.hyper_signal.signal._get_cache_dask_chunk(self.current_indexes_dense,
                                                                             get_result=True)
                print("Current Image", current_img.shape)
            else:
                tuple_inds = tuple([self.current_indexes_dense[:, ind]
                                    for ind in np.arange(self.current_indexes_dense.shape[1])])
                current_img = np.sum(self.hyper_signal.signal.data[tuple_inds], axis=0)
            self.data = current_img
            name = "Sig Plot"
        else:  # go back to the top? up one level?
            key_hypersignal = self.hyper_signal.nav_sig
            while key_hypersignal.nav_sig is not None:
                key_hypersignal = key_hypersignal.nav_sig

            self.current_indexes = [s.get_selected_indices() for s in self.selector_list]
            self.current_indexes_dense = self.get_dense_indexes()

            self.hyper_signal = key_hypersignal
            self.data = self.hyper_signal.signal.data
            name = "Virtual Image Plot"

        if self.data.ndim == 2:
            self.fpl_image = self.fpl_fig[0, 0].add_image(self.data, name="signal")
        elif self.data.ndim == 1:
            if key_navigator:
                axis = self.hyper_signal.signal.axes_manager.navigation_axes[0].axis
            else:
                axis = self.hyper_signal.signal.axes_manager.signal_axes[0].axis
            data = np.vstack([axis, (self.data-np.min(self.data))/
                              (np.max(self.data)-np.min(self.data))*np.max(axis),  # normalize
                              np.zeros_like(self.data)]).T
            self.fpl_image = self.fpl_fig[0, 0].add_line(data, name="signal")
            self.fpl_fig[0, 0].auto_scale()
            self.fpl_fig[0, 0].axes.visible = True
            self.fpl_image.title = name
        else:
            raise ValueError("Invalid data shape")

        self.fpl_fig[0, 0].center_graphic(self.fpl_image)
        self.fpl_image.add_event_handler(self.get_context_menu, "pointer_up")

        # Recursively install event filters on all child widgets
        for child in self.findChildren(QtWidgets.QWidget):
            child.installEventFilter(self)


    @property
    def ndim(self):
        return self.data.ndim

    @property
    def is_live(self):
        if self.selector_list is not []:
            return np.all([s.is_live for s in self.selector_list])
        else:
            return False

    @is_live.setter
    def is_live(self, value):
        if self.selector_list is not [] and self.is_signal:
            for s in self.selector_list:
                s.is_live = value
        else:
            print("Selector is not set")

    @property
    def is_integrating(self):
        if self.selector_list is not []:
            return np.all([s.is_integrating for s in self.selector_list])
        else:
            return False

    def add_selector_and_new_plot(self,
                                  down_step=True,
                                  type="RectangleSelector",
                                  *args,
                                  **kwargs):
        """
        Add a selector and a new plot to the main window.

        Parameters
        ----------
        type : str
            The type of selector to add.  Either "RectangleSelector", "CircleSelector" or
            "RingSelector".
        """

        limits = (0, self.fpl_image._data.value.shape[1], 0,
                  self.fpl_image._data.value.shape[0])

        if self.ndim == 2:
            kwargs = {"resizable": True,
                      "edge_color": "red",
                      "vertex_color": "red",
                      "fill_color": (0, 0, 0.35, 0.2),
                      "limits": limits}

            if type == "RectangleSelector":
                kwargs["selection"] = (1, 4, 1, 4)
                if self.is_signal:
                    kwargs["size_limits"] = None
                else:
                    kwargs["size_limits"] = (1, 15, 1, 15)
            elif type == "CircleSelector":
                kwargs["center"] = (0, 0)
                kwargs["radius"] = 2
                kwargs["size_limits"] = None
                kwargs["inner_radius"] = None
                kwargs["resizable"] = True
            elif type == "RingSelector":
                kwargs["center"] = (0, 0)
                kwargs["radius"] = 2
                kwargs["size_limits"] = None
                kwargs["inner_radius"] = 0
                kwargs["resizable"] = True
            else:
                raise ValueError("Invalid Selector Type")
        elif self.ndim == 1:
            type = "LineSelector"
            kwargs["selection"] = 0
            kwargs["size"] = np.max(self.fpl_image.data[:, 1])
            kwargs["center"] =np.max(self.fpl_image.data[:, 1])/2
            kwargs["limits"] = (self.fpl_image.data[0, 0],self.fpl_image.data[-1, 0])
            print("Line Selector Limits", kwargs["limits"])

        print("add Selector")
        selector = Selector(type=type,
                            parent=self.fpl_image,
                            integration_order=len(self.selector_list),
                            on_nav=not self.is_signal,
                            **kwargs)

        selector.is_live = True
        self.fpl_image._plot_area.add_graphic(selector.selector, center=False)
        self.selectors.append(selector)
        if down_step:
            print("Down Stepping")
            hypersignal = self.hyper_signal.parent_signal
            sel_list = self.selector_list + [selector,]
            print("Create Plot")
            new_plot = Plot(hypersignal,
                        is_signal=True,
                        selector_list=sel_list,
                        )  # flip/flop for back and forth
            for s in new_plot.selector_list:  # this plot will update with all new selectors?
                s.plots.append(new_plot)
        else:
            new_plot = Plot(self.hyper_signal,
                            is_signal=False,
                            selector_list=[selector,],
                            key_navigator=True
                            )  # flip/flop for back and fortj
            selector.is_live = False
            selector.is_integrating = True
        selector.plots.append(new_plot)
        new_plot.main_window = self.main_window
        self.main_window.add_plot(new_plot)
        selector.add_event_handler(selector.update_data, "selection")
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
        histogram_figure[0, 0].add_graphic(histogram_widget, )
        histogram_figure[0, 0].axes.visible = False
        histogram_figure.show()

    def get_context_menu(self, ev):
        if ev.button == 2:  # Right click
            context_menu = QtWidgets.QMenu()
            selector_menu = context_menu.addMenu("Add Selector")
            down_step = self.hyper_signal.parent_signal is not None
            if self.ndim == 1:
                action = selector_menu.addAction("Add Line Selector")
                partial_add_selector = partial(self.add_selector_and_new_plot, type="LineSelector",
                                               down_step=down_step)
                action.triggered.connect(partial_add_selector)
            else:  # 2D
                action = selector_menu.addAction("Add Rectangle Selector and New Plot")
                partial_add_selector = partial(self.add_selector_and_new_plot,
                                               type="RectangleSelector",
                                               down_step=down_step)
                action.triggered.connect(partial_add_selector)
                action = selector_menu.addAction("Add Circle Selector and New Plot")
                partial_add_selector = partial(self.add_selector_and_new_plot, type="CircleSelector",
                                               down_step=down_step)
                action.triggered.connect(partial_add_selector)
                action = selector_menu.addAction("Add Ring Selector and New Plot")
                partial_add_selector = partial(self.add_selector_and_new_plot,
                                               type="RingSelector",
                                               down_step=down_step)
                function_menu = context_menu.addMenu("Functions")
                action = function_menu.addAction("Center Zero Beam")


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

    def compute_data(self, reverse=True):
        """
        Compute the virtual image from the "selection".
        """
        self.current_indexes = [s.get_selected_indices() for s in self.selector_list]
        self.current_indexes_dense = self.get_dense_indexes()
        indexes = self.get_dense_indexes()
        parent_signal = self.hyper_signal
        if parent_signal.parent_signal is not None:
            while parent_signal.parent_signal is not None:
                parent_signal = parent_signal.parent_signal
        signal = parent_signal.signal
        result = fast_index_virtual(signal.data, indexes, reverse=reverse)
        if isinstance(result, da.Array):
            lazy_arr = result  # make non blocking
            print(f"{time.time()}:Computing Virtual Image")
            tic = time.time()
            data = self.hyper_signal.client.compute(lazy_arr)
            print(f"{time.time()-tic}:Virtual Image Computation submit time")
            self.data = data
        else:
            self.data = result
            self.update()

    def get_dense_indexes(self):
        """
        Get the current indexes for the plot.  This is used to update the plot when the
        selector is moved.
        """
        indexes = None
        for item in self.current_indexes:
            item = np.array(item)
            if item.ndim == 1:
                item = item[:, None]
            if indexes is None:
                indexes = item
            else:
                indexes = np.hstack(
                    [np.repeat(indexes, len(item), axis=0), np.repeat(item, len(indexes), axis=0)])
        return indexes

    def update_plot(self, get_result=False):
        indexes = self.get_dense_indexes()
        if (self.current_indexes_dense.shape != indexes.shape or not
        np.array_equal(self.current_indexes_dense, indexes)):
            self.current_indexes_dense = indexes
            if self.hyper_signal.signal._lazy:
                tic = time.time()
                self.hyper_signal.signal.cached_dask_array.cache_padding = 1
                print(f"getting current image from {self.hyper_signal.signal}")
                current_img = self.hyper_signal.signal._get_cache_dask_chunk(indexes,
                                                                             get_result=get_result)
                print(f"{time.time()-tic}: Signal Image submit time")
            else:
                tuple_inds = tuple([indexes[:, ind]
                                    for ind in np.arange(indexes.shape[1])])
                current_img = np.sum(self.hyper_signal.signal.data[tuple_inds], axis=0)

            self.data = current_img
            if not isinstance(current_img, Future): # update immediately otherwise send to event loop
                self.update()

    def update(self):
        """
        Update the plot with the new data.
        """
        if self.data.ndim == 1:
            data = np.vstack([self.hyper_signal.signal.axes_manager.signal_axes[0].axis,
                              self.data,
                              np.zeros_like(self.data)]).T
            self.fpl_image.data = data
            self.fpl_fig[0, 0].auto_scale()
        else:
            self.fpl_image.data = self.data
            self.fpl_image.reset_vmin_vmax()

    def closeEvent(self, event):
        # Add your custom close logic here
        super().closeEvent(event)
        if self.selector_list != []:
            for s in self.selector_list:
                s.remove_event_handler("selection")
                s.visible = False
                s.widget.hide()
