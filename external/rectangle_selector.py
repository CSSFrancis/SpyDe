import warnings
from numbers import Real
from typing import *
import numpy as np

import pygfx
from fastplotlib.graphics._collection_base import GraphicCollection
from pygfx.geometries._cylinder import generate_cap, generate_torso, merge

from fastplotlib.graphics._base import Graphic
from fastplotlib.graphics.selectors._base_selector import BaseSelector

from typing import Sequence, Tuple

from fastplotlib.utils import mesh_masks
from fastplotlib.graphics._features._selection_features import GraphicFeature, FeatureEvent


class RectangleSelectionFeature(GraphicFeature):
    """
    **additional event attributes:**

    +----------------------+----------+------------------------------------+
    | attribute            | type     | description                        |
    +======================+==========+====================================+
    | get_selected_indices | callable | returns indices under the selector |
    +----------------------+----------+------------------------------------+
    | get_selected_data    | callable | returns data under the selector    |
    +----------------------+----------+------------------------------------+

    **info dict:**

    +-------------0--+------------+-------------------------------------------+
    |   dict key    | value type | value description                         |
    +===============+============+===========================================+
    | value        | np.ndarray | new [xmin, xmax, ymin, ymax] of selection |
    +-------------+------------+-------------------------------------------+
    | limits      | np.ndarray | new [xmin, xmax, ymin, ymax] for the ind |
    +------------+------------+-------------------------------------------+
    | size_limits| np.ndarray | new [xmin, xmax, ymin, ymax] for the ind |
    +------------+------------+-------------------------------------------+
    """

    def __init__(
        self,
        value: tuple[float, float, float, float],
        limits: tuple[float, float, float, float],
        size_limits: tuple[float, float, float, float]=None,
    ):
        super().__init__()

        self._limits = limits
        self._value = tuple(int(v) for v in value)
        self._size_limits = size_limits

    @property
    def value(self) -> np.ndarray[float]:
        """
        (xmin, xmax, ymin, ymax) of the selection, in data space
        """
        return self._value


    @property
    def width(self) -> float:
        """
        width of the selection
        """
        return self._value[1] - self._value[0]

    @property
    def height(self) -> float:
        """
        height of the selection
        """
        return self._value[3] - self._value[2]

    def set_value(self, selector, value: Sequence[float]):
        """
        Set the selection of the rectangle selector.

        Parameters
        ----------
        selector: RectangleSelector

        value: (float, float, float, float)
            new values (xmin, xmax, ymin, ymax) of the selection
        """
        if not len(value) == 4:
            raise TypeError(
                "Selection must be an array, tuple, list, or sequence in the form of `(xmin, xmax, ymin, ymax)`, "
                "where `xmin`, `xmax`, `ymin`, `ymax` are numeric values."
            )

        # convert to array
        value = np.asarray(value, dtype=np.float32)

        # clip values if they are beyond the limits but force the rectangle to
        # be the same size as before
        width = self.width
        value[:2] = value[:2].clip(self._limits[0], self._limits[1])
        if value[0] == self._limits[0]:
            value[1] = value[0] + width
        if value[1] == self._limits[1]:
            value[0] = value[1] - width
        # clip y
        height = self.height
        value[2:] = value[2:].clip(self._limits[2], self._limits[3])
        if value[2] == self._limits[2]:
            value[3] = value[2] + height
        if value[3] == self._limits[3]:
            value[2] = value[3] - height

        xmin, xmax, ymin, ymax = value

        # make sure `selector width >= 2` and selector height >=2 , left edge must not move past right edge!
        # or bottom edge must not move past top edge!
        if not (xmax - xmin) >= 0 or not (ymax - ymin) >= 0:
            return

        # change fill mesh
        # change left x position of the fill mesh
        selector.fill.geometry.positions.data[mesh_masks.x_left] = xmin

        # change right x position of the fill mesh
        selector.fill.geometry.positions.data[mesh_masks.x_right] = xmax

        # change bottom y position of the fill mesh
        selector.fill.geometry.positions.data[mesh_masks.y_bottom] = ymin

        # change top position of the fill mesh
        selector.fill.geometry.positions.data[mesh_masks.y_top] = ymax

        # change the edge lines

        # each edge line is defined by two end points which are stored in the
        # geometry.positions
        # [x0, y0, z0]
        # [x1, y1, z0]

        # left line
        z = selector.edges[0].geometry.positions.data[:, -1][0]
        selector.edges[0].geometry.positions.data[:] = np.array(
            [[xmin, ymin, z], [xmin, ymax, z]]
        )

        # right line
        selector.edges[1].geometry.positions.data[:] = np.array(
            [[xmax, ymin, z], [xmax, ymax, z]]
        )

        # bottom line
        selector.edges[2].geometry.positions.data[:] = np.array(
            [[xmin, ymin, z], [xmax, ymin, z]]
        )

        # top line
        selector.edges[3].geometry.positions.data[:] = np.array(
            [[xmin, ymax, z], [xmax, ymax, z]]
        )

        # change the vertex positions

        # bottom left
        selector.vertices[0].geometry.positions.data[:] = np.array([[xmin, ymin, 1]])

        # bottom right
        selector.vertices[1].geometry.positions.data[:] = np.array([[xmax, ymin, 1]])

        # top left
        selector.vertices[2].geometry.positions.data[:] = np.array([[xmin, ymax, 1]])

        # top right
        selector.vertices[3].geometry.positions.data[:] = np.array([[xmax, ymax, 1]])

        self._value = value

        # send changes to GPU
        selector.fill.geometry.positions.update_range()

        for edge in selector.edges:
            edge.geometry.positions.update_range()

        for vertex in selector.vertices:
            vertex.geometry.positions.update_range()

        # send event
        if len(self._event_handlers) < 1:
            return

        event = FeatureEvent("selection", {"value": self.value})

        event.get_selected_indices = selector.get_selected_indices
        event.get_selected_data = selector.get_selected_data

        # calls any events
        self._call_event_handlers(event)


class CircleSelectionFeature(GraphicFeature):
    """
    **additional event attributes:**

    +----------------------+----------+------------------------------------+
    | attribute            | type     | description                        |
    +======================+==========+====================================+
    | get_selected_indices | callable | returns indices under the selector |
    +----------------------+----------+------------------------------------+
    | get_selected_data    | callable | returns data under the selector    |
    +----------------------+----------+------------------------------------+

    **info dict:**

    +-------------0--+------------+-------------------------------------------+
    |   dict key    | value type | value description                         |
    +===============+============+===========================================+
    | value        | np.ndarray | new [centerx, centery, radius, inner_radius] of selection |
    +-------------+------------+-------------------------------------------+
    | limits      | np.ndarray | new [xmin, xmax, ymin, ymax] for the ind |
    +------------+------------+-------------------------------------------+
    | size_limits| np.ndarray | new [xmin, xmax, ymin, ymax] for the ind |
    +------------+------------+-------------------------------------------+
    """

    def __init__(
        self,
        selection: tuple[float, float, float, float],
        limits: tuple[float, float, float, float],
        has_inner_radius: bool,
        size_limits: tuple[float, float, float, float]=None,
    ):
        super().__init__()

        self._limits = limits
        self._selection = tuple(int(v) for v in selection)
        self._size_limits = size_limits
        self._has_inner_radius = has_inner_radius

    @property
    def selection(self) -> np.ndarray[float]:
        """
        (centerx, centery, radius, inner_radius) of the selection, in data space
        """
        return self._selection


    def set_selection(self, selector, selection: Sequence[float]):
        """
        Set the selection of the rectangle selector.

        Parameters
        ----------
        selector: RectangleSelector

        selection: (float, float, float, float)
            new values (centerx, centery, radius, inner_radius) of the selection
        """
        if not len(selection) == 4:
            raise TypeError(
                "Selection must be an array, tuple, list, or sequence in the form of `(xmin, xmax, ymin, ymax)`, "
                "where `xmin`, `xmax`, `ymin`, `ymax` are numeric values."
            )

        # convert to array
        selection = np.asarray(selection, dtype=np.float32)

        # clip the center values if they are outside of the selection.
        # the radius should be allowed to move beyond the limits
        selection[0] = selection[0].clip(self._limits[0], self._limits[1])
        selection[1] = selection[1].clip(self._limits[2], self._limits[3])


        x, y, radius, inner_radius = selection

        # make sure that the radius is greater than 0 and the inner radius is less than the radius
        if radius <= 0 or inner_radius> radius:
            return
        radial_segments = 360

        cap = generate_cap(radius=radius, height=1, radial_segments=radial_segments, theta_start=0,
                           theta_length=np.pi * 2)

        cap_outline = generate_cap(radius=radius, height=1.01, radial_segments=radial_segments, theta_start=0,
                                   theta_length=np.pi * 2)

        if self._has_inner_radius:

            inner_cap_outline = generate_cap(radius=inner_radius, height=1.01, radial_segments=radial_segments, theta_start=0,
                                     theta_length=np.pi * 2)

            positions, normals, texcoords, indices = inner_cap_outline

            selector.edges[1].geometry.positions.data[:] = positions[1:] + np.array([x, y, 0])
            selector.vertices[1].geometry.positions.data[:] = np.array([[x + inner_radius, y, 1.1]])


        positions, normals, texcoords, indices = cap

        selector.fill.geometry.positions.data[:] = positions + np.array([x, y, 0])
        selector.vertices[0].geometry.positions.data[:] = np.array([[x+radius, y, 1.1]])

        positions, normals, texcoords, indices = cap_outline
        # shift the circle to the correct position

        selector.edges[0].geometry.positions.data[:] = positions[1:] + np.array([x, y, 0])


        # change the edge lines
        self._selection = selection

        # send changes to GPU
        selector.fill.geometry.positions.update_range()

        for edge in selector.edges:
            edge.geometry.positions.update_range()

        for vertex in selector.vertices:
            vertex.geometry.positions.update_range()

        # send event
        if len(self._event_handlers) < 1:
            return

        event = FeatureEvent("selection", {"value": self.selection})

        event.get_selected_indices = selector.get_selected_indices
        event.get_selected_data = selector.get_selected_data

        # calls any events
        self._call_event_handlers(event)


class RectangleSelector(BaseSelector):
    @property
    def parent(self) -> Graphic | None:
        """Graphic that selector is associated with."""
        return self._parent

    @property
    def selection(self) -> np.ndarray[float]:
        """
        (xmin, xmax, ymin, ymax) of the rectangle selection
        """
        return self._selection.value

    @selection.setter
    def selection(self, selection: Sequence[float]):
        # set (xmin, xmax, ymin, ymax) of the selector in data space
        graphic = self._parent

        if isinstance(graphic, GraphicCollection):
            pass

        self._selection.set_value(self, selection)

    @property
    def limits(self) -> Tuple[float, float, float, float]:
        """Return the limits of the selector."""
        return self._limits

    @limits.setter
    def limits(self, values: Tuple[float, float, float, float]):
        if len(values) != 4 or not all(map(lambda v: isinstance(v, Real), values)):
            raise TypeError("limits must be an iterable of two numeric values")
        self._limits = tuple(
            map(round, values)
        )  # if values are close to zero things get weird so round them
        self._selection._limits = self._limits

    def __init__(
        self,
        selection: Sequence[float],
        limits: Sequence[float],
        parent: Graphic = None,
        resizable: bool = True,
        fill_color=(0, 0, 0.35),
        edge_color=(0.8, 0.6, 0),
        edge_thickness: float = 8,
        vertex_color=(0.7, 0.4, 0),
        vertex_thickness: float = 8,
        arrow_keys_modifier: str = "Shift",
        size_limits: Sequence[float]=None,
        name: str = None,
    ):
        """
        Create a RectangleSelector graphic which can be used to select a rectangular region of data.
        Allows sub-selecting data from a ``Graphic`` or from multiple Graphics.

        Parameters
        ----------
        selection: (float, float, float, float)
            the initial selection of the rectangle, ``(x_min, x_max, y_min, y_max)``

        limits: (float, float, float, float)
            limits of the selector, ``(x_min, x_max, y_min, y_max)``

        parent: Graphic, default ``None``
            associate this selector with a parent Graphic

        resizable: bool, default ``True``
            if ``True``, the edges can be dragged to resize the selection

        fill_color: str, array, or tuple
            fill color for the selector, passed to pygfx.Color

        edge_color: str, array, or tuple
            edge color for the selector, passed to pygfx.Color

        edge_thickness: float, default 8
            edge thickness

        arrow_keys_modifier: str
            modifier key that must be pressed to initiate movement using arrow keys, must be one of:
            "Control", "Shift", "Alt" or ``None``

        name: str
            name for this selector graphic
        """

        if not len(selection) == 4 or not len(limits) == 4:
            raise ValueError()

        # lots of very close to zero values etc. so round them
        selection = tuple(map(round, selection))
        limits = tuple(map(round, limits))

        self._parent = parent
        self._limits = np.asarray(limits)
        self._resizable = resizable

        selection = np.asarray(selection)

        # world object for this will be a group
        # basic mesh for the fill area of the selector
        # line for each edge of the selector
        group = pygfx.Group()

        xmin, xmax, ymin, ymax = selection

        self._fill_color = pygfx.Color(fill_color)
        self._edge_color = pygfx.Color(edge_color)
        self._vertex_color = pygfx.Color(vertex_color)

        width = xmax - xmin
        height = ymax - ymin

        if width < 0 or height < 0:
            raise ValueError()

        self.fill = pygfx.Mesh(
            pygfx.box_geometry(width, height, 1),
            pygfx.MeshBasicMaterial(
                color=pygfx.Color(self.fill_color), pick_write=True
            ),
        )

        self.fill.world.position = (0, 0, -2)
        self.fill.world.z = -0.2
        group.add(self.fill)

        # position data for the left edge line
        left_line_data = np.array(
            [
                [xmin, ymin, 0],
                [xmin, ymax, 0],
            ]
        ).astype(np.float32)

        left_line = pygfx.Line(
            pygfx.Geometry(positions=left_line_data.copy()),
            pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color),
        )

        # position data for the right edge line
        right_line_data = np.array(
            [
                [xmax, ymin, 0],
                [xmax, ymax, 0],
            ]
        ).astype(np.float32)

        right_line = pygfx.Line(
            pygfx.Geometry(positions=right_line_data.copy()),
            pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color),
        )

        # position data for the left edge line
        bottom_line_data = np.array(
            [
                [xmin, ymax, 0],
                [xmax, ymax, 0],
            ]
        ).astype(np.float32)

        bottom_line = pygfx.Line(
            pygfx.Geometry(positions=bottom_line_data.copy()),
            pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color),
        )

        # position data for the right edge line
        top_line_data = np.array(
            [
                [xmin, ymin, 0],
                [xmax, ymin, 0],
            ]
        ).astype(np.float32)

        top_line = pygfx.Line(
            pygfx.Geometry(positions=top_line_data.copy()),
            pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color),
        )

        self.edges: Tuple[pygfx.Line, pygfx.Line, pygfx.Line, pygfx.Line] = (
            left_line,
            right_line,
            bottom_line,
            top_line,
        )  # left line, right line, bottom line, top line

        # add the edge lines
        for edge in self.edges:
            edge.world.z = -0.5
            group.add(edge)

        # vertices
        top_left_vertex_data = (xmin, ymax, 1)
        top_right_vertex_data = (xmax, ymax, 1)
        bottom_left_vertex_data = (xmin, ymin, 1)
        bottom_right_vertex_data = (xmax, ymin, 1)

        top_left_vertex = pygfx.Points(
            pygfx.Geometry(positions=[top_left_vertex_data], sizes=[vertex_thickness]),
            pygfx.PointsMarkerMaterial(
                marker="square",
                size=vertex_thickness,
                color=self.vertex_color,
                size_mode="vertex",
                edge_color=self.vertex_color,
            ),
        )

        top_right_vertex = pygfx.Points(
            pygfx.Geometry(positions=[top_right_vertex_data], sizes=[vertex_thickness]),
            pygfx.PointsMarkerMaterial(
                marker="square",
                size=vertex_thickness,
                color=self.vertex_color,
                size_mode="vertex",
                edge_color=self.vertex_color,
            ),
        )

        bottom_left_vertex = pygfx.Points(
            pygfx.Geometry(
                positions=[bottom_left_vertex_data], sizes=[vertex_thickness]
            ),
            pygfx.PointsMarkerMaterial(
                marker="square",
                size=vertex_thickness,
                color=self.vertex_color,
                size_mode="vertex",
                edge_color=self.vertex_color,
            ),
        )

        bottom_right_vertex = pygfx.Points(
            pygfx.Geometry(
                positions=[bottom_right_vertex_data], sizes=[vertex_thickness]
            ),
            pygfx.PointsMarkerMaterial(
                marker="square",
                size=vertex_thickness,
                color=self.vertex_color,
                size_mode="vertex",
                edge_color=self.vertex_color,
            ),
        )

        self.vertices: Tuple[pygfx.Points, pygfx.Points, pygfx.Points, pygfx.Points] = (
            bottom_left_vertex,
            bottom_right_vertex,
            top_left_vertex,
            top_right_vertex,
        )

        for vertex in self.vertices:
            vertex.world.z = -0.25
            group.add(vertex)

        self._selection = RectangleSelectionFeature(selection, limits=self._limits,
                                                     size_limits=size_limits)

        # include parent offset
        if parent is not None:
            offset = (parent.offset[0], parent.offset[1], 0)
        else:
            offset = (0, 0, 0)

        BaseSelector.__init__(
            self,
            edges=self.edges,
            fill=(self.fill,),
            vertices=self.vertices,
            hover_responsive=(*self.edges, *self.vertices),
            arrow_keys_modifier=arrow_keys_modifier,
            parent=parent,
            name=name,
            offset=offset,
        )

        self._set_world_object(group)

        self.selection = selection

    @property
    def size_limits(self) -> Tuple[float, float, float, float]:
        """Return the size limits of the selector."""
        return self._selection._size_limits

    @size_limits.setter
    def size_limits(self, values: Tuple[float, float, float, float]):
        if len(values) != 4 or not all(map(lambda v: isinstance(v, Real), values)):
            raise TypeError("size_limits must be an iterable of two numeric values")
        self._selection._size_limits = tuple(
            map(round, values)
        )

    def get_selected_data(
        self, graphic: Graphic = None, mode: str = "full"
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get the ``Graphic`` data bounded by the current selection.
        Returns a view of the data array.

        If the ``Graphic`` is a collection, such as a ``LineStack``, it returns a list of views of the full array.
        Can be performed on the ``parent`` Graphic or on another graphic by passing to the ``graphic`` arg.

        Parameters
        ----------
        graphic: Graphic, optional, default ``None``
            if provided, returns the data selection from this graphic instead of the graphic set as ``parent``
        mode: str, default 'full'
            One of 'full', 'partial', or 'ignore'. Indicates how selected data should be returned based on the
            selectors position over the graphic. Only used for ``LineGraphic``, ``LineCollection``, and ``LineStack``
            | If 'full', will return all data bounded by the x and y limits of the selector even if partial indices
            along one axis are not fully covered by the selector.
            | If 'partial' will return only the data that is bounded by the selector, missing indices not bounded by the
            selector will be set to NaNs
            | If 'ignore', will only return data for graphics that have indices completely bounded by the selector

        Returns
        -------
        np.ndarray or List[np.ndarray]
            view or list of views of the full array, returns empty array if selection is empty
        """
        source = self._get_source(graphic)
        ixs = self.get_selected_indices(source)

        # do not need to check for mode for images, because the selector is bounded by the image shape
        # will always be `full`
        if "Image" in source.__class__.__name__:
            row_ixs, col_ixs = ixs
            row_slice = slice(row_ixs[0], row_ixs[-1] + 1)
            col_slice = slice(col_ixs[0], col_ixs[-1] + 1)

            return source.data[row_slice, col_slice]

        if mode not in ["full", "partial", "ignore"]:
            raise ValueError(
                f"`mode` must be one of 'full', 'partial', or 'ignore', you have passed {mode}"
            )
        if "Line" in source.__class__.__name__:

            if isinstance(source, GraphicCollection):
                data_selections: List[np.ndarray] = list()

                for i, g in enumerate(source.graphics):
                    # want to keep same length as the original line collection
                    if ixs[i].size == 0:
                        data_selections.append(
                            np.array([], dtype=np.float32).reshape(0, 3)
                        )
                    else:
                        # s gives entire slice of data along the x
                        s = slice(
                            ixs[i][0], ixs[i][-1] + 1
                        )  # add 1 because these are direct indices
                        # slices n_datapoints dim

                        # calculate missing ixs using set difference
                        # then calculate shift
                        missing_ixs = (
                            np.setdiff1d(np.arange(ixs[i][0], ixs[i][-1] + 1), ixs[i])
                            - ixs[i][0]
                        )

                        match mode:
                            # take all ixs, ignore missing
                            case "full":
                                data_selections.append(g.data[s])
                            # set missing ixs data to NaNs
                            case "partial":
                                if len(missing_ixs) > 0:
                                    data = g.data[s].copy()
                                    data[missing_ixs] = np.nan
                                    data_selections.append(data)
                                else:
                                    data_selections.append(g.data[s])
                            # ignore lines that do not have full ixs to start
                            case "ignore":
                                if len(missing_ixs) > 0:
                                    data_selections.append(
                                        np.array([], dtype=np.float32).reshape(0, 3)
                                    )
                                else:
                                    data_selections.append(g.data[s])
                return data_selections
            else:  # for lines
                if ixs.size == 0:
                    # empty selection
                    return np.array([], dtype=np.float32).reshape(0, 3)

                s = slice(
                    ixs[0], ixs[-1] + 1
                )  # add 1 to end because these are direct indices
                # slices n_datapoints dim
                # slice with min, max is faster than using all the indices

                # get missing ixs
                missing_ixs = np.setdiff1d(np.arange(ixs[0], ixs[-1] + 1), ixs) - ixs[0]

                match mode:
                    # return all, do not care about missing
                    case "full":
                        return source.data[s]
                    # set missing to NaNs
                    case "partial":
                        if len(missing_ixs) > 0:
                            data = source.data[s].copy()
                            data[missing_ixs] = np.nan
                            return data
                        else:
                            return source.data[s]
                    # missing means nothing will be returned even if selector is partially over data
                    # warn the user and return empty
                    case "ignore":
                        if len(missing_ixs) > 0:
                            warnings.warn(
                                "You have selected 'ignore' mode. Selected graphic has incomplete indices. "
                                "Move the selector or change the mode to one of `partial` or `full`."
                            )
                            return np.array([], dtype=np.float32)
                        else:
                            return source.data[s]

    def get_selected_indices(
        self, graphic: Graphic = None
    ) -> np.ndarray | tuple[np.ndarray]:
        """
        Returns the indices of the ``Graphic`` data bounded by the current selection.

        These are the data indices which correspond to the data under the selector.

        Parameters
        ----------
        graphic: Graphic, default ``None``
            If provided, returns the selection indices from this graphic instrad of the graphic set as ``parent``

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            data indicies of the selection
            | tuple of [row_indices, col_indices] if the graphic is an image
            | list of indices along the x-dimension for each line if graphic is a line collection
            | array of indices along the x-dimension if graphic is a line
        """
        # get indices from source
        source = self._get_source(graphic)

        # selector (xmin, xmax, ymin, ymax) values
        xmin, xmax, ymin, ymax = self.selection

        # image data does not need to check for mode because the selector is always bounded
        # to the image
        if "Image" in source.__class__.__name__:
            col_ixs = np.arange(xmin, xmax, dtype=int)
            row_ixs = np.arange(ymin, ymax, dtype=int)
            return row_ixs, col_ixs

        if "Line" in source.__class__.__name__:
            if isinstance(source, GraphicCollection):
                ixs = list()
                for g in source.graphics:
                    data = g.data.value
                    g_ixs = np.where(
                        (data[:, 0] >= xmin - g.offset[0])
                        & (data[:, 0] <= xmax - g.offset[0])
                        & (data[:, 1] >= ymin - g.offset[1])
                        & (data[:, 1] <= ymax - g.offset[1])
                    )[0]
                    ixs.append(g_ixs)
            else:
                # map only this graphic
                data = source.data.value
                ixs = np.where(
                    (data[:, 0] >= xmin)
                    & (data[:, 0] <= xmax)
                    & (data[:, 1] >= ymin)
                    & (data[:, 1] <= ymax)
                )[0]

            return ixs

    def _move_graphic(self, delta: np.ndarray):

        # new selection positions
        xmin_new = self.selection[0] + delta[0]
        xmax_new = self.selection[1] + delta[0]
        ymin_new = self.selection[2] + delta[1]
        ymax_new = self.selection[3] + delta[1]

        # move entire selector if source is fill
        if self._move_info.source == self.fill:
            # set thew new bounds
            self._selection.set_value(self, (xmin_new, xmax_new, ymin_new, ymax_new))
            return

        # if selector not resizable return
        if not self._resizable:
            return

        xmin, xmax, ymin, ymax = self.selection

        if self._move_info.source == self.vertices[0]:  # bottom left
            values = (xmin_new, xmax, ymin_new, ymax)
        elif self._move_info.source == self.vertices[1]:  # bottom right
            values = (xmin, xmax_new, ymin_new, ymax)
        elif self._move_info.source == self.vertices[2]:  # top left
            values = (xmin_new, xmax, ymin, ymax_new)
        elif self._move_info.source == self.vertices[3]:  # top right
            values = (xmin, xmax_new, ymin, ymax_new)
        # if event source was an edge and selector is resizable, move the edge that caused the event
        elif self._move_info.source == self.edges[0]:
            values = (xmin_new, xmax, ymin_new, ymax)
        elif self._move_info.source == self.edges[1]:
            values = (xmin, xmax_new, ymin_new, ymax)
        elif self._move_info.source == self.edges[2]:
            values = (xmin_new, xmax, ymin_new, ymax)
        elif self._move_info.source == self.edges[3]:
            values = (xmin, xmax_new, ymin, ymax)
        else:
            return

        if self.size_limits is not None:
            # check if the new selection is within the size limits
            if (
                    values[1] - values[0] < self.size_limits[0]
                    or values[1] - values[0] > self.size_limits[1]
                    or values[3] - values[2] < self.size_limits[2]
                    or values[3] - values[2] > self.size_limits[3]
            ):
                return
        self._selection.set_value(self, values)

    def _move_to_pointer(self, ev):
        pass


class CircleSelector(BaseSelector):

    def __init__(self,
                 center: Sequence[float],
                 radius: float,
                 limits: Sequence[float],
                 inner_radius: float = 0,
                 parent: Graphic = None,
                 resizable: bool = True,
                 fill_color=(0, 0, 0.35),
                 edge_color=(0.8, 0.6, 0),
                 vertex_color=(0.7, 0.4, 0),
                 edge_thickness: float = 4,
                 vertex_thickness: float = 12,
                 arrow_keys_modifier: str = "Shift",
                 size_limits: Sequence[float] = None,
                 name: str = None,
                 ):

        if not len(center) == 2:
                raise ValueError()

        # lots of very close to zero values etc. so round them
        center = tuple(map(round, center))
        radius = round(radius)
        if inner_radius is not None:
            inner_radius = round(inner_radius)

        self._parent = parent
        self._center = np.asarray(center)
        self._resizable = resizable
        self._limits = np.asarray(limits)
        self.size_limits = size_limits

        center = np.asarray(center)

        # world object for this will be a group
        # basic mesh for the fill area of the selector
        # line for each edge of the selector
        group = pygfx.Group()


        self._fill_color = pygfx.Color(fill_color)
        self._edge_color = pygfx.Color(edge_color)
        self._vertex_color = pygfx.Color(vertex_color)
        if radius < 0:
            raise ValueError()

        radial_segments = 360

        cap = generate_cap(radius=radius, height=1, radial_segments=radial_segments, theta_start=0,
                           theta_length=np.pi * 2)

        cap_outline = generate_cap(radius=radius, height=1.01, radial_segments=radial_segments, theta_start=0,
                                   theta_length=np.pi * 2)

        positions, normals, texcoords, indices = cap

        positions = positions + np.array([center[0], center[1], 0], dtype=np.float32)
        circle_geo = pygfx.Geometry(
            indices=indices.reshape((-1, 3)),
            positions=positions,
        )

        self.fill = pygfx.Mesh(circle_geo,
                               pygfx.MeshBasicMaterial(
                                   color=pygfx.Color(self.fill_color), pick_write=True
                               ),
                               )
        self.fill.world.position = (0, 0, -0.1)


        positions, normals, texcoords, indices = cap_outline
        positions = positions + np.array([center[0], center[1], 0], dtype=np.float32)

        positions_outline = positions[1:]
        circle_outline_geo = pygfx.Geometry(
            indices=indices.reshape((-1, 3))[1:],
            positions=positions_outline,
        )
        outline = pygfx.Line(
            geometry=circle_outline_geo,
            material=pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color)
        )
        vert = [center[0], center[1], 1]
        vert = np.array(vert) + [radius, 0, 0]
        vertex = pygfx.Points(
            pygfx.Geometry(positions=[vert], sizes=[vertex_thickness]),
            pygfx.PointsMarkerMaterial(
                marker="circle",
                size=vertex_thickness,
                color=self.vertex_color,
                size_mode="vertex",
                edge_color=self.vertex_color,
            ),
        )

        if inner_radius is not None:
            inner_cap = generate_cap(radius=inner_radius, height=1, radial_segments=radial_segments, theta_start=0,
                                     theta_length=np.pi * 2)

            inner_positions, _, _, inner_indices = inner_cap
            inner_positions = inner_positions + np.array([center[0], center[1], 0], dtype=np.float32)
            inner_positions_outline = inner_positions[1:]
            inner_circle_outline_geo = pygfx.Geometry(
                indices=inner_indices.reshape((-1, 3))[1:],
                positions=inner_positions_outline,
            )
            inner_outline = pygfx.Line(
                geometry=inner_circle_outline_geo,
                material=pygfx.LineMaterial(thickness=edge_thickness, color=self.edge_color)
            )
            inner_outline.world.z = -0.5
            group.add(inner_outline)
            self.edges = (outline, inner_outline)

            vert = [center[0], center[1], 1.1]
            vert = np.array(vert)
            inner_vertex = pygfx.Points(
                pygfx.Geometry(positions=[vert],
                               sizes=[vertex_thickness]),
                pygfx.PointsMarkerMaterial(
                    marker="circle",
                    size=vertex_thickness,
                    color=self.vertex_color,
                    size_mode="vertex",
                    edge_color=self.vertex_color,
                ),
            )
            self.vertices = (vertex, inner_vertex)
        else:
            self.edges = (outline,)
            self.vertices = (vertex,)

        for vertex in self.vertices:
            vertex.world.z = -0.25
            group.add(vertex)

        for outline in self.edges:
            outline.world.z = -0.5
            group.add(outline)

        group.add(self.fill)

        has_inner_radius = inner_radius is not None
        if not has_inner_radius:
            inner_radius = 0
        selection = np.asarray(tuple(center) + (radius, inner_radius))
        self._selection = CircleSelectionFeature(selection,
                                                 limits=self._limits,
                                                 size_limits=size_limits,
                                                 has_inner_radius=has_inner_radius)

        # include parent offset
        if parent is not None:
            offset = (parent.offset[0], parent.offset[1], 0)
        else:
            offset = (0, 0, 0)

        BaseSelector.__init__(
            self,
            edges=self.edges,
            vertices=self.vertices,
            fill=(self.fill,),
            hover_responsive=(*self.vertices,),
            arrow_keys_modifier=arrow_keys_modifier,
            parent=parent,
            name=name,
            offset=offset,
        )

        self._set_world_object(group)

        self.selection = selection

    def _move_graphic(self, delta: np.ndarray):

        # new selection positions
        centerx_new = self.selection[0] + delta[0]
        centery_new = self.selection[1] + delta[1]
        print(self._pygfx_event.pick_info["world_object"])
        print(self._pygfx_event.pick_info["world_object"] == self.vertices[1])
        if self._resizable:
            if self._pygfx_event.pick_info["world_object"] == self.vertices[0]:
                if delta[0] < 0 or delta[1] < 0:
                    radius_change = -np.sqrt(delta[0] ** 2 + delta[1] ** 2)
                else:
                    radius_change = np.sqrt(delta[0] ** 2 + delta[1] ** 2)

                new_radius = self.selection[2] + radius_change
                if new_radius < 0:
                    return
                if self.size_limits is not None:
                    if new_radius < self.size_limits[0] or new_radius > self.size_limits[1]:
                        return
                values = (self.selection[0], self.selection[1], self.selection[2] + radius_change, self.selection[3])
                self._selection.set_selection(self, values)
                self.selection = values
                return
            elif len(self.vertices) > 1 and self._pygfx_event.pick_info["world_object"] == self.vertices[1]:
                if delta[0] < 0 or delta[1] < 0:
                    radius_change = -np.sqrt(delta[0] ** 2 + delta[1] ** 2)
                else:
                    radius_change = np.sqrt(delta[0] ** 2 + delta[1] ** 2)

                new_radius = self.selection[3] + radius_change
                if new_radius <= 0 or new_radius > self.selection[2]:
                    return
                if self.size_limits is not None:
                    if new_radius < self.size_limits[0] or new_radius > self.size_limits[1]:
                        return
                values = (self.selection[0],
                          self.selection[1],
                          self.selection[2],
                          self.selection[3] + radius_change)
                self._selection.set_selection(self, values)
                self.selection = values
                return




        # move entire selector if source is fill
        if self._move_info.source == self.fill:
            # set thew new bounds
            self._selection.set_selection(self, (centerx_new, centery_new, self.selection[2], self.selection[3]))
            self.selection = (centerx_new, centery_new, self.selection[2], self.selection[3])

        # if selector not resizable return

    def get_selected_indices(
        self, graphic: Graphic = None
    ) -> np.ndarray | tuple[np.ndarray]:
        """
        Returns the indices of the ``Graphic`` data bounded by the current selection.

        These are the data indices which correspond to the data under the selector.

        Parameters
        ----------
        graphic: Graphic, default ``None``
            If provided, returns the selection indices from this graphic instrad of the graphic set as ``parent``

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            data indicies of the selection
            | tuple of [row_indices, col_indices] if the graphic is an image
            | list of indices along the x-dimension for each line if graphic is a line collection
            | array of indices along the x-dimension if graphic is a line
        """
        # get indices from source
        source = self._get_source(graphic)
        x_center, y_center, radius, inner_rad = self.selection

        if "Image" in source.__class__.__name__:
            xx = np.arange(-radius, radius, dtype=int)
            yy = np.arange(-radius, radius, dtype=int)
            inds = np.reshape(np.meshgrid(xx, yy), (2, -1))
            rad = np.linalg.norm(inds, axis=0)
            inds = inds[:, rad <= radius]
            inds[0] += int(x_center)
            inds[1] += int(y_center)
            g_ixs = np.where(
                (inds[:, 0] >= - source.offset[0])
                & (inds[:, 0] <= source.offset[0])
                & (inds[:, 1] >= source.offset[1])
                & (inds[:, 1] <= source.offset[1])
            )[0]
            inds = inds[g_ixs]
            return inds