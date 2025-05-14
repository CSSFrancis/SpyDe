
import sys

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QRadioButton,
    QButtonGroup, QLabel, QDockWidget, QLineEdit, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    import deapi
    DEAPI_INSTALLED = True
except ImportError:
    DEAPI_INSTALLED = False

def remove_all_widgets(layout):
    """Remove all widgets from a layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            remove_all_widgets(item.layout())


class CameraControlWidget(QWidget):

    def initialize(self):
        """Initialize the camera control widget by connecting to the camera."""
        remove_all_widgets(self.layout)
        self.layout.update()
        layout = self.layout
        font = QFont()
        font.setBold(True)

        # Insert Camera Button
        camera_control_layout = QHBoxLayout()

        self.insert_camera_button = QPushButton("Insert Camera")
        camera_control_layout.addWidget(self.insert_camera_button)
        self.insert_camera_button.setCheckable(True)  # Make the button toggleable
        self.insert_camera_button.toggled.connect(self.on_camera_button_toggled)

        self.cool_camera_button = QPushButton("Cool Camera")
        camera_control_layout.addWidget(self.cool_camera_button)
        layout.addLayout(camera_control_layout)
        self.cool_camera_button.setCheckable(True)
        self.cool_camera_button.toggled.connect(self.on_cool_camera_button_toggled)

        resolution_label = QLabel("Repeats:")
        resolution_label.setFont(font)
        layout.addWidget(resolution_label,  0, Qt.AlignmentFlag.AlignLeft)
        # Integer inputs
        row1 = QHBoxLayout()
        self.scan_repeats_input = self.create_labeled_spinbox("Scan:", row1)
        self.summed_frames_input = self.create_labeled_spinbox("Frames:", row1)
        self.scan_repeats_input.setValue(1)
        self.summed_frames_input.setValue(1)
        layout.addLayout(row1)
        resolution_label = QLabel("Scan Positions:")
        resolution_label.setFont(font)
        layout.addWidget(resolution_label,  0, Qt.AlignmentFlag.AlignLeft)
        row2 = QHBoxLayout()
        self.scan_x_positions_input = self.create_labeled_spinbox("X:", row2)
        self.scan_y_positions_input = self.create_labeled_spinbox("Y:", row2)
        self.scan_x_positions_input.setValue(128)
        self.scan_y_positions_input.setValue(128)

        layout.addLayout(row2)

        # Radio buttons for resolution
        resolution_label = QLabel("Resolution:")
        resolution_label.setFont(font)
        layout.addWidget(resolution_label)

        self.resolution_group = QButtonGroup(self)
        resolution_layout = QHBoxLayout()
        for resolution in [1024, 512, 256, 128, 64]:
            radio_button = QRadioButton(str(resolution))
            if resolution == 1024:
                radio_button.setChecked(True)
            self.resolution_group.addButton(radio_button)
            resolution_layout.addWidget(radio_button)

        layout.addLayout(resolution_layout)
        self.resolution_group.buttonClicked.connect(self.on_resolution_button_clicked)

        # Test, Acquire, and Initialize Buttons
        self.search_button = QPushButton("Search")
        self.acquire_button = QPushButton("Acquire")
        self.acquire_button.toggled.connect(self.on_acquire_button_clicked)
        self.acquire_button.setCheckable(True)


        self.initialize_button = QPushButton("Initialize")  # New Initialize button
        self.initialize_button.toggled.connect(self.on_initialize_button_clicked)

        self.search_button.setCheckable(True)
        self.search_button.toggled.connect(self.on_search_button_clicked)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.initialize_button)  # Add Initialize button to layout
        button_layout.addWidget(self.search_button)


        button_layout.addWidget(self.acquire_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Connect to the camera
        self.camera_connection_layout = QVBoxLayout()

        camera_connection_label = QLabel("Connect to Camera")
        camera_connection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_connection_layout.addWidget(camera_connection_label)

        ip_layout = QVBoxLayout()
        ip_layout.addSpacerItem(QSpacerItem(0, 2, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        ip_address_label = QLabel("IP Address:")
        ip_layout.addWidget(ip_address_label)
        self.camera_connection_button = QPushButton("Connect")
        camera_connection_inputs = QHBoxLayout()
        starting_ip = "127.0.0.1"
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(starting_ip)
        self.input_field.setText(starting_ip)
        camera_connection_inputs.addWidget(self.input_field)
        camera_connection_inputs.addWidget(self.camera_connection_button)
        ip_layout.addLayout(camera_connection_inputs)
        self.camera_connection_layout.addLayout(ip_layout)

        self.camera_connection_button.clicked.connect(self.on_connect_to_camera)

        self.layout.addLayout(self.camera_connection_layout)
        self.setLayout(self.layout)
        self.de_client = None


    def on_connect_to_camera(self):

        try:
            self.de_client = deapi.Client()
            self.de_client.connect(host=self.input_field.text())
            if not "win" in sys.platform:
                self.de_client.usingMmf = False
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return

        self.initialize()

    def on_resolution_button_clicked(self):
        """
        Slot to handle the resolution button click.
        This method is called when a resolution radio button is clicked.
        """
        # Get the selected resolution
        selected_button = self.resolution_group.checkedButton()
        if selected_button:
            resolution = int(selected_button.text())
            print(f"Selected resolution: {resolution}")

            self.de_client.set_adaptive_roi(size_x=resolution, size_y=resolution)

    def create_labeled_spinbox(self, label_text, parent_layout):
        """Helper method to create a labeled QSpinBox."""
        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        spinbox = QSpinBox()
        spinbox.setRange(0, 10000)  # Set a reasonable range for the spinbox

        container_layout.addWidget(label,  0, Qt.AlignmentFlag.AlignTop)
        container_layout.addWidget(spinbox,  0, Qt.AlignmentFlag.AlignTop)
        container.setLayout(container_layout)
        parent_layout.addWidget(container,  0, Qt.AlignmentFlag.AlignTop)

        return spinbox

    def on_camera_button_toggled(self, checked):
        """Slot to handle the toggled state of the Insert Camera button."""
        if checked:
            self.insert_camera_button.setText("Retract Camera")
            self.de_client["Camera Position Control"] = "Extend"
        else:
            self.insert_camera_button.setText("Insert Camera")
            self.de_client["Camera Position Control"] = "Retract"

    def on_cool_camera_button_toggled(self, checked):
        """Slot to handle the toggled state of the Cool Camera button."""
        if checked:
            self.cool_camera_button.setText("Warm Camera")
            self.de_client["Temperature - Control"] = "Cool Down"

        else:
            self.cool_camera_button.setText("Cool Camera")
            self.de_client["Temperature - Control"] = "Warm Up"


    def on_initialize_button_clicked(self):
        """Initialize a data object. This object is initially a set of dask `Futures`.
        This creates a placeholder for the `HyperSignal` object which will be
        constructed from a set of Futures.
        """
        


    def on_search_button_clicked(self, checked):
        """Slot to handle the search button click."""
        if checked:
            self.search_button.setText("Stop")
        else:
            self.search_button.setText("Search")

    def on_acquire_button_clicked(self, checked):
        """Slot to handle the Acquire button click."""
        # Implement the acquire functionality here
        if checked:
            self.acquire_button.setText("Stop")
            self.de_client.start_acquisition(1)
        else:
            self.acquire_button.setText("Acquire")
            self.de_client.stop_acquisition()





class CameraControlDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Camera Control", parent)

        # Set up the dockable widget
        self.camera_control_widget = CameraControlWidget()

        self.setWidget(self.camera_control_widget)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)