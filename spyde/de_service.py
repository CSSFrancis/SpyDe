from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QTextEdit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QMessageBox
from PyQt6 import QtCore
import numpy as np
from PyQt6.QtWidgets import QSlider, QLabel
from PyQt6.QtCore import Qt
import deapi
import time

class AcquiringThread(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):
        self.main_window.text_output.append("Starting Acquiring Thread")
        while self.main_window.client.acquiring:
            time.sleep(.1)
        self.main_window.update_plots()
        if self.main_window.start_button.text() =="Stop":
            self.main_window.start_button.setText("Get Raw")
        #self.finished.emit()

class BadPixelCorrectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bad Pixel Correction")
        self.resize(1300, 500)

        # Create the big plot
        self.big_plot_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.big_plot_axes = self.big_plot_canvas.figure.add_subplot(111)
        self.big_plot_axes.set_title("Big Plot")

        # Create smaller plots
        self.small_plot_axes = []
        self.small_plot_canvases = FigureCanvas(Figure(figsize=(5, 7)))

        for i in range(6):
            self.small_plot_axes.append(self.small_plot_canvases.figure.add_subplot(2, 3, i+1))
            self.small_plot_axes[i].imshow(np.ones((10, 10)))
            self.small_plot_axes[i].set_yticks([])
            self.small_plot_axes[i].set_xticks([])
        self.small_plot_axes[0].set_title("Uncorrected")
        self.small_plot_axes[1].set_title("Mask")
        self.small_plot_axes[2].set_title("Corrected")

        # Create text output panel
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Output messages will appear here...")

        # Create range slider
        self.range_slider_min = QSlider(Qt.Orientation.Vertical)
        self.range_slider_min.setMinimum(0)
        self.range_slider_min.setMaximum(100)
        self.range_slider_min.setValue(20)
        self.range_slider_min.valueChanged.connect(self.update_range_labels)

        self.range_slider_max = QSlider(Qt.Orientation.Vertical)
        self.range_slider_max.setMinimum(0)
        self.range_slider_max.setMaximum(100)
        self.range_slider_max.setValue(80)
        self.range_slider_max.valueChanged.connect(self.update_range_labels)

        self.range_label_min = QLabel("Min: 20")
        self.range_label_max = QLabel("Max: 80")
        self.range_label_max.setMinimumWidth(50)
        self.range_label_min.setMinimumWidth(50)

        self.acquiring_thread = QtCore.QThread()
        self.acquiring_worker = AcquiringThread(self)
        self.acquiring_worker.moveToThread(self.acquiring_thread)

        # Set layout
        main_layout = QVBoxLayout()
        all_plots_layout = QHBoxLayout()
        all_plots_layout.addWidget(self.text_output)
        plots_layout = QHBoxLayout()
        # Add big plot
        plots_layout.addWidget(self.big_plot_canvas)
        plots_layout.addWidget(self.small_plot_canvases)

        # Add smaller plots
        all_plots_layout.addLayout(plots_layout)

        # Add range slider to layout
        slider_layout = QHBoxLayout()
        slider_layout_vbox = QVBoxLayout()
        slider_layout_vbox.addWidget(self.range_slider_min)
        slider_layout_vbox.addWidget(self.range_label_min)
        slider_layout.addLayout(slider_layout_vbox)
        slider_layout_vbox2 = QVBoxLayout()
        slider_layout_vbox2.addWidget(self.range_slider_max)
        slider_layout_vbox2.addWidget(self.range_label_max)
        slider_layout.addLayout(slider_layout_vbox2)
        all_plots_layout.addLayout(slider_layout)

        main_layout.addLayout(all_plots_layout)

        # Set central widget
        container = QWidget()

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Get Raw")
        self.start_button.clicked.connect(self.toggle_start_stop_bad_pixel)
        self.start_button.setCheckable(True)
        button_layout.addWidget(self.start_button)

        self.roi_button = QPushButton("New ROI")
        self.roi_button.clicked.connect(self.push_roi_button)
        button_layout.addWidget(self.roi_button)

        self.test_bad_pixel = QPushButton("Test Bad Pixel Mask")
        self.test_bad_pixel.clicked.connect(self.test_bad_pixel_func)
        button_layout.addWidget(self.test_bad_pixel)

        self.save_correction = QPushButton("Save Bad Pixel Correction")
        self.save_correction.clicked.connect(self.save_correction_func)
        button_layout.addWidget(self.save_correction)

        main_layout.addLayout(button_layout)

        # Add the start/stop button to the layout
        container.setLayout(main_layout)

        self.setCentralWidget(container)
        self.show()

        self.client = deapi.Client()
        self.client.connect()
        self.text_output.append(f"Connected to Client:{self.client}")

    def start_thread(self):

        self.acquiring_thread.started.connect(self.acquiring_worker.run)
        #self.acquiring_thread.finished.connect(self.acquiring_thread.quit)
        self.text_output.append(f"Starting Thread{self.acquiring_thread}")
        self.acquiring_thread.start()

    def update_plots(self):
        self.text_output.append("Updating Plots")
        image, pix, attr, hist = self.client.get_result()
        self.big_plot_axes.imshow(image)
        self.big_plot_axes.set_yticks([])
        self.big_plot_axes.set_xticks([])
        self.text_output.append("Plots updated...")
        self.big_plot_canvas.draw_idle()



    def update_range_labels(self):
        self.range_label_min.setText(f"Min: {self.range_slider_min.value()}")
        self.range_label_max.setText(f"Max: {self.range_slider_max.value()}")

    def push_roi_button(self):
        self.text_output.append("Getting 4 new ROIs")
        # Add logic for getting new ROIs here

    def test_bad_pixel_func(self):
        self.test_bad_pixel.setText("Stop")
        self.text_output.append("Creating Bad Pixel Mask")
        self.text_output.append("Setting Apply Bad Pixel Correction To True")
        self.text_output.append("Acquiring a flat field image for 40 sec")

    def save_correction_func(self):
        self.text_output.append("Saving Bad Pixel Correction")
        # Add logic for saving the correction here

    def toggle_start_stop_bad_pixel(self):
        if self.start_button.text() == "Get Raw":
            self.start_button.setText("Stop")
            # Show a pop-up message
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Flat Illumination Warning")
            msg_box.setText(
                "Make sure there is a flat illumination on the detector of somewhere from 5-30 electrons/pixel/sec")
            msg_box.exec()
            self.text_output.append("Starting Bad Pixel Correction...")
            loc_time = time.localtime()
            self.text_output.append("Setting output Directory")

            self.client["Autosave Directory"] = f"D:\\Service\\{loc_time.tm_year}-{loc_time.tm_mon}-{loc_time.tm_mday}"
            self.client["Autosave Final Image"] = "On"
            num_frames = 15
            self.client["Autosave Movie Sum Count"] = num_frames
            self.client["Frame Count"] = num_frames
            self.text_output.append(f"Starting Acquiring: For {num_frames/60} sec")
            self.client.start_acquisition(1)
            self.start_thread()

        else:
            self.start_button.setText("Get Raw")
            self.text_output.append("Stopping Bad Pixel Correction...")
            # Add logic for stopping the process here

    def test_bad_pixel(self):
        self.text_output.append("Setting Apply Bad Pixel Correction To False")
        acquisition_time = 40
        self.text_output.append(f"Acquiring a flat field image for {acquisition_time} sec")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Tools")

        # Create buttons
        self.bad_pixel_button = QPushButton("Bad Pixel Correction")
        self.magnification_button = QPushButton("Magnification Calibration")

        # Connect buttons to their respective methods
        self.bad_pixel_button.clicked.connect(self.bad_pixel_correction)
        self.magnification_button.clicked.connect(self.magnification_calibration)

        # Set layout
        layout = QHBoxLayout()
        layout.addWidget(self.bad_pixel_button)
        layout.addWidget(self.magnification_button)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.bad_pixel_window = None
        self.mag_calibration_window = None

    def bad_pixel_correction(self):
        self.close()
        self.bad_pixel_window = BadPixelCorrectionWindow()

    def magnification_calibration(self):
        print("Magnification Calibration clicked")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()