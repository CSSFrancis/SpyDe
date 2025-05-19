from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from qdarktheme import load_stylesheet
from PyQt6.QtGui import QCursor


class CustomSubWindow(QtWidgets.QMdiSubWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._moving = False
        self._move_start = None
        self._resize_start = None

        self.title_bar = QtWidgets.QWidget(self)
        self.title_bar.setFixedHeight(30)
        self.title_bar.setMinimumHeight(30)
        self.title_bar.setMinimumWidth(200)
        self.title_bar.setStyleSheet(load_stylesheet())
        self.title_bar_layout = QtWidgets.QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("Custom Window", self.title_bar)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_bar_layout.addWidget(self.title_label)
        self.minimize_button = QtWidgets.QPushButton('_', self.title_bar)
        self.minimize_button.setFixedSize(25, 25)
        self.minimize_button.clicked.connect(self.toggle_minimize)

        self.maximize_button = QtWidgets.QPushButton('■', self.title_bar)
        self.maximize_button.setCheckable(True)
        self.maximize_button.setChecked(False)
        self.maximize_button.setFixedSize(25, 25)
        self.maximize_button.clicked.connect(self.toggle_maximize)

        self.close_button = QtWidgets.QPushButton('X', self.title_bar)
        self.close_button.setFixedSize(25, 25)
        self.close_button.clicked.connect(self.close)

        self.title_bar.mousePressEvent = self.start_move
        self.title_bar.mouseMoveEvent = self.move_window
        self.title_bar.mouseReleaseEvent = self.end_move

        self.title_bar_layout.addWidget(self.minimize_button)
        self.title_bar_layout.addWidget(self.maximize_button)
        self.title_bar_layout.addWidget(self.close_button)
        self.title_bar_layout.setContentsMargins(0, 0, 5, 0)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.title_bar)
        self.layout().setSpacing(0)
        self.old_size = self.size()
        self.is_minimized = False
        self.setMouseTracking(True)
        self.title_bar.setMouseTracking(True)
        self.title_label.setMouseTracking(True)
        self.setMouseTracking(True)
        self.plot_widget = None
        self.installEventFilter(self)

        self._resizing_top = False
        self._resizing_bottom = False
        self._resizing_left = False
        self._resizing_right = False

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.MouseMove:
            self.mouseMoveEvent(event)
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
            self.mousePressEvent(event)
        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            self.mouseReleaseEvent(event)
        return super().eventFilter(obj, event)

    def toggle_minimize(self):
        # Minimize the window
        if self.is_minimized:
            self.resize(self.old_size)
            self.is_minimized = False
        else:
            self.is_minimized = True
            self.old_size = self.size()
            self.resize(QtCore.QSize(300, 30))

    def toggle_maximize(self):
        # Toggle between maximized and normal window states
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    @property
    def resizing(self):
        return self._resizing_top or self._resizing_bottom or self._resizing_left or self._resizing_right

    def start_move(self, event):
        # Start dragging the window
        if event.button() == Qt.MouseButton.LeftButton and not self.resizing:
            self._moving = True
            self._move_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def move_window(self, event):
        # Move the window while dragging
        if self._moving:
            self.move(event.globalPosition().toPoint() - self._move_start)

    def end_move(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._moving = False

    def mouseMoveEvent(self, event):
        margins = 10  # Resize margin
        rect = self.rect()
        pos = event.pos()

        if pos.x() < margins and pos.y() < margins:
            self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))  # Top-left corner
        elif pos.x() > rect.width() - margins and pos.y() < margins:
            self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))  # Top-right corner
        elif pos.x() < margins and pos.y() > rect.height() - margins-30:
            self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))  # Bottom-left corner
        elif pos.x() > rect.width() - margins and pos.y() > rect.height() - margins - 30:
            self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))  # Bottom-right corner
        elif pos.x() < margins:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))  # Left edge
        elif pos.x() > rect.width() - margins:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))  # Right edge
        elif pos.y() < margins:
            self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))  # Top edge
        elif pos.y() > rect.height() - margins - 30:
            self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))  # Bottom edge
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))  # Default cursor
        if self.resizing:
            # Handle resizing
            pos_int = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            new_width = self.width()
            new_height = self.height()
            new_x = self.x()
            new_y = self.y()
            if self._resizing_top:
                new_height = self.height() - (pos_int.y() - self._resize_start.y())
                new_y = self.y() + (pos_int.y() - self._resize_start.y())
            if self._resizing_bottom:
                new_height = self.height() + (pos_int.y() - self._resize_start.y())
            if self._resizing_left:
                new_width = self.width() - (pos_int.x() - self._resize_start.x())
                new_x = self.x() + (pos_int.x() - self._resize_start.x())
            if self._resizing_right:
                new_width = self.width() + (pos_int.x() - self._resize_start.x())

            if new_height < 60:
                new_height = 60
            if new_width < 100:
                new_width = 100
            self.resize(new_width, new_height)
            self.move(new_x, new_y)

            self._resize_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mousePressEvent(self, event):
        margins = 10  # Resize margin
        rect = self.rect()
        pos = event.pos()
        # Start resizing the window
        if event.button() == Qt.MouseButton.LeftButton:
            if pos.x() < margins and pos.y() < margins:
                self._resizing_top = True  # Top-left corner
                self._resizing_left = True
            elif pos.x() > rect.width() - margins and pos.y() < margins:
                self._resizing_top = True  # Top-right corner
                self._resizing_right = True
            elif pos.x() < margins and pos.y() > rect.height() - margins-30:
                self._resizing_bottom = True
                self._resizing_left = True  # Bottom-left corner
            elif pos.x() > rect.width() - margins and pos.y() > rect.height() - margins-30:
                self._resizing_bottom = True
                self._resizing_right = True  # Bottom-right corner
            elif pos.x() < margins:
                self._resizing_left = True
            elif pos.x() > rect.width() - margins:
                self._resizing_right = True
            elif pos.y() < margins:
                self._resizing_top = True
            elif pos.y() > rect.height() - margins-30:
                self._resizing_bottom = True

            if self.resizing:
                # Store the starting position for resizing
                self._resize_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseReleaseEvent(self, event):
        # Stop resizing the window
        if event.button() == Qt.MouseButton.LeftButton:
            self._resizing_top = False
            self._resizing_bottom = False
            self._resizing_left = False
            self._resizing_right = False
