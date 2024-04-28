from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QFileDialog, QSizePolicy, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
# from functions import apply_hough_transform, apply_harris_corner_transform  # Assuming you have these functions
import cv2
from functions import apply_canny, hough_transform_visual , harris_corner_visual

class HoughLineDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing - Hough Line Detection and Harris Corner Detection')
        self.setFixedSize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.init_tabs()
        self.loaded_image = None
        self.loaded_image_tab2 = None

    def init_tabs(self):
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        self.create_hough_line_tab()
        self.create_harris_corner_tab()

        self.tabs.addTab(self.tab1, "Hough Line Detection")
        self.tabs.addTab(self.tab2, "Harris Corner Detection")
        self.setCentralWidget(self.tabs)

    def create_hough_line_tab(self):
        self.tab1_layout = QVBoxLayout(self.tab1)
        
        # Top layout with browse and apply buttons
        self.top_layout = QHBoxLayout()
        self.browse_button = QPushButton('Browse', self)
        self.apply_button = QPushButton('Apply', self)
        self.top_layout.addWidget(self.browse_button)
        self.top_layout.addWidget(self.apply_button)
        
        # Layout for images and their labels
        self.images_layout = QHBoxLayout()
        self.original_image_layout = QVBoxLayout()
        
        # Original image title
        self.original_label = QLabel('Original')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(350, 350)
        self.original_image_label.setStyleSheet("background-color: black")  # Remove this line
        self.original_image_layout.addWidget(self.original_label)
        self.original_image_layout.addWidget(self.original_image_label)
        
        # Detected (Result) image and label
        self.detected_image_layout = QVBoxLayout()
        self.detected_label = QLabel('Result')
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.detected_lines_image_label = QLabel()
        self.detected_lines_image_label.setAlignment(Qt.AlignCenter)
        self.detected_lines_image_label.setFixedSize(350, 350)
        self.detected_lines_image_label.setStyleSheet("background-color: white")  # Remove this line`
        self.detected_image_layout.addWidget(self.detected_label)
        self.detected_image_layout.addWidget(self.detected_lines_image_label)
        
        # Add image layouts to the images layout
        self.images_layout.addLayout(self.original_image_layout)
        self.images_layout.addLayout(self.detected_image_layout)

        # Controls layout
        self.controls_layout = QGridLayout()

        # Theta input below the original image
        self.resolution_input = QLineEdit()
        self.controls_layout.addWidget(QLabel('Resolution'), 1, 0)
        self.controls_layout.addWidget(self.resolution_input, 2, 0)


        # Number of lines input centered below Theta and Rho
        self.threshold_input = QLineEdit()
        self.controls_layout.addWidget(QLabel('Threshold'), 1, 1)
        self.controls_layout.addWidget(self.threshold_input, 2, 1)

        # Combine layouts into tab 1 layout
        self.tab1_layout.addLayout(self.top_layout)
        self.tab1_layout.addLayout(self.images_layout)
        self.tab1_layout.addLayout(self.controls_layout)
        self.browse_button.clicked.connect(self.browse_image)
        self.apply_button.clicked.connect(self.apply_hough_transform)  # Should correctly reference `apply_hough_transform`
        self.tab1.setLayout(self.tab1_layout)

    def create_harris_corner_tab(self):
        self.tab2_layout = QVBoxLayout(self.tab2)

        self.top_layout_tab2 = QHBoxLayout()
        self.browse_button_tab2 = QPushButton('Browse', self)
        self.apply_button_tab2 = QPushButton('Apply', self)
        self.top_layout_tab2.addWidget(self.browse_button_tab2)
        self.top_layout_tab2.addWidget(self.apply_button_tab2)

        self.images_layout_tab2 = QHBoxLayout()
        self.original_image_layout_tab2 = QVBoxLayout()
        self.original_label_tab2 = QLabel('Original')
        self.original_label_tab2.setAlignment(Qt.AlignCenter)
        self.original_label_tab2.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.original_image_label_tab2 = QLabel()
        self.original_image_label_tab2.setAlignment(Qt.AlignCenter)
        self.original_image_label_tab2.setFixedSize(350, 350)
        self.original_image_label_tab2.setStyleSheet("background-color: black")
        self.original_image_layout_tab2.addWidget(self.original_label_tab2)
        self.original_image_layout_tab2.addWidget(self.original_image_label_tab2)

        self.detected_image_layout_tab2 = QVBoxLayout()
        self.detected_label_tab2 = QLabel('Result')
        self.detected_label_tab2.setAlignment(Qt.AlignCenter)
        self.detected_label_tab2.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.detected_image_label_tab2 = QLabel()
        self.detected_image_label_tab2.setAlignment(Qt.AlignCenter)
        self.detected_image_label_tab2.setFixedSize(350, 350)
        self.detected_image_label_tab2.setStyleSheet("background-color: white")
        self.detected_image_layout_tab2.addWidget(self.detected_label_tab2)
        self.detected_image_layout_tab2.addWidget(self.detected_image_label_tab2)

        self.images_layout_tab2.addLayout(self.original_image_layout_tab2)
        self.images_layout_tab2.addLayout(self.detected_image_layout_tab2)

        self.controls_layout_tab2 = QVBoxLayout()
        self.threshold_input_tab2 = QLineEdit()
        self.controls_layout_tab2.addWidget(QLabel('Threshold'))
        self.controls_layout_tab2.addWidget(self.threshold_input_tab2)

        self.tab2_layout.addLayout(self.top_layout_tab2)
        self.tab2_layout.addLayout(self.images_layout_tab2)
        self.tab2_layout.addLayout(self.controls_layout_tab2)

        self.browse_button_tab2.clicked.connect(self.browse_image)
        self.apply_button_tab2.clicked.connect(self.apply_harris_corner_transform)
        self.tab2.setLayout(self.tab2_layout)

    def browse_image(self):
        # Check if there's already an open dialog
        if hasattr(self, '_is_browsing') and self._is_browsing:
            return  # Skip if a dialog is already open
        
        self._is_browsing = True
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        self._is_browsing = False  # Reset the flag after the dialog closes

        if filename:
            current_tab = self.tabs.currentIndex()
            if current_tab == 0:  # Hough Line Detection tab
                self.load_image_to_label(filename, self.original_image_label)
            elif current_tab == 1:  # Harris Corner Detection tab
                self.load_image_to_label(filename, self.original_image_label_tab2)


    def load_image_to_label(self, image_path, label):
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.information(self, "Error", "Failed to load image. Check the file path and format.")
            return

        # Depending on the label, assign to the correct attribute
        if label == self.original_image_label:
            self.loaded_image = image
        elif label == self.original_image_label_tab2:
            self.loaded_image_tab2 = image

        self.display_image(image, label)
# Update UI with the loaded image



    def display_image(self, image, label):
        # Convert to QImage and then to QPixmap
        if image.ndim == 3:  # Color image
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).rgbSwapped()
        else:  # Grayscale image
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)

        # Convert to QPixmap for displaying
        pixmap = QPixmap.fromImage(qimage)

        # Scale pixmap to fit label's width while maintaining aspect ratio
        pixmap = pixmap.scaledToWidth(label.width(), Qt.SmoothTransformation)

        # Calculate the scaled pixmap's height to adjust the label accordingly
        scaled_height = pixmap.height() if pixmap.height() <= label.height() else label.height()
        
        # Adjust label's fixed size to match scaled pixmap's width and calculated height
        label.setFixedSize(label.width(), scaled_height)
        label.setPixmap(pixmap)



    def apply_hough_transform(self):
        if self.loaded_image is None:
            QMessageBox.information(self, "Error", "No image loaded!")
        
            return
       
        resoultion = int(self.resolution_input.text() or "2")
        threshold = float(self.threshold_input.text() or "0.2")
        result_image = hough_transform_visual(self.loaded_image.copy(), resoultion , threshold)
        self.display_image(result_image, self.detected_lines_image_label)


    def apply_harris_corner_transform(self):
        if self.loaded_image_tab2 is None:
            QMessageBox.information(self, "Error", "No image loaded!")
            return

        threshold = float(self.threshold_input_tab2.text() or "0.01")
        result_image = harris_corner_visual(self.loaded_image_tab2.copy(), k=0.05, threshold_ratio=threshold)
        self.display_image(result_image, self.detected_image_label_tab2)



if __name__ == '__main__':
    app = QApplication([])
    window = HoughLineDetectionUI()
    window.show()
    app.exec_()
