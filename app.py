import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QGroupBox, QComboBox, QGridLayout,
                             QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PIL import Image
from datetime import datetime
import torch

from models.model_loader import load_model, load_custom_unet, load_unet_resnet34, load_fpn_efficientnet
from utils.detector import PotholeDetector
from utils.danger_assessment import DangerAssessment
from mlflow_tracking.mlops import MLflowTracker
from config import Config


class PotholeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Pothole Detection System")
        self.setGeometry(50, 50, 1600, 900)

        # MLflow tracking
        self.mlflow = MLflowTracker()
        self.mlflow.start_run()

        # Models config
        self.models = {
            'DeepLabV3+ (Best)': ('models/checkpoints/deeplabv3plus_best.pth', load_model),
            'U-Net ResNet34': ('models/checkpoints/unet_resnet34.pth', load_unet_resnet34),
            'FPN EfficientNet': ('models/checkpoints/fpn_efficientnet.pth', load_fpn_efficientnet),
            'Custom U-Net': ('models/checkpoints/custom_unet.pth', load_custom_unet)
        }

        self.current_detector = None
        self.assessor = DangerAssessment()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0

        self.stats = {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel
        left = QVBoxLayout()

        # Model selection
        model_box = QGroupBox("🤖 AI Model")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.models.keys()))

        self.load_btn = QPushButton("📥 Load Model")
        self.load_btn.clicked.connect(self.load_model)

        self.model_status = QLabel("⚠️ No model loaded")
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_btn)
        model_layout.addWidget(self.progress)
        model_layout.addWidget(self.model_status)
        model_box.setLayout(model_layout)

        # Controls
        ctrl_box = QGroupBox("🎮 Detection Controls")
        ctrl_layout = QGridLayout()

        self.btn_img = QPushButton("📷 Upload Image")
        self.btn_img.clicked.connect(self.upload_image)
        self.btn_img.setEnabled(False)

        self.btn_vid = QPushButton("🎥 Upload Video")
        self.btn_vid.clicked.connect(self.upload_video)
        self.btn_vid.setEnabled(False)

        self.btn_cam = QPushButton("📹 Start Camera")
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_cam.setEnabled(False)

        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)

        ctrl_layout.addWidget(self.btn_img, 0, 0)
        ctrl_layout.addWidget(self.btn_vid, 0, 1)
        ctrl_layout.addWidget(self.btn_cam, 1, 0)
        ctrl_layout.addWidget(self.btn_stop, 1, 1)
        ctrl_box.setLayout(ctrl_layout)

        # Stats
        stats_box = QGroupBox("📊 Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel()
        self.update_stats_display()
        stats_layout.addWidget(self.stats_label)
        stats_box.setLayout(stats_layout)

        # Results
        results_box = QGroupBox("📋 Analysis Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_box.setLayout(results_layout)

        # MLflow
        mlflow_box = QGroupBox("🔬 MLOps Tracking")
        mlflow_layout = QVBoxLayout()
        self.mlflow_status = QLabel("✅ MLflow Active")
        self.btn_mlflow = QPushButton("📊 Open Dashboard")
        self.btn_mlflow.clicked.connect(self.open_mlflow)
        mlflow_layout.addWidget(self.mlflow_status)
        mlflow_layout.addWidget(self.btn_mlflow)
        mlflow_box.setLayout(mlflow_layout)

        left.addWidget(model_box)
        left.addWidget(ctrl_box)
        left.addWidget(stats_box)
        left.addWidget(results_box)
        left.addWidget(mlflow_box)

        # Right panel
        right = QVBoxLayout()
        title = QLabel("🖼️ Detection View")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #89b4fa;")

        self.image_label = QLabel("Load a model to begin")
        self.image_label.setMinimumSize(1000, 700)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #3d3d5c; border-radius: 10px; background-color: #1e1e2e;")

        right.addWidget(title)
        right.addWidget(self.image_label)

        main_layout.addLayout(left, 1)
        main_layout.addLayout(right, 2)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QGroupBox {
                background-color: #2d2d44;
                border: 2px solid #3d3d5c;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                color: #cdd6f4;
                font-weight: bold;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a6c8ff;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c6f85;
            }
            QLabel {
                color: #cdd6f4;
            }
            QTextEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 2px solid #3d3d5c;
                border-radius: 6px;
            }
            QComboBox {
                background-color: #2d2d44;
                color: #cdd6f4;
                border: 2px solid #3d3d5c;
                border-radius: 6px;
                padding: 5px;
            }
        """)

    def load_model(self):
        try:
            self.load_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setValue(30)
            self.model_status.setText("Loading...")
            QApplication.processEvents()

            model_name = self.model_combo.currentText()
            path, loader = self.models[model_name]

            self.progress.setValue(60)
            model = loader(path, Config.DEVICE)
            self.current_detector = PotholeDetector(model, Config.DEVICE)

            self.progress.setValue(100)
            self.model_status.setText(f"✅ {model_name} loaded!")

            # Enable controls
            self.btn_img.setEnabled(True)
            self.btn_vid.setEnabled(True)
            self.btn_cam.setEnabled(True)

            # Log to MLflow with comprehensive details
            self.mlflow.log_param("model_name", model_name)
            self.mlflow.log_param("device", str(Config.DEVICE))
            self.mlflow.log_param("model_path", path)
            self.mlflow.log_param("timestamp", datetime.now().isoformat())

            # Log model performance metrics from checkpoint
            try:
                checkpoint = torch.load(path, map_location=Config.DEVICE)
                if 'val_iou' in checkpoint:
                    self.mlflow.log_metric("model_val_iou", float(checkpoint['val_iou']))
                if 'val_accuracy' in checkpoint:
                    self.mlflow.log_metric("model_val_accuracy", float(checkpoint['val_accuracy']))
            except:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.model_status.setText("❌ Load failed")
            import traceback
            traceback.print_exc()
        finally:
            self.progress.setVisible(False)
            self.load_btn.setEnabled(True)

    def upload_image(self):
        if not self.current_detector:
            return

        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            img = np.array(Image.open(path).convert('RGB'))
            mask, overlay = self.current_detector.predict(img)
            dims = self.assessor.calculate_dimensions(mask, pixel_to_cm=0.5)
            assess = self.assessor.assess_danger(dims)

            if dims:
                self.update_stats(assess)
                self.mlflow.log_detection(path, dims, assess)
            else:
                # Log when no pothole found
                self.mlflow.log_detection(path, None, {'level': 'NONE', 'score': 0})

            self.display_results(overlay, dims, assess)

    def upload_video(self):
        if not self.current_detector:
            return

        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.timer.start(50)
            self.btn_stop.setEnabled(True)
            self.btn_vid.setEnabled(False)
            self.btn_cam.setEnabled(False)

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open camera")
                return

            self.timer.start(50)
            self.btn_cam.setText("⏸️ Pause")
            self.btn_stop.setEnabled(True)
            self.btn_img.setEnabled(False)
            self.btn_vid.setEnabled(False)
        else:
            self.stop_camera()

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.setText("📹 Start Camera")
        self.btn_stop.setEnabled(False)
        self.btn_img.setEnabled(True)
        self.btn_vid.setEnabled(True)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % 2 != 0:
                    return

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask, overlay = self.current_detector.predict(frame)
                dims = self.assessor.calculate_dimensions(mask)
                assess = self.assessor.assess_danger(dims)

                if dims:
                    self.update_stats(assess)
                    self.mlflow.log_detection("webcam_frame", dims, assess)

                self.display_results(overlay, dims, assess)

    def update_stats(self, assess):
        self.stats['total'] += 1
        level = assess['level'].lower()
        if level in self.stats:
            self.stats[level] += 1
        self.update_stats_display()

        # Log comprehensive stats to MLflow
        self.mlflow.log_stats(self.stats)

    def update_stats_display(self):
        txt = f"""
        <b>Total: {self.stats['total']}</b><br>
        <span style='color: #f38ba8;'>🔴 Critical: {self.stats['critical']}</span><br>
        <span style='color: #fab387;'>🟠 High: {self.stats['high']}</span><br>
        <span style='color: #f9e2af;'>🟡 Medium: {self.stats['medium']}</span><br>
        <span style='color: #a6e3a1;'>🟢 Low: {self.stats['low']}</span>
        """
        self.stats_label.setText(txt)

    def display_results(self, img, dims, assess):
        h, w, c = img.shape
        qt_img = QImage(img.data, w, h, c * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pix.scaled(1000, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if dims:
            color = {'CRITICAL': '#f38ba8', 'HIGH': '#fab387', 'MEDIUM': '#f9e2af', 'LOW': '#a6e3a1'}[assess['level']]
            txt = f"""
            <div style='color: {color}; font-size: 16px;'><b>⚠️ POTHOLE DETECTED</b></div><br>
            <b>Level:</b> <span style='color: {color};'>{assess['level']}</span><br>
            <b>Score:</b> {assess['score']}/100<br>
            <b>Width:</b> {dims['width_px']} px ({dims['width_cm']:.1f} cm)<br>
            <b>Height:</b> {dims['height_px']} px ({dims['height_cm']:.1f} cm)<br>
            <b>Area:</b> {dims['area_px']} px² ({dims['area_cm2']:.1f} cm²)<br>
            <b>Coverage:</b> {dims['coverage_percent']:.2f}%<br>
            <b>Action:</b> {assess['action']}
            """
        else:
            txt = "<div style='color: #a6e3a1;'><b>✅ No pothole detected</b></div>"

        self.results_text.setHtml(txt)

    def open_mlflow(self):
        import webbrowser
        import subprocess
        subprocess.Popen(['mlflow', 'ui'], shell=True)
        webbrowser.open('http://127.0.0.1:5000')

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.mlflow.end_run()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 10))
    window = PotholeDetectionApp()
    window.show()
    sys.exit(app.exec_())