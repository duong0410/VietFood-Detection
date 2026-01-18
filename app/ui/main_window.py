"""
Main Window UI for VietFood Detection Application
PyQt5 based GUI with image, video, and webcam detection support
"""
import sys
import os
import cv2
import time
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QTextEdit, QGroupBox, QScrollArea,
    QSlider, QApplication, QMessageBox, QSplitter,
    QFrame, QGridLayout, QTabWidget, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont

import numpy as np

# Import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.detector import FoodDetector
from app.utils.image_utils import load_image, resize_image, numpy_to_qimage
from app.utils.video_utils import VideoProcessor, WebcamCapture, VideoWriter
from app import config


class DetectionThread(QThread):
    """Worker thread for image detection"""
    finished = pyqtSignal(object, list)
    error = pyqtSignal(str)
    
    def __init__(self, detector, image):
        super().__init__()
        self.detector = detector
        self.image = image
    
    def run(self):
        try:
            results = self.detector.detect_from_array(self.image)
            detections = self.detector.parse_results(results)
            self.finished.emit(results, detections)
        except Exception as e:
            self.error.emit(str(e))


class VideoProcessingThread(QThread):
    """Worker thread for video processing"""
    frame_ready = pyqtSignal(np.ndarray, list, int)
    progress_update = pyqtSignal(float, int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, detector, video_path, save_path=None):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.save_path = save_path
        self.is_running = True
        self.is_paused = False
    
    def run(self):
        try:
            processor = VideoProcessor(self.video_path)
            writer = None
            
            if self.save_path:
                writer = VideoWriter(
                    self.save_path,
                    processor.fps,
                    (processor.width, processor.height)
                )
            
            frame_count = 0
            
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = processor.read_frame()
                if not ret:
                    break
                
                # Detect
                results = self.detector.detect_from_array(frame)
                detections = self.detector.parse_results(results)
                
                # Draw detections
                processed_frame = self.detector.draw_detections(
                    frame, detections, config.BOX_COLORS
                )
                
                # Write to output
                if writer:
                    writer.write_frame(processed_frame)
                
                # Emit signals
                self.frame_ready.emit(processed_frame, detections, frame_count)
                progress = processor.get_progress()
                self.progress_update.emit(progress, frame_count, processor.frame_count)
                
                frame_count += 1
            
            processor.release()
            if writer:
                writer.release()
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def stop(self):
        self.is_running = False


class WebcamThread(QThread):
    """Worker thread for webcam capture and detection"""
    frame_ready = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)
    
    def __init__(self, detector, camera_index=0):
        super().__init__()
        self.detector = detector
        self.camera_index = camera_index
        self.is_running = True
    
    def run(self):
        try:
            webcam = WebcamCapture(self.camera_index)
            if not webcam.open():
                self.error.emit("Cannot open webcam")
                return
            
            while self.is_running:
                ret, frame = webcam.read_frame()
                if not ret:
                    break
                
                # Detect
                results = self.detector.detect_from_array(frame)
                detections = self.detector.parse_results(results)
                
                # Draw detections
                processed_frame = self.detector.draw_detections(
                    frame, detections, config.BOX_COLORS
                )
                
                self.frame_ready.emit(processed_frame, detections)
                
                time.sleep(0.03)  # ~30 FPS
            
            webcam.release()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.detector = None
        self.current_image = None
        self.original_image = None
        self.detections = []
        
        self.detection_thread = None
        self.video_thread = None
        self.webcam_thread = None
        
        self.current_mode = "image"  # image, video, webcam
        
        self.init_ui()
        self.init_detector()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([900, 500])
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage('Sẵn sàng - Ready')
    
    def create_left_panel(self):
        """Create left panel with display and controls"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        
        self.image_mode_btn = QPushButton("📸 Chế độ Ảnh / Image Mode")
        self.image_mode_btn.setCheckable(True)
        self.image_mode_btn.setChecked(True)
        self.image_mode_btn.clicked.connect(lambda: self.switch_mode("image"))
        mode_layout.addWidget(self.image_mode_btn)
        
        self.video_mode_btn = QPushButton("🎬 Chế độ Video / Video Mode")
        self.video_mode_btn.setCheckable(True)
        self.video_mode_btn.clicked.connect(lambda: self.switch_mode("video"))
        mode_layout.addWidget(self.video_mode_btn)
        
        self.webcam_mode_btn = QPushButton("📹 Chế độ Webcam / Webcam Mode")
        self.webcam_mode_btn.setCheckable(True)
        self.webcam_mode_btn.clicked.connect(lambda: self.switch_mode("webcam"))
        mode_layout.addWidget(self.webcam_mode_btn)
        
        layout.addLayout(mode_layout)
        
        # Display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        self.image_label.setMinimumSize(600, 400)
        
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Progress bar (for video)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Control buttons
        self.create_control_buttons(layout)
        
        return panel
    
    def create_control_buttons(self, layout):
        """Create mode-specific control buttons"""
        # Image mode controls
        self.image_controls = QWidget()
        image_layout = QHBoxLayout()
        self.image_controls.setLayout(image_layout)
        
        self.upload_image_btn = QPushButton("📁 Tải ảnh / Upload Image")
        self.upload_image_btn.clicked.connect(self.upload_image)
        image_layout.addWidget(self.upload_image_btn)
        
        self.detect_image_btn = QPushButton("🔍 Nhận diện / Detect")
        self.detect_image_btn.setEnabled(False)
        self.detect_image_btn.clicked.connect(self.detect_image)
        image_layout.addWidget(self.detect_image_btn)
        
        layout.addWidget(self.image_controls)
        
        # Video mode controls
        self.video_controls = QWidget()
        self.video_controls.setVisible(False)
        video_layout = QHBoxLayout()
        self.video_controls.setLayout(video_layout)
        
        self.upload_video_btn = QPushButton("📁 Tải video / Upload Video")
        self.upload_video_btn.clicked.connect(self.upload_video)
        video_layout.addWidget(self.upload_video_btn)
        
        self.start_video_btn = QPushButton("▶️ Bắt đầu / Start")
        self.start_video_btn.setEnabled(False)
        self.start_video_btn.clicked.connect(self.start_video)
        video_layout.addWidget(self.start_video_btn)
        
        self.pause_video_btn = QPushButton("⏸️ Tạm dừng / Pause")
        self.pause_video_btn.setEnabled(False)
        self.pause_video_btn.clicked.connect(self.pause_video)
        video_layout.addWidget(self.pause_video_btn)
        
        self.stop_video_btn = QPushButton("⏹️ Dừng / Stop")
        self.stop_video_btn.setEnabled(False)
        self.stop_video_btn.clicked.connect(self.stop_video)
        video_layout.addWidget(self.stop_video_btn)
        
        self.save_video_btn = QPushButton("💾 Lưu kết quả / Save Result")
        self.save_video_btn.setEnabled(False)
        self.save_video_btn.clicked.connect(self.save_video_result)
        video_layout.addWidget(self.save_video_btn)
        
        layout.addWidget(self.video_controls)
        
        # Webcam mode controls
        self.webcam_controls = QWidget()
        self.webcam_controls.setVisible(False)
        webcam_layout = QHBoxLayout()
        self.webcam_controls.setLayout(webcam_layout)
        
        webcam_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        webcam_layout.addWidget(self.camera_combo)
        
        self.start_webcam_btn = QPushButton("▶️ Bắt đầu / Start")
        self.start_webcam_btn.clicked.connect(self.start_webcam)
        webcam_layout.addWidget(self.start_webcam_btn)
        
        self.stop_webcam_btn = QPushButton("⏹️ Dừng / Stop")
        self.stop_webcam_btn.setEnabled(False)
        self.stop_webcam_btn.clicked.connect(self.stop_webcam)
        webcam_layout.addWidget(self.stop_webcam_btn)
        
        self.capture_btn = QPushButton("📷 Chụp ảnh / Capture")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_frame)
        webcam_layout.addWidget(self.capture_btn)
        
        layout.addWidget(self.webcam_controls)
        
        # Common controls
        common_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ Xóa / Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        common_layout.addWidget(self.clear_btn)
        
        layout.addLayout(common_layout)
    
    def create_right_panel(self):
        """Create right panel with settings and results"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Settings
        settings_group = QGroupBox("Cài đặt / Settings")
        settings_layout = QGridLayout()
        settings_group.setLayout(settings_layout)
        
        settings_layout.addWidget(QLabel("Ngưỡng tin cậy / Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(config.CONF_THRESHOLD * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        settings_layout.addWidget(self.conf_slider, 0, 1)
        
        self.conf_label = QLabel(f"{config.CONF_THRESHOLD:.2f}")
        settings_layout.addWidget(self.conf_label, 0, 2)
        
        settings_layout.addWidget(QLabel("Ngưỡng IoU / IoU:"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(int(config.IOU_THRESHOLD * 100))
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        settings_layout.addWidget(self.iou_slider, 1, 1)
        
        self.iou_label = QLabel(f"{config.IOU_THRESHOLD:.2f}")
        settings_layout.addWidget(self.iou_label, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Results
        results_group = QGroupBox("Kết quả / Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont('Courier New', 9))
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Model info
        info_group = QGroupBox("Thông tin Model / Model Info")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setFont(QFont('Courier New', 8))
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        return panel
    
    def init_detector(self):
        """Initialize detector"""
        try:
            self.statusBar().showMessage('Đang tải model... / Loading model...')
            QApplication.processEvents()
            
            self.detector = FoodDetector(
                model_path=config.MODEL_PATH,
                conf_threshold=config.CONF_THRESHOLD,
                iou_threshold=config.IOU_THRESHOLD,
                max_det=config.MAX_DET
            )
            
            info = self.detector.get_model_info()
            info_text = f"Status: {info['status']}\n"
            info_text += f"Device: {info['device']}\n"
            info_text += f"Model: {os.path.basename(info['model_path'])}\n"
            info_text += f"Confidence: {info['conf_threshold']}\n"
            info_text += f"IoU: {info['iou_threshold']}"
            self.info_text.setText(info_text)
            
            self.statusBar().showMessage('Model đã sẵn sàng / Model ready')
            
        except Exception as e:
            self.statusBar().showMessage(f'Lỗi tải model / Error: {str(e)}')
            QMessageBox.critical(self, "Error", f"Không thể tải model:\n{str(e)}")
    
    def switch_mode(self, mode):
        """Switch between image/video/webcam modes"""
        self.current_mode = mode
        
        # Update button states
        self.image_mode_btn.setChecked(mode == "image")
        self.video_mode_btn.setChecked(mode == "video")
        self.webcam_mode_btn.setChecked(mode == "webcam")
        
        # Show/hide controls
        self.image_controls.setVisible(mode == "image")
        self.video_controls.setVisible(mode == "video")
        self.webcam_controls.setVisible(mode == "webcam")
        self.progress_bar.setVisible(mode == "video")
        
        # Stop any running processes
        self.stop_all_threads()
        self.clear_all()
        
        self.statusBar().showMessage(f'Chế độ: {mode.upper()}')
    
    def refresh_cameras(self):
        """Refresh available cameras"""
        self.camera_combo.clear()
        cameras = WebcamCapture.get_available_cameras()
        for cam in cameras:
            self.camera_combo.addItem(f"Camera {cam}", cam)
    
    def update_conf_label(self):
        value = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{value:.2f}")
        if self.detector:
            self.detector.conf_threshold = value
    
    def update_iou_label(self):
        value = self.iou_slider.value() / 100.0
        self.iou_label.setText(f"{value:.2f}")
        if self.detector:
            self.detector.iou_threshold = value
    
    # Image mode methods
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh / Select Image", "", ";;".join(config.SUPPORTED_FORMATS)
        )
        
        if file_path:
            try:
                self.original_image = load_image(file_path)
                self.current_image = self.original_image.copy()
                self.display_image(self.current_image)
                self.detect_image_btn.setEnabled(True)
                self.results_text.clear()
                self.statusBar().showMessage(f'Đã tải: {os.path.basename(file_path)}')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Không thể tải ảnh:\n{str(e)}")
    
    def detect_image(self):
        if self.original_image is None or self.detector is None:
            return
        
        self.detect_image_btn.setEnabled(False)
        self.statusBar().showMessage('Đang nhận diện... / Detecting...')
        
        self.detection_thread = DetectionThread(self.detector, self.original_image)
        self.detection_thread.finished.connect(self.on_image_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()
    
    def on_image_detection_finished(self, results, detections):
        self.detections = detections
        
        if len(detections) > 0:
            self.current_image = self.detector.draw_detections(
                self.original_image, detections, config.BOX_COLORS
            )
        else:
            self.current_image = self.original_image.copy()
        
        self.display_image(self.current_image)
        self.update_results_display()
        
        self.detect_image_btn.setEnabled(True)
        self.statusBar().showMessage(f'Phát hiện {len(detections)} món / Detected {len(detections)} items')
    
    # Video mode methods
    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video / Select Video", "", ";;".join(config.SUPPORTED_VIDEO_FORMATS)
        )
        
        if file_path:
            self.video_path = file_path
            self.start_video_btn.setEnabled(True)
            self.statusBar().showMessage(f'Đã tải video: {os.path.basename(file_path)}')
    
    def start_video(self):
        if not hasattr(self, 'video_path'):
            return
        
        self.start_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(True)
        self.upload_video_btn.setEnabled(False)
        
        self.video_thread = VideoProcessingThread(self.detector, self.video_path)
        self.video_thread.frame_ready.connect(self.on_video_frame_ready)
        self.video_thread.progress_update.connect(self.on_video_progress)
        self.video_thread.finished.connect(self.on_video_finished)
        self.video_thread.error.connect(self.on_detection_error)
        self.video_thread.start()
        
        self.statusBar().showMessage('Đang xử lý video... / Processing video...')
    
    def pause_video(self):
        if self.video_thread:
            if self.video_thread.is_paused:
                self.video_thread.resume()
                self.pause_video_btn.setText("⏸️ Tạm dừng / Pause")
            else:
                self.video_thread.pause()
                self.pause_video_btn.setText("▶️ Tiếp tục / Resume")
    
    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.reset_video_controls()
    
    def save_video_result(self):
        if not hasattr(self, 'video_path'):
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu video / Save Video", "", "MP4 Files (*.mp4);;All Files (*.*)"
        )
        
        if save_path:
            # Re-run with save path
            self.video_save_path = save_path
            self.start_video_btn.setEnabled(False)
            
            self.video_thread = VideoProcessingThread(
                self.detector, self.video_path, save_path
            )
            self.video_thread.frame_ready.connect(self.on_video_frame_ready)
            self.video_thread.progress_update.connect(self.on_video_progress)
            self.video_thread.finished.connect(self.on_video_save_finished)
            self.video_thread.error.connect(self.on_detection_error)
            self.video_thread.start()
            
            self.statusBar().showMessage('Đang lưu video... / Saving video...')
    
    def on_video_frame_ready(self, frame, detections, frame_num):
        self.detections = detections
        self.display_image(frame)
        self.update_results_display()
    
    def on_video_progress(self, progress, current_frame, total_frames):
        self.progress_bar.setValue(int(progress))
        self.statusBar().showMessage(
            f'Đang xử lý: {current_frame}/{total_frames} frames ({progress:.1f}%)'
        )
    
    def on_video_finished(self):
        self.reset_video_controls()
        self.save_video_btn.setEnabled(True)
        self.statusBar().showMessage('Hoàn thành xử lý video / Video processing complete')
    
    def on_video_save_finished(self):
        self.reset_video_controls()
        QMessageBox.information(self, "Success", "Video đã được lưu thành công!")
        self.statusBar().showMessage('Video đã được lưu / Video saved')
    
    def reset_video_controls(self):
        self.start_video_btn.setEnabled(True)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)
        self.upload_video_btn.setEnabled(True)
        self.progress_bar.setValue(0)
    
    # Webcam mode methods
    def start_webcam(self):
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            QMessageBox.warning(self, "Warning", "Không tìm thấy camera!")
            return
        
        self.start_webcam_btn.setEnabled(False)
        self.stop_webcam_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        
        self.webcam_thread = WebcamThread(self.detector, camera_index)
        self.webcam_thread.frame_ready.connect(self.on_webcam_frame_ready)
        self.webcam_thread.error.connect(self.on_detection_error)
        self.webcam_thread.start()
        
        self.statusBar().showMessage('Webcam đang chạy... / Webcam running...')
    
    def stop_webcam(self):
        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread.wait()
        
        self.start_webcam_btn.setEnabled(True)
        self.stop_webcam_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        self.statusBar().showMessage('Webcam đã dừng / Webcam stopped')
    
    def on_webcam_frame_ready(self, frame, detections):
        self.detections = detections
        self.current_image = frame.copy()
        self.display_image(frame)
        self.update_results_display()
    
    def capture_frame(self):
        if self.current_image is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh / Save Image", filename, "JPEG Files (*.jpg);;PNG Files (*.png)"
        )
        
        if save_path:
            cv2.imwrite(save_path, self.current_image)
            QMessageBox.information(self, "Success", f"Đã lưu: {os.path.basename(save_path)}")
    
    # Common methods
    def display_image(self, image):
        display_image = resize_image(
            image, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT, keep_aspect_ratio=True
        )
        qimage = numpy_to_qimage(display_image)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
    
    def update_results_display(self):
        if not self.detections:
            self.results_text.setText("Không phát hiện món ăn nào.\nNo food items detected.")
            return
        
        results_text = f"{'='*50}\n"
        results_text += f"  PHÁT HIỆN {len(self.detections)} MÓN ĂN\n"
        results_text += f"  DETECTED {len(self.detections)} FOOD ITEMS\n"
        results_text += f"{'='*50}\n\n"
        
        class_counts = {}
        for det in self.detections:
            cls_name = det['class_name']
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        for i, (cls_name, count) in enumerate(sorted(class_counts.items()), 1):
            results_text += f"{i}. {cls_name}: {count} món\n"
        
        self.results_text.setText(results_text)
    
    def on_detection_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Lỗi:\n{error_msg}")
        self.statusBar().showMessage('Lỗi / Error')
        self.reset_all_controls()
    
    def stop_all_threads(self):
        """Stop all running threads"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.wait()
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread.wait()
    
    def reset_all_controls(self):
        """Reset all controls to default state"""
        self.detect_image_btn.setEnabled(False)
        self.reset_video_controls()
        self.start_webcam_btn.setEnabled(True)
        self.stop_webcam_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
    
    def clear_all(self):
        """Clear everything"""
        self.stop_all_threads()
        
        self.current_image = None
        self.original_image = None
        self.detections = []
        
        self.image_label.clear()
        self.image_label.setText("Chưa có dữ liệu / No data")
        self.results_text.clear()
        
        self.reset_all_controls()
        self.statusBar().showMessage('Đã xóa / Cleared')
    
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_all_threads()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
