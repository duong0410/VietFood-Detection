"""
Configuration file for VietFood Detection App
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, "result_train", "result_4", "vietfood68", "yolo11s_vietfood", "weights", "best.pt")
YAML_PATH = os.path.join(BASE_DIR, "result_train", "result_4", "vietfood68.yaml")

# Detection parameters
CONF_THRESHOLD = 0.25  # Confidence threshold
IOU_THRESHOLD = 0.45   # IoU threshold for NMS
MAX_DET = 300          # Maximum detections per image

# Class names (68 Vietnamese food categories)
CLASS_NAMES = [
    'Banh canh', 'Banh chung', 'Banh cuon', 'Banh khot', 'Banh mi',
    'Banh trang', 'Banh trang tron', 'Banh xeo', 'Bo kho', 'Bo la lot',
    'Bong cai', 'Bun', 'Bun bo Hue', 'Bun cha', 'Bun dau',
    'Bun mam', 'Bun rieu', 'Ca', 'Ca chua', 'Ca phao',
    'Ca rot', 'Canh', 'Cha', 'Cha gio', 'Chanh',
    'Com', 'Com tam', 'Con nguoi', 'Cu kieu', 'Cua',
    'Dau hu', 'Dua chua', 'Dua leo', 'Goi cuon', 'Hamburger',
    'Heo quay', 'Hu tieu', 'Kho qua thit', 'Khoai tay chien', 'Lau',
    'Long heo', 'Mi', 'Muc', 'Nam', 'Oc',
    'Ot chuong', 'Pho', 'Pho mai', 'Rau', 'Salad',
    'Thit bo', 'Thit ga', 'Thit heo', 'Thit kho', 'Thit nuong',
    'Tom', 'Trung', 'Xoi', 'Banh beo', 'Cao lau',
    'Mi Quang', 'Com chien Duong Chau', 'Bun cha ca', 'Com chien ga', 'Chao long',
    'Nom hoa chuoi', 'Nui xao bo', 'Sup cua'
]

# UI Configuration
WINDOW_TITLE = "VietFood Detection System"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800

# Image display size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Supported image formats
SUPPORTED_FORMATS = [
    "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)",
    "All Files (*.*)"
]

# Supported video formats
SUPPORTED_VIDEO_FORMATS = [
    "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)",
    "All Files (*.*)"
]

# Video processing settings
DEFAULT_FPS = 30
SKIP_FRAMES = 0  # Set to 1 to process every other frame, 2 for every third, etc.
VIDEO_CODEC = 'mp4v'  # Video codec for output

# Webcam settings
DEFAULT_CAMERA_INDEX = 0
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
WEBCAM_FPS = 30

# Colors for bounding boxes (BGR format)
BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
