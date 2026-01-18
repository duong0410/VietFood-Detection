# VietFood Detection 🍜

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green)](https://pypi.org/project/PyQt5/)
[![YOLO](https://img.shields.io/badge/YOLO-v11s-red)](https://github.com/ultralytics/ultralytics)

**Hệ thống nhận diện món ăn Việt Nam tự động sử dụng Deep Learning và YOLO**  
*Automatic Vietnamese Food Detection System using Deep Learning & YOLO*

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)

---

##  Overview

VietFood-Detection là một dự án object detection dựa trên deep learning để tự động phát hiện và phân loại các món ăn Việt Nam trong ảnh, video và webcam theo thời gian thực.


### Goals
-  Xây dựng hệ thống tự động phát hiện và phân loại món ăn Việt với độ chính xác cao
-  Hỗ trợ nhiều nguồn đầu vào: ảnh, video, webcam
-  Ứng dụng thực tế: đánh giá dinh dưỡng, quảng bá ẩm thực Việt Nam
-  Giao diện người dùng thân thiện với PyQt5

---

## ✨ Features

### 🖼️ Giao diện Desktop Application
- Giao diện đồ họa với PyQt5
- Hỗ trợ **3 chế độ detection**:
  - **Image Mode**: Detect từ ảnh tĩnh
  - **Video Mode**: Detect từ video file
  - **Webcam Mode**: Detect real-time từ webcam
- Điều chỉnh tham số ngay trên giao diện (Confidence, IoU threshold)
- Hiển thị kết quả với bounding boxes và labels
- Lưu kết quả detection
- Giao diện song ngữ Việt-Anh

###  Model Detection
- Nhận diện **68 loại món ăn Việt Nam**
- Model: YOLOv11s (Small variant) - cân bằng tốc độ và độ chính xác
- Hỗ trợ GPU/CUDA để tăng tốc
- Xử lý đa luồng không lag giao diện

###  Vietnamese Food Categories (68 classes)
Bánh mì, Phở, Bún bò Huế, Bún chả, Bánh xèo, Cơm tấm, Gỏi cuốn, Chả giò, Bánh cuốn, Cao lầu, Mì Quảng, và nhiều món khác...

---

##  Project Structure

```
VietFood-Detection/
│
├── app/                              # Main application directory
│   ├── __init__.py
│   ├── config.py                     # Configuration file
│   ├── models/                       # Model modules
│   │   ├── __init__.py
│   │   └── detector.py               # YOLO detector wrapper
│   ├── ui/                           # User Interface
│   │   ├── __init__.py
│   │   └── main_window.py            # Main GUI window (PyQt5)
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── image_utils.py            # Image processing
│   │   └── video_utils.py            # Video & webcam processing
│   └── assets/                       # Resources (icons, images)
│
├── result/                           # Training results
│   └── result_4/
│       └── vietfood68/
│           └── yolo11s_vietfood/
│               └── weights/
│                   ├── best.pt       # Best model checkpoint
│                   └── last.pt       # Latest checkpoint
│
├── train-yolov11s.ipynb             # Training notebook (Kaggle)
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies
├── run_app.bat                       # Windows run script
├── run_app.sh                        # Linux/Mac run script
└── README.md                         # This file
```

---

##  Installation

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows/Linux/MacOS
- **GPU**: Optional (NVIDIA with CUDA recommended for better performance)
- **Webcam**: Optional (for webcam mode)

### Step 1: Clone Repository

```bash
git clone https://github.com/duong0410/VietFood-Detection.git
cd VietFood-Detection
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have NVIDIA GPU and want to use CUDA acceleration, install PyTorch with CUDA support:
```bash
# Visit https://pytorch.org/ for the appropriate command
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Model Checkpoint

Ensure the model file exists at:
```
result/result_4/vietfood68/yolo11s_vietfood/weights/best.pt
```

If your model is in a different location, update `MODEL_PATH` in `app/config.py`:
```python
MODEL_PATH = "path/to/your/model/best.pt"
```

---

##  Usage

### Running the Application

**Method 1: Python Command**
```bash
python main.py
```

**Method 2: Script Files**
- **Windows**: Double-click `run_app.bat`
- **Linux/Mac**: `chmod +x run_app.sh && ./run_app.sh`

### User Guide

#### 1.  Image Detection
1. Click **" Chế độ Ảnh / Image Mode"**
2. Click **" Tải ảnh / Upload Image"**
3. Adjust parameters (optional):
   - **Confidence**: Detection confidence threshold (0.01-1.00)
   - **IoU**: Intersection over Union threshold (0.01-1.00)
4. Click **" Nhận diện / Detect"**
5. View results with bounding boxes and labels

#### 2.  Video Detection
1. Click **" Chế độ Video / Video Mode"**
2. Click **" Tải video / Upload Video"**
3. Adjust parameters if needed
4. Click **" Bắt đầu / Start"** to begin processing
5. Use **" Tạm dừng / Pause"** and **"⏹️ Dừng / Stop"** to control playback
6. Click **" Lưu kết quả / Save Result"** to export processed video

#### 3.  Webcam Detection
1. Click **" Chế độ Webcam / Webcam Mode"**
2. Select camera from dropdown (if multiple cameras available)
3. Click **" Bắt đầu / Start"** to begin real-time detection
4. Click **" Dừng / Stop"** to stop camera
5. Take snapshots with **"📷 Chụp ảnh / Capture"**

---

##  Model Training

### Training Environment
- **Platform**: Kaggle Notebook with GPU support
- **Hardware**: NVIDIA GPU (Kaggle-provided)
- **Framework**: Ultralytics YOLO
- **Image Size**: 640 × 640
- **Strategy**: Fine-tuning from pretrained YOLOv11s weights
- **Checkpointing**: Resume training using `last.pt` due to Kaggle GPU time limits

### Training Objectives
1. **Dataset Preparation**
   - Use VietFood68 dataset
   - Verify annotation quality (YOLO format)
   - Analyze class distribution

2. **Model Training**
   - Train YOLOv11s architecture
   - Experiment with feature levels (P2)
   - Fine-tune hyperparameters
   - Monitor metrics: Loss, Precision, Recall, mAP

3. **Evaluation**
   - Validate on test set
   - Visualize training curves
   - Compare model variants

4. **Deployment**
   - Export model weights (`best.pt`)
   - Prepare for inference

### Training Notebook
See `train-yolov11s.ipynb` for the complete training pipeline.

---

##  Dataset

### VietFood67: Vietnamese Food Image Dataset

This project uses the **VietFood67 dataset**, publicly released by its original authors.

- **Images**: ~33,000 images
- **Categories**: 68 Vietnamese food classes
- **Annotations**: Bounding boxes in YOLO format
- **Purpose**: Food detection and recognition research
- **Application**: FoodDetector real-time dietary assessment system

**Dataset Features:**
- High-quality images with diverse angles and backgrounds
- Professional bounding box annotations
- Peer-reviewed and validated
- Used in multiple academic publications

---

##  Technical Details

### Technologies Used
- **Model**: YOLOv11s (Small variant)
- **Framework**: Ultralytics YOLO
- **GUI**: PyQt5
- **Image Processing**: OpenCV, Pillow
- **Deep Learning**: PyTorch
- **Video Processing**: OpenCV VideoCapture/VideoWriter

### Model Configuration
Edit `app/config.py` to customize:

```python
# Model paths
MODEL_PATH = "path/to/model.pt"

# Detection parameters
CONF_THRESHOLD = 0.25  # Default confidence threshold
IOU_THRESHOLD = 0.45   # Default IoU threshold
MAX_DET = 300          # Maximum detections per image

# UI settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
```

### Performance Optimization
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Multi-threading**: Separate threads for detection to prevent UI freezing
- **Frame Skip**: Configurable for video processing
- **Batch Processing**: Future enhancement for multiple images

---

##  Troubleshooting

### Common Issues

**1. "Model file not found"**
```bash
# Check model path in app/config.py
# Ensure best.pt exists in the specified directory
```

**2. PyQt5 Import Error**
```bash
pip uninstall PyQt5
pip install PyQt5==5.15.10
```

**3. CUDA Out of Memory**
- Reduce input image/video resolution
- Lower `MAX_DET` in config
- Use CPU mode (automatically selected if GPU unavailable)

**4. Webcam Not Detected**
```bash
# Check camera permissions
# Try different camera indices in the dropdown
# Restart the application
```

**5. Slow Performance**
- Ensure GPU and CUDA are properly installed
- Check GPU usage: `nvidia-smi` (Windows/Linux)
- Reduce video resolution or enable frame skipping

**6. Video Encoding Error**
```bash
# Install additional codecs
pip install opencv-contrib-python
```

---

## 📚 Citation

**Note:** This project uses the VietFood67 dataset created by Nguyen Viet Hoang Nam et al. We do not own the dataset. All credit goes to the original authors.

If you use this project or the VietFood67 dataset, please cite:

```bibtex
@inproceedings{vietfood67_soict2024,
  title={Now I Know What I Am Eating: Real-Time Tracking and Nutritional Insights Using VietFood67 to Enhance User Experience},
  author={Nguyen, Viet Hoang Nam and Tran, Bao Tu and Ton That, Minh Vu and Vi, Chi Thanh},
  booktitle={SOICT 2024},
  series={Communications in Computer and Information Science},
  volume={2352},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-981-96-4288-5_35}
}

@incollection{vietfood67_igi2025,
  title={It's Yummy: Real-Time Detection and Recognition of Vietnamese Dishes},
  author={Nguyen, Viet Hoang Nam and Vi, Chi Thanh},
  booktitle={Navigating Computing Challenges for a Sustainable World},
  publisher={IGI Global},
  year={2025},
  doi={10.4018/979-8-3373-0462-5.ch001}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---


## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Enjoy detecting Vietnamese food! / Chúc bạn phát hiện món ăn Việt vui vẻ!**
