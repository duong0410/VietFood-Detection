# VietFood-Detection

## Overview
VietFood-Detection is a deep learning–based object detection project for automatic detection and classification of Vietnamese food items in images, built on the YOLO framework.  
The project is implemented and trained primarily on **Kaggle with GPU support**, targeting research and practical applications such as food recognition, dietary assessment, and Vietnamese cuisine promotion.

---

## Goal
Build an automated system capable of detecting and classifying Vietnamese dishes in real-world images with high accuracy and stability using YOLO-based object detection models.

---

## Objectives

### 1. Dataset Preparation
- Utilize the VietFood67/VietFood68 dataset for Vietnamese food detection.
- Verify dataset integrity and annotation quality (YOLO format).
- Analyze class distribution and bounding box statistics.

### 2. Model Training
- Train YOLO-based object detection models (YOLOv11s).
- Experiment with architectural variants (e.g., P2 feature level).
- Fine-tune hyperparameters and training strategies.
- Monitor training metrics (loss, Precision, Recall, mAP).

### 3. Evaluation & Visualization
- Evaluate model performance on validation and test sets.
- Visualize training curves and detection results.
- Compare results between different YOLO variants.

### 4. Deployment & Application
- Export trained model weights and configuration files.
- Prepare inference-ready models for real-world applications.

---

## Training Environment
- Platform: Kaggle Notebook  
- Hardware: NVIDIA GPU (Kaggle-provided)  
- Framework: Ultralytics YOLO  
- Image size: 640 × 640  
- Training strategy: Fine-tuning from pretrained YOLO weights  
- Checkpointing: Resume training using saved checkpoints (`last.pt`) due to Kaggle GPU time limits

---

## Dataset

### VietFood67: Vietnamese Food Image Dataset for Detection

This project uses the **VietFood67 dataset**, which is **publicly released and maintained by its original authors**.  
VietFood67 is a curated dataset consisting of approximately **33,000 images across 68 food categories**, designed for Vietnamese food detection and recognition tasks.

The dataset provides **bounding box annotations** suitable for object detection models such as YOLO and has been used in multiple peer-reviewed academic publications.  
It is also integrated into the **FoodDetector** real-time dietary assessment system.

## Citation

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
  title={It’s Yummy: Real-Time Detection and Recognition of Vietnamese Dishes},
  author={Nguyen, Viet Hoang Nam and Vi, Chi Thanh},
  booktitle={Navigating Computing Challenges for a Sustainable World},
  publisher={IGI Global},
  year={2025},
  doi={10.4018/979-8-3373-0462-5.ch001}
}


