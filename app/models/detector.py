"""
Food Detector Module
Handles loading YOLO model and performing inference
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import torch


class FoodDetector:
    """
    Wrapper class for YOLO model to detect Vietnamese food items
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, max_det: int = 300):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            max_det: Maximum number of detections per image
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the YOLO model from checkpoint
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            print(f"Loading model from {self.model_path}")
            print(f"Using device: {self.device}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def detect(self, image_path: str) -> Optional[object]:
        """
        Perform detection on an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Detection results object or None if error
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded!")
            
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )
            
            return results[0] if len(results) > 0 else None
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None
    
    def detect_from_array(self, image: np.ndarray) -> Optional[object]:
        """
        Perform detection on a numpy array (image)
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Detection results object or None if error
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded!")
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )
            
            return results[0] if len(results) > 0 else None
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None
    
    def parse_results(self, results) -> List[dict]:
        """
        Parse detection results into a list of dictionaries
        
        Args:
            results: YOLO results object
            
        Returns:
            List of detection dictionaries with keys: class_id, class_name, 
            confidence, bbox (x1, y1, x2, y2)
        """
        if results is None:
            return []
        
        detections = []
        
        try:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = results.names[cls_id]
                
                detection = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                }
                
                detections.append(detection)
        
        except Exception as e:
            print(f"Error parsing results: {str(e)}")
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                       colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            colors: List of BGR colors for different classes
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cls_name = detection['class_name']
            conf = detection['confidence']
            cls_id = detection['class_id']
            
            # Select color
            color = colors[cls_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{cls_name}: {conf:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                result_image, 
                (x1, y1 - label_h - baseline - 5), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return result_image
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "Not loaded"}
        
        return {
            "status": "Loaded",
            "device": self.device,
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_det": self.max_det
        }
