"""
Image processing utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def resize_image(image: np.ndarray, target_width: int, target_height: int, 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target dimensions
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        # Calculate aspect ratio
        h, w = image.shape[:2]
        aspect = w / h
        
        if w > h:
            new_w = target_width
            new_h = int(target_width / aspect)
        else:
            new_h = target_height
            new_w = int(target_height * aspect)
        
        # Ensure dimensions don't exceed target
        if new_h > target_height:
            new_h = target_height
            new_w = int(target_height * aspect)
        
        if new_w > target_width:
            new_w = target_width
            new_h = int(target_width / aspect)
    else:
        new_w = target_width
        new_h = target_height
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB
    
    Args:
        image: BGR image
        
    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR
    
    Args:
        image: RGB image
        
    Returns:
        BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def numpy_to_qimage(image: np.ndarray):
    """
    Convert numpy array to QImage
    
    Args:
        image: Image as numpy array (BGR format)
        
    Returns:
        QImage object
    """
    from PyQt5.QtGui import QImage
    
    # Convert BGR to RGB
    rgb_image = bgr_to_rgb(image)
    
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


def qimage_to_numpy(qimage) -> np.ndarray:
    """
    Convert QImage to numpy array
    
    Args:
        qimage: QImage object
        
    Returns:
        Image as numpy array (BGR format)
    """
    from PyQt5.QtGui import QImage
    
    # Convert to RGB888 format if not already
    if qimage.format() != QImage.Format_RGB888:
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
    
    width = qimage.width()
    height = qimage.height()
    
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 3)
    
    # Convert RGB to BGR
    return rgb_to_bgr(arr)


def create_blank_image(width: int, height: int, color: Tuple[int, int, int] = (240, 240, 240)) -> np.ndarray:
    """
    Create a blank image with specified color
    
    Args:
        width: Image width
        height: Image height
        color: BGR color tuple
        
    Returns:
        Blank image as numpy array
    """
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image
