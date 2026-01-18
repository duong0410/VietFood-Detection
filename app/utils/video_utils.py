"""
Video and Webcam processing utilities
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Callable
import time


class VideoProcessor:
    """
    Class for processing video files
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        self.current_frame = 0
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def set_position(self, frame_number: int):
        """
        Set video position to specific frame
        
        Args:
            frame_number: Frame number to seek to
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
    
    def get_progress(self) -> float:
        """
        Get current progress as percentage
        
        Returns:
            Progress percentage (0-100)
        """
        if self.frame_count == 0:
            return 0
        return (self.current_frame / self.frame_count) * 100
    
    def reset(self):
        """Reset video to beginning"""
        self.set_position(0)
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


class WebcamCapture:
    """
    Class for capturing from webcam
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize webcam capture
        
        Args:
            camera_index: Camera device index (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_opened = False
    
    def open(self) -> bool:
        """
        Open webcam connection
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        self.is_opened = self.cap.isOpened()
        
        if self.is_opened:
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return self.is_opened
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from webcam
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Release webcam"""
        if self.cap:
            self.cap.release()
            self.is_opened = False
    
    def get_properties(self) -> dict:
        """
        Get camera properties
        
        Returns:
            Dictionary with camera properties
        """
        if not self.is_opened or self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }
    
    @staticmethod
    def get_available_cameras(max_test: int = 2) -> list:
        """
        Get list of available camera indices
        
        Args:
            max_test: Maximum number of cameras to test (default: 2)
            
        Returns:
            List of available camera indices
        """
        available = []
        # Suppress OpenCV warnings temporarily
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(max_test):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
        return available
    
    def __del__(self):
        """Destructor"""
        self.release()


class VideoWriter:
    """
    Class for writing processed video
    """
    
    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int], 
                 codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height) tuple
            codec: Video codec (default: 'mp4v')
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create video writer
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video
        
        Args:
            frame: Frame to write
        """
        # Ensure frame size matches
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


def process_video_with_callback(video_path: str, 
                                output_path: Optional[str],
                                process_func: Callable,
                                progress_callback: Optional[Callable] = None,
                                skip_frames: int = 0) -> bool:
    """
    Process video file with a processing function
    
    Args:
        video_path: Input video path
        output_path: Output video path (None to not save)
        process_func: Function to process each frame
        progress_callback: Callback function for progress updates
        skip_frames: Number of frames to skip (0 = process all)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open video
        processor = VideoProcessor(video_path)
        
        # Create writer if output path provided
        writer = None
        if output_path:
            writer = VideoWriter(
                output_path,
                processor.fps,
                (processor.width, processor.height)
            )
        
        frame_count = 0
        
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            
            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Process frame
            processed_frame = process_func(frame)
            
            # Write to output if writer exists
            if writer and processed_frame is not None:
                writer.write_frame(processed_frame)
            
            # Update progress
            if progress_callback:
                progress = processor.get_progress()
                progress_callback(progress, frame_count)
            
            frame_count += 1
        
        # Cleanup
        processor.release()
        if writer:
            writer.release()
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False


def estimate_processing_time(video_duration: float, fps: int, 
                            sample_processing_time: float) -> float:
    """
    Estimate total video processing time
    
    Args:
        video_duration: Video duration in seconds
        fps: Frames per second
        sample_processing_time: Time to process one frame
        
    Returns:
        Estimated total time in seconds
    """
    total_frames = video_duration * fps
    return total_frames * sample_processing_time
