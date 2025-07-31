#!/usr/bin/env python3
"""
Chicken Parts Identifier Using Computer Vision
Capstone Project - Autopack
Author: Hendrick
Date: 31 July 2025

Main application for real-time chicken part detection using YOLO
"""

import cv2
import torch
import numpy as np
import streamlit as st
from datetime import datetime
import pandas as pd
import os
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chicken_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChickenPartDetector:
    """Main class for chicken part detection system"""
    
    def __init__(self, model_path: str = "models/best.pt", confidence_threshold: float = 0.5):
        """
        Initialize the chicken part detector
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence for detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.class_names = ['breast', 'thigh', 'wing', 'drumstick']
        self.colors = {
            'breast': (0, 255, 0),    # Green
            'thigh': (255, 0, 0),     # Blue  
            'wing': (0, 0, 255),      # Red
            'drumstick': (255, 255, 0) # Cyan
        }
        
        # Initialize model
        self.model = None
        self.load_model()
        
        # Initialize logging
        self.results_log = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['models', 'data', 'logs', 'captured_images', 'results']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def load_model(self):
        """Load the YOLO model for inference"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=self.model_path, force_reload=True)
                self.model.conf = self.confidence_threshold
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {self.model_path}. Loading default YOLOv5s")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                logger.info("Default YOLOv5s model loaded")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def detect_chicken_parts(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect chicken parts in the input image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return image, []
            
        try:
            # Run inference
            results = self.model(image)
            
            # Parse results
            detections = []
            annotated_image = image.copy()
            
            # Extract predictions
            predictions = results.pandas().xyxy[0]
            
            for _, detection in predictions.iterrows():
                if detection['confidence'] >= self.confidence_threshold:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                                   int(detection['xmax']), int(detection['ymax'])
                    
                    # Get class name and confidence
                    class_name = detection['name'] if 'name' in detection else f"class_{int(detection['class'])}"
                    confidence = detection['confidence']
                    
                    # Store detection info
                    detection_info = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'timestamp': datetime.now().isoformat()
                    }
                    detections.append(detection_info)
                    
                    # Draw bounding box and label
                    color = self.colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_image, detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return image, []
    
    def log_detection_results(self, detections: List[Dict]):
        """Log detection results to CSV and memory"""
        if detections:
            for detection in detections:
                self.results_log.append(detection)
            
            # Save to CSV
            df = pd.DataFrame(self.results_log)
            csv_path = f"logs/detections_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(csv_path, index=False)
            
    def capture_and_save_image(self, image: np.ndarray, detections: List[Dict]):
        """Save captured image with detection info"""
        if detections:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"captured_images/detection_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            
            # Save detection metadata
            metadata_file = f"captured_images/detection_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'detections': detections,
                    'image_path': filename
                }, f, indent=2)
            
            logger.info(f"Saved detection image: {filename}")

class CameraManager:
    """Manages camera input for the detection system"""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera manager
        
        Args:
            camera_id: Camera device ID (0 for default)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_recording = False
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_id}")
                return False
                
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Chicken Parts Detector",
        page_icon="🐔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐔 Chicken Parts Identifier")
    st.markdown("### Real-time Computer Vision Detection System")
    
    # Sidebar controls
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    camera_id = st.sidebar.selectbox("Camera Source", [0, 1, 2], index=0)
    
    # Initialize detector and camera
    @st.cache_resource
    def load_detector():
        return ChickenPartDetector(confidence_threshold=confidence_threshold)
    
    detector = load_detector()
    camera_manager = CameraManager(camera_id)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Detection Feed")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection Results")
        results_placeholder = st.empty()
        
        # Control buttons
        if st.button("Start Detection"):
            if camera_manager.initialize_camera():
                st.success("Camera initialized!")
                
                # Detection loop placeholder
                # (In actual implementation, this would run in a separate thread)
                frame = camera_manager.get_frame()
                if frame is not None:
                    annotated_frame, detections = detector.detect_chicken_parts(frame)
                    
                    # Display results
                    video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                    
                    if detections:
                        results_df = pd.DataFrame(detections)
                        results_placeholder.dataframe(results_df)
                        detector.log_detection_results(detections)
                    
            else:
                st.error("Failed to initialize camera")
        
        if st.button("Stop Detection"):
            camera_manager.release_camera()
            st.info("Detection stopped")
    
    # Statistics section
    st.subheader("Detection Statistics")
    if detector.results_log:
        stats_df = pd.DataFrame(detector.results_log)
        st.dataframe(stats_df.tail(10))
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(detector.results_log))
        with col2:
            if stats_df['confidence'].count() > 0:
                st.metric("Average Confidence", f"{stats_df['confidence'].mean():.2f}")
        with col3:
            most_common = stats_df['class'].mode()
            if len(most_common) > 0:
                st.metric("Most Detected Class", most_common[0])

if __name__ == "__main__":
    main()