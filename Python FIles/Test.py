#!/usr/bin/env python3
"""
Chicken Parts Identifier Using Computer Vision - PHOTO TEST VERSION
Capstone Project - Autopack
Author: [Your Name]
Date: July 2025

Test version that works with uploaded photos instead of camera
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
from PIL import Image

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
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the chicken part detector
        
        Args:
            confidence_threshold: Minimum confidence for detection
        """
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
            logger.info("Loading YOLOv5s model...")
            # Use YOLOv5s pretrained model - works for general object detection
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            self.model.conf = self.confidence_threshold
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Failed to load model: {e}")
            raise
            
    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects in the input image (using general YOLO model)
        
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
                    class_name = detection['name']
                    confidence = detection['confidence']
                    
                    # Store detection info
                    detection_info = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'timestamp': datetime.now().isoformat()
                    }
                    detections.append(detection_info)
                    
                    # Choose color (use default for non-chicken parts)
                    color = self.colors.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box and label
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return annotated_image, detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            st.error(f"Detection error: {e}")
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

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Chicken Parts Detector - Photo Test",
        page_icon="🐔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐔 Chicken Parts Identifier - Photo Test Version")
    st.markdown("### Upload a photo to test object detection")
    
    # Sidebar controls
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return ChickenPartDetector(confidence_threshold=confidence_threshold)
    
    try:
        detector = load_detector()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    
    # File uploader
    st.subheader("📸 Upload Test Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload any image to test object detection"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection Results")
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run detection
            with st.spinner("🔍 Detecting objects..."):
                annotated_image, detections = detector.detect_objects(opencv_image)
            
            # Convert back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.image(annotated_rgb, caption="Detection Results", use_column_width=True)
            
            # Log results
            if detections:
                detector.log_detection_results(detections)
        else:
            st.info("👆 Upload an image to start testing!")
    
    with col2:
        st.subheader("Detection Info")
        
        if uploaded_file is not None and 'detections' in locals():
            if detections:
                st.success(f"✅ Found {len(detections)} objects!")
                
                # Show detection details
                for i, detection in enumerate(detections):
                    with st.expander(f"Detection {i+1}: {detection['class']}"):
                        st.write(f"**Class:** {detection['class']}")
                        st.write(f"**Confidence:** {detection['confidence']:.2f}")
                        st.write(f"**Bounding Box:** {detection['bbox']}")
                
                # Show as dataframe
                results_df = pd.DataFrame(detections)
                st.dataframe(results_df[['class', 'confidence']], use_container_width=True)
                
            else:
                st.warning("⚠️ No objects detected")
                st.info("Try:")
                st.write("- Lowering confidence threshold")
                st.write("- Using a clearer image")
                st.write("- Uploading a different photo")
    
    # Instructions
    st.subheader("📋 How to Test")
    st.markdown("""
    **Step 1:** Adjust the confidence threshold (left sidebar)
    
    **Step 2:** Upload any image (chicken, food, objects, etc.)
    
    **Step 3:** See what the AI detects!
    
    **Note:** This uses a general YOLO model that detects common objects. 
    For specific chicken parts, we'll train a custom model later.
    """)
    
    # Current capabilities
    with st.expander("🤖 What can this model detect?"):
        st.write("Current model can detect:")
        common_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
        ]
        st.write(", ".join(common_classes))
    
    # Statistics section
    if detector.results_log:
        st.subheader("📊 Session Statistics")
        stats_df = pd.DataFrame(detector.results_log)
        
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
        
        # Show recent detections
        st.dataframe(stats_df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()