#!/usr/bin/env python3
"""
Chicken Parts Identifier - ULTRALYTICS VERSION (FIXED)
Capstone Project 2025 - Autopack
Author: Hendrick
Date: August 2025

Fixed version using direct ultralytics YOLO - no more cache issues!
"""

import cv2
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

# Configure logging without emojis (fixes unicode error)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chicken_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedChickenDetector:
    """Fixed chicken part detection using ultralytics YOLO"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the detector with ultralytics YOLO"""
        self.confidence_threshold = confidence_threshold
        
        # Custom trained chicken part classes
        self.chicken_classes = ['breast', 'thigh', 'wing', 'drumstick']
        
        # Colors for visualization
        self.colors = {
            'breast': (0, 255, 0),      # Green
            'thigh': (255, 0, 0),       # Blue  
            'wing': (0, 0, 255),        # Red
            'drumstick': (255, 255, 0), # Cyan
        }
        
        self.model = None
        self.results_log = []
        self.setup_directories()
        self.load_ultralytics_model()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['models', 'logs', 'results', 'test_results']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def load_ultralytics_model(self):
        """Load model using ultralytics YOLO (much more reliable)"""
        custom_model_path = "models/chicken_best.pt"
        fallback_model_path = "runs/detect/train/weights/best.pt"
        
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            if os.path.exists(custom_model_path):
                logger.info(f"Loading custom model: {custom_model_path}")
                self.model = YOLO(custom_model_path)
                st.success(f"✅ Custom model loaded: {custom_model_path}")
                self.model_type = "CUSTOM TRAINED (95.6% accuracy)"
                
            elif os.path.exists(fallback_model_path):
                logger.info(f"Loading from training folder: {fallback_model_path}")
                self.model = YOLO(fallback_model_path)
                st.success(f"✅ Training model loaded: {fallback_model_path}")
                self.model_type = "CUSTOM TRAINED (95.6% accuracy)"
                
            else:
                st.error("❌ Custom model not found!")
                st.code('Copy model: cp "runs/detect/train/weights/best.pt" "models/chicken_best.pt"')
                st.stop()
            
            # Set confidence threshold
            self.model.conf = self.confidence_threshold
            logger.info(f"Model loaded successfully - Type: {self.model_type}")
            
        except ImportError:
            st.error("❌ Ultralytics not installed!")
            st.info("Install with: pip install ultralytics")
            st.stop()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Model loading failed: {e}")
            raise
            
    def fix_class_mapping(self, detected_class_name):
        """Fix incorrect class mapping from mislabeled training data"""
        
        # Based on debug results: drumsticks were labeled as "breast" during training
        class_fix_mapping = {
            "breast": "drumstick",    # Model says "breast" but means "drumstick"
            "thigh": "thigh",         # Thigh is probably correct
            "wing": "wing",           # Wing is probably correct  
            "drumstick": "breast"     # Model says "drumstick" but might mean "breast"
        }
        
        return class_fix_mapping.get(detected_class_name, detected_class_name)
    
    def detect_chicken_parts(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect chicken parts using ultralytics YOLO
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return image, []
            
        try:
            # Run inference with ultralytics YOLO
            results = self.model(image, verbose=False)
            
            detections = []
            annotated_image = image.copy()
            
            # Parse ultralytics results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box data
                        coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if confidence >= self.confidence_threshold:
                            # Get class name and fix the mapping
                            raw_class_name = self.model.names[class_id]
                            class_name = self.fix_class_mapping(raw_class_name)
                            
                            # Extract coordinates
                            x1, y1, x2, y2 = map(int, coords)
                            
                            # Store detection info
                            detection_info = {
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'timestamp': datetime.now().isoformat(),
                                'model_type': self.model_type,
                                'is_chicken_part': class_name in self.chicken_classes
                            }
                            detections.append(detection_info)
                            
                            # Choose color for this chicken part
                            color = self.colors.get(class_name, (255, 255, 255))
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 4)
                            
                            # Enhanced label
                            label = f"{class_name}: {confidence:.2f}"
                            if class_name in self.chicken_classes:
                                label = f"🐔 {label}"
                            
                            # Calculate label size for background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            
                            # Draw label background
                            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 20), 
                                        (x1 + label_size[0] + 10, y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_image, label, (x1 + 5, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            return annotated_image, detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            st.error(f"Detection failed: {e}")
            return image, []
    
    def save_test_results(self, detections: List[Dict], image_name: str):
        """Save test results"""
        if detections:
            for detection in detections:
                detection['image_name'] = image_name
                detection['test_session'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.results_log.append(detection)
            
            # Save to CSV
            df = pd.DataFrame(self.results_log)
            csv_path = f"test_results/ultralytics_tests_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Test results saved to {csv_path}")

def main():
    """Main Streamlit application - ULTRALYTICS VERSION"""
    st.set_page_config(
        page_title="Fixed Chicken Detector Test",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎯 Fixed Chicken Parts Detector - Test Interface")
    st.markdown("### Using Ultralytics YOLO (No More Cache Issues!)")
    
    # Model info banner
    st.success("🛠️ **Fixed Version** - Using reliable ultralytics YOLO!")
    
    # Sidebar controls
    st.sidebar.markdown("## ⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Lower = more detections, Higher = only confident detections"
    )
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return FixedChickenDetector(confidence_threshold=confidence_threshold)
    
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"❌ Failed to load detection model: {e}")
        st.stop()
    
    # File uploader
    st.markdown("## 📸 Upload Chicken Images for Testing")
    uploaded_files = st.file_uploader(
        "Choose chicken part images", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload images of chicken parts to test your custom model"
    )
    
    # Process uploaded images
    if uploaded_files:
        st.markdown(f"**Testing {len(uploaded_files)} images with ultralytics YOLO...**")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### 🔍 Test {idx + 1}: {uploaded_file.name}")
            
            # Create columns for layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Convert to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Run detection
                with st.spinner(f"🎯 Analyzing with ultralytics YOLO..."):
                    annotated_image, detections = detector.detect_chicken_parts(opencv_image)
                
                # Display results
                if len(detections) > 0:
                    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="🎯 Detection Results", use_column_width=True)
                else:
                    st.info("No chicken parts detected. Try lowering confidence threshold.")
            
            with col2:
                # Detection results panel
                st.markdown("#### 📊 Detection Results")
                
                if detections:
                    chicken_detections = [d for d in detections if d.get('is_chicken_part', False)]
                    
                    if chicken_detections:
                        st.success(f"🐔 Found {len(chicken_detections)} chicken parts!")
                        
                        # Show each chicken detection
                        for i, detection in enumerate(chicken_detections):
                            with st.expander(f"🎯 Chicken Part {i+1}: {detection['class'].title()}", expanded=True):
                                confidence_pct = detection['confidence'] * 100
                                st.write(f"**Type:** {detection['class'].title()}")
                                st.write(f"**Confidence:** {confidence_pct:.1f}%")
                                
                                # Confidence indicator
                                if confidence_pct >= 90:
                                    st.success(f"🎯 Excellent confidence!")
                                elif confidence_pct >= 70:
                                    st.info(f"✅ Good confidence")
                                else:
                                    st.warning(f"⚠️ Low confidence")
                        
                        # Results summary table
                        chicken_df = pd.DataFrame(chicken_detections)[['class', 'confidence']]
                        chicken_df['confidence'] = chicken_df['confidence'].apply(lambda x: f"{x:.2f}")
                        st.dataframe(chicken_df, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ No chicken parts detected")
                        if detections:
                            st.write("🔍 Other objects detected:")
                            for det in detections:
                                st.write(f"• {det['class']}: {det['confidence']:.2f}")
                
                else:
                    st.warning("⚠️ No objects detected")
                    st.markdown("**Try:**")
                    st.write("• Lower confidence threshold")
                    st.write("• Better lighting in image")
                    st.write("• Clearer chicken part image")
                
                # Save results
                if detections:
                    detector.save_test_results(detections, uploaded_file.name)
            
            st.divider()
    
    else:
        # Instructions
        st.info("👆 **Upload chicken part images above to test the fixed model!**")
        
        st.markdown("## 📋 Testing Instructions")
        st.markdown("""
        **Step 1:** Upload chicken part images using the file uploader above
        
        **Step 2:** Adjust confidence threshold in sidebar (start with 0.5)
        
        **Step 3:** Review detection results for accuracy
        
        **Step 4:** Compare with your training results (95.6% accuracy)
        """)
        
        # Model performance info
        with st.expander("🏆 Your Custom Model Performance"):
            st.markdown("""
            **Training Results (100 epochs, 1.23 hours):**
            - 🎯 **Overall Accuracy: 95.6%** (Excellent!)
            - 🥩 **Breast Detection: 99.5%** 
            - 🍗 **Thigh Detection: 92.8%**
            - 🍖 **Wing Detection: 90.6%**
            - 🦴 **Drumstick Detection: 99.5%**
            
            **Speed Performance:**
            - ⚡ **17 FPS** - Real-time capable
            - 🚀 **58.7ms inference** - Very fast
            """)
    
    # Session statistics
    if detector.results_log:
        st.markdown("## 📈 Test Session Statistics")
        stats_df = pd.DataFrame(detector.results_log)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_detections = len(detector.results_log)
            st.metric("Total Detections", total_detections)
        with col2:
            if 'confidence' in stats_df.columns:
                avg_conf = stats_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_conf:.2f}")
        with col3:
            if 'class' in stats_df.columns:
                most_common = stats_df['class'].mode()
                if len(most_common) > 0:
                    st.metric("Most Detected", most_common[0])
        
        # Recent detections
        if len(stats_df) > 0:
            st.dataframe(stats_df[['class', 'confidence', 'image_name']].tail(10))

if __name__ == "__main__":
    main()