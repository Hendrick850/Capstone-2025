#!/usr/bin/env python3
"""
Detailed Detection Debug - See exactly what model detects and why
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

def debug_detection():
    """Debug exactly what the model is detecting and confidence levels"""
    
    st.title("🔍 Detailed Detection Debug")
    st.markdown("### See exactly what your model detects and why")
    
    # Load model
    custom_model_path = "models/chicken_best.pt"
    
    if not os.path.exists(custom_model_path):
        st.error("❌ Model not found!")
        return
    
    try:
        model = YOLO(custom_model_path)
        st.success("✅ Model loaded successfully")
        
        # Show model info
        st.markdown("## 📊 Model Information")
        st.write(f"**Classes:** {list(model.names.values())}")
        
        # Confidence threshold control
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.25,  # Lower to see all detections
            step=0.05
        )
        
        # File uploader
        st.markdown("## 📸 Upload Test Image")
        uploaded_file = st.file_uploader(
            "Upload a chicken part image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload ONE image to see detailed detection analysis"
        )
        
        if uploaded_file:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width=400)
            
            # Convert for detection
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run detection with detailed output
            st.markdown("## 🎯 Detailed Detection Results")
            
            with st.spinner("Running detection..."):
                results = model(opencv_image, verbose=False)
            
            # Show ALL detections (even low confidence)
            all_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        coords = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        all_detections.append({
                            'detection_id': i,
                            'class_name': class_name,
                            'confidence': confidence,
                            'class_id': class_id,
                            'bbox': coords.tolist(),
                            'above_threshold': confidence >= confidence_threshold
                        })
            
            # Display results
            if all_detections:
                st.markdown(f"### Found {len(all_detections)} detections:")
                
                for detection in all_detections:
                    threshold_status = "✅ Above threshold" if detection['above_threshold'] else "❌ Below threshold"
                    confidence_pct = detection['confidence'] * 100
                    
                    # Color code by confidence
                    if confidence_pct >= 90:
                        confidence_color = "🟢"
                    elif confidence_pct >= 70:
                        confidence_color = "🟡"
                    elif confidence_pct >= 50:
                        confidence_color = "🟠"
                    else:
                        confidence_color = "🔴"
                    
                    st.markdown(f"""
                    **Detection {detection['detection_id'] + 1}:**
                    - **Class:** {detection['class_name']} (ID: {detection['class_id']})
                    - **Confidence:** {confidence_color} {confidence_pct:.1f}%
                    - **Status:** {threshold_status}
                    - **Bounding Box:** {[round(x, 1) for x in detection['bbox']]}
                    """)
                
                # Show only high-confidence detections
                high_conf_detections = [d for d in all_detections if d['above_threshold']]
                
                if high_conf_detections:
                    st.markdown(f"### 🎯 High-Confidence Detections (≥{confidence_threshold:.0%}):")
                    
                    for detection in high_conf_detections:
                        confidence_pct = detection['confidence'] * 100
                        
                        if confidence_pct >= 80:
                            st.success(f"🎯 **{detection['class_name']}** - {confidence_pct:.1f}% confidence")
                        elif confidence_pct >= 60:
                            st.info(f"🤔 **{detection['class_name']}** - {confidence_pct:.1f}% confidence")
                        else:
                            st.warning(f"❓ **{detection['class_name']}** - {confidence_pct:.1f}% confidence")
                
                # Analysis
                st.markdown("## 🔍 Detection Analysis")
                
                # Check for multiple detections
                if len(high_conf_detections) > 1:
                    st.warning(f"⚠️ **Multiple objects detected** - Model found {len(high_conf_detections)} different chicken parts")
                    st.write("This could explain confusion if image contains multiple parts")
                
                # Check confidence levels
                max_confidence = max([d['confidence'] for d in all_detections])
                
                if max_confidence < 0.7:
                    st.warning("⚠️ **Low maximum confidence** - Model is unsure about this image")
                    st.write("Possible causes:")
                    st.write("- Image quality issues")
                    st.write("- Unusual angle or lighting")
                    st.write("- Chicken part not clearly visible")
                    st.write("- Part looks different from training data")
                
                # Show class distribution
                class_counts = {}
                for detection in all_detections:
                    class_name = detection['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                if len(class_counts) > 1:
                    st.write("**Detected classes:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count} detection(s)")
                
            else:
                st.warning("❌ No objects detected at any confidence level")
                st.write("**Possible causes:**")
                st.write("- Image doesn't contain recognizable chicken parts")
                st.write("- Lighting/quality issues")
                st.write("- Very different from training data")
                
                # Suggest lowering threshold
                st.info("💡 Try lowering the confidence threshold to see if anything is detected")
        
        else:
            st.info("👆 Upload an image to see detailed detection analysis")
            
            st.markdown("## 💡 What This Tool Shows")
            st.markdown("""
            **This debug tool will help identify:**
            
            1. **Multiple detections** - If model sees multiple chicken parts in one image
            2. **Low confidence** - If model is unsure about predictions
            3. **Wrong class selection** - If model detects correctly but picks wrong class
            4. **Detection failure** - If model fails to detect anything
            
            **Upload a problematic image** (one where drumstick shows as thigh/breast) to see exactly what's happening.
            """)
            
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    debug_detection()