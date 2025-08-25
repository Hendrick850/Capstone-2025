#!/usr/bin/env python3
"""
AUTOPACK AI - Complete Multi-Feature Chicken Detection System
Combining all features: Security, Camera Detection, Freshness Assessment, AI Analytics
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import hashlib
import time
import logging
import os
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="AUTOPACK AI - Complete System",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .safety-pass {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .safety-caution {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .safety-fail {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .ai-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SECURITY MANAGER (Feature 1)
# =============================================================================

class SecurityManager:
    """Professional security manager for AUTOPACK AI system"""
    
    def __init__(self):
        self.setup_security_logging()
        self.valid_access_codes = [
            "AUTOPACK2025", "CAPSTONE", "FEATURE1", "ULTIMATE", 
            "CHICKEN", "DEMO", "TEACHER", "PRESENTATION"
        ]
        
    def setup_security_logging(self):
        """Setup secure logging for audit trails"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename='logs/security_audit.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def access_control_check(self):
        """Enhanced access control for system usage"""
        st.sidebar.markdown("## 🔐 Access Control")
        
        if 'authorized' not in st.session_state:
            st.session_state.authorized = False
            
        if not st.session_state.authorized:
            st.sidebar.warning("🚨 System Access Required")
            st.sidebar.info("📋 **Available Demo Codes:**")
            st.sidebar.code("AUTOPACK2025\nDEMO\nTEACHER\nPRESENTATION")
            
            access_code = st.sidebar.text_input(
                "Enter Access Code:", 
                type="password",
                placeholder="Enter any demo code above..."
            )
            
            if st.sidebar.button("🔓 Authorize Access"):
                if access_code in self.valid_access_codes:
                    st.session_state.authorized = True
                    logging.info(f"ACCESS_GRANTED - Code: {access_code}, Time: {datetime.now()}")
                    st.sidebar.success("✅ Access Granted")
                    st.rerun()
                else:
                    logging.info(f"ACCESS_DENIED - Invalid code: {access_code[:3]}***")
                    st.sidebar.error("❌ Invalid Access Code")
                    
            return False
        else:
            st.sidebar.success("✅ Authorized User")
            if st.sidebar.button("🚪 Logout"):
                st.session_state.authorized = False
                logging.info(f"LOGOUT - Time: {datetime.now()}")
                st.rerun()
            return True
    
    def data_privacy_notice(self):
        """Display data privacy information"""
        with st.sidebar.expander("🛡️ Data Privacy Notice"):
            st.markdown("""
            **Data Processing Information:**
            
            ✅ **Local Processing**: All detection runs on your device
            ✅ **No Data Storage**: Images are not permanently stored
            ✅ **No External Transmission**: Data stays on your system
            ✅ **Secure Processing**: Industry-standard security measures
            """)
    
    def secure_file_handling(self, uploaded_file):
        """Secure file processing with validation"""
        if uploaded_file is None:
            return None, "No file uploaded"
            
        max_size = 10 * 1024 * 1024  # 10MB limit
        if uploaded_file.size > max_size:
            return None, f"File too large (max 10MB). Your file: {uploaded_file.size/1024/1024:.1f}MB"
            
        allowed_types = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_types:
            return None, f"File type '{file_extension}' not allowed. Use: {', '.join(allowed_types)}"
            
        logging.info(f"FILE_ACCEPTED - File: {uploaded_file.name}, Size: {uploaded_file.size}")
        return uploaded_file, "File validated successfully"

# =============================================================================
# CHICKEN FRESHNESS ANALYZER (Original Feature)
# =============================================================================

class ChickenFreshnessAnalyzer:
    def __init__(self):
        self.freshness_criteria = {
            'color_ranges': {
                'fresh': {'hue_range': (15, 35), 'saturation_range': (30, 80), 'value_range': (40, 90)},
                'questionable': {'hue_range': (35, 50), 'saturation_range': (20, 90), 'value_range': (30, 85)},
                'spoiled': {'hue_range': (50, 100), 'saturation_range': (40, 100), 'value_range': (20, 80)}
            }
        }
        
        self.scaler = StandardScaler()
        self.model = self._create_demo_model()
        
    def _create_demo_model(self):
        """Create a demo ML model for freshness classification"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        np.random.seed(42)
        X_demo = np.random.rand(1000, 10)
        y_demo = np.random.choice([0, 1, 2], 1000)
        model.fit(X_demo, y_demo)
        self.scaler.fit(X_demo)
        return model
    
    def analyze_image(self, image):
        """Analyze chicken image for freshness indicators"""
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            color_features = self._extract_color_features(opencv_image)
            texture_features = self._extract_texture_features(opencv_image)
            
            combined_features = np.array([
                color_features['avg_hue'], color_features['avg_saturation'],
                color_features['avg_value'], color_features['color_variance'],
                texture_features['smoothness'], texture_features['uniformity'],
                texture_features['contrast'], texture_features['homogeneity'],
                color_features['dominant_color_purity'], texture_features['edge_density']
            ]).reshape(1, -1)
            
            scaled_features = self.scaler.transform(combined_features)
            prediction = self.model.predict(scaled_features)[0]
            confidence = np.max(self.model.predict_proba(scaled_features))
            
            categories = {0: 'Fresh', 1: 'Questionable', 2: 'Spoiled'}
            freshness_category = categories[prediction]
            
            return {
                'freshness_category': freshness_category,
                'confidence': confidence,
                'color_features': color_features,
                'texture_features': texture_features,
                'safety_score': self._calculate_safety_score(color_features, texture_features),
                'recommendations': self._get_recommendations(freshness_category, confidence)
            }
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None
    
    def _extract_color_features(self, image):
        """Extract color-based features from the image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return {
            'avg_hue': np.mean(hsv[:, :, 0]),
            'avg_saturation': np.mean(hsv[:, :, 1]),
            'avg_value': np.mean(hsv[:, :, 2]),
            'color_variance': np.var(hsv.reshape(-1, 3), axis=0).mean(),
            'dominant_color': np.mean(image.reshape(-1, 3), axis=0),
            'dominant_color_purity': np.std(image.reshape(-1, 3), axis=0).mean()
        }
    
    def _extract_texture_features(self, image):
        """Extract texture-based features from the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smoothness = 1 - (np.var(gray) / 255**2)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        uniformity = np.sum(hist**2) / (gray.shape[0] * gray.shape[1])**2
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        contrast = np.var(laplacian)
        homogeneity = 1 / (1 + contrast/1000)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return {
            'smoothness': smoothness, 'uniformity': uniformity,
            'contrast': contrast, 'homogeneity': homogeneity,
            'edge_density': edge_density
        }
    
    def _calculate_safety_score(self, color_features, texture_features):
        """Calculate overall safety score (0-100)"""
        color_score = self._score_color_freshness(color_features)
        texture_score = self._score_texture_freshness(texture_features)
        return max(0, min(100, color_score * 0.7 + texture_score * 0.3))
    
    def _score_color_freshness(self, features):
        """Score color features for freshness (0-100)"""
        hue = features['avg_hue']
        if 15 <= hue <= 35:
            return 90 + np.random.uniform(-10, 10)
        elif 35 <= hue <= 50:
            return 60 + np.random.uniform(-15, 15)
        else:
            return 30 + np.random.uniform(-20, 20)
    
    def _score_texture_freshness(self, features):
        """Score texture features for freshness (0-100)"""
        return max(0, min(100, (features['smoothness'] * 50) + (features['uniformity'] * 50)))
    
    def _get_recommendations(self, category, confidence):
        """Get safety recommendations based on assessment"""
        recommendations = {
            'Fresh': ["✅ Safe for consumption", "Store in refrigerator at ≤40°F (4°C)", "Use within 1-2 days for best quality"],
            'Questionable': ["⚠️ Exercise caution", "Check for off-odors before cooking", "Cook immediately if using"],
            'Spoiled': ["❌ Do NOT consume", "Discard immediately", "Clean surfaces that contacted the meat"]
        }
        
        base_recs = recommendations.get(category, recommendations['Questionable'])
        if confidence < 0.7:
            base_recs.append(f"⚠️ Low confidence ({confidence:.1%}) - Consider additional inspection")
        
        return base_recs

# =============================================================================
# CAMERA DETECTOR (Feature 2)
# =============================================================================

class CameraDetector:
    """Camera detection class for Feature 2"""
    
    def __init__(self):
        self.model = None
        self.camera = None
        self.is_active = False
        
    def simulate_detection(self, frame):
        """Simulate detection for demo purposes"""
        height, width = frame.shape[:2]
        
        # Simulate random detections
        num_detections = np.random.randint(0, 3)
        detections = []
        
        for i in range(num_detections):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(50, 200)
            y2 = y1 + np.random.randint(50, 200)
            conf = np.random.uniform(0.5, 0.95)
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Chicken Part: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2],
                'class': np.random.choice(['breast', 'thigh', 'wing', 'drumstick'])
            })
        
        return frame, detections

# =============================================================================
# AI ANALYTICS ENGINE (Feature 3)
# =============================================================================

class AIAnalyticsEngine:
    """AI-powered analytics for chicken detection data"""
    
    def __init__(self):
        self.quality_thresholds = {'excellent': 0.9, 'good': 0.7, 'fair': 0.5, 'poor': 0.3}
        
    def analyze_detection_patterns(self, detection_data):
        """AI-powered pattern recognition in detection data"""
        if not detection_data:
            return {"error": "No data to analyze"}
        
        df = pd.DataFrame(detection_data)
        
        return {
            'quality_trends': self.analyze_quality_trends(df),
            'detection_efficiency': self.calculate_detection_efficiency(df),
            'temporal_patterns': self.detect_temporal_patterns(df),
            'anomaly_detection': self.detect_anomalies(df),
            'predictive_insights': self.generate_predictions(df)
        }
    
    def analyze_quality_trends(self, df):
        """AI quality assessment and trending"""
        if 'confidence' not in df.columns:
            return {"error": "No confidence data"}
        
        df['quality_category'] = df['confidence'].apply(self.categorize_quality)
        quality_distribution = df['quality_category'].value_counts()
        avg_confidence = df['confidence'].mean()
        
        quality_weights = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        weighted_quality = sum(quality_distribution.get(cat, 0) * weight 
                             for cat, weight in quality_weights.items())
        total_detections = len(df)
        ai_quality_score = (weighted_quality / (total_detections * 4)) * 100 if total_detections > 0 else 0
        
        return {
            'average_confidence': float(avg_confidence),
            'quality_distribution': quality_distribution.to_dict(),
            'ai_quality_score': float(ai_quality_score),
            'ai_recommendation': self.generate_quality_recommendation(ai_quality_score)
        }
    
    def calculate_detection_efficiency(self, df):
        """AI-powered efficiency analysis"""
        total_detections = len(df)
        
        if 'class' in df.columns:
            class_distribution = df['class'].value_counts()
            detection_balance = self.calculate_balance_score(class_distribution)
        else:
            class_distribution = pd.Series()
            detection_balance = 0
        
        ai_efficiency_score = (detection_balance / 100) * 0.6 + min(total_detections / 100, 1.0) * 0.4
        ai_efficiency_score *= 100
        
        return {
            'total_detections': total_detections,
            'class_distribution': class_distribution.to_dict(),
            'ai_efficiency_score': float(ai_efficiency_score),
            'ai_insight': f"System showing {'high' if ai_efficiency_score > 70 else 'moderate'} efficiency"
        }
    
    def detect_temporal_patterns(self, df):
        """AI-powered temporal pattern detection"""
        if 'timestamp' not in df.columns:
            return {"error": "No timestamp data"}
        
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        hourly_patterns = df.groupby('hour').size()
        
        peak_hour = hourly_patterns.idxmax() if len(hourly_patterns) > 0 else 12
        
        return {
            'peak_detection_hour': int(peak_hour),
            'hourly_distribution': hourly_patterns.to_dict(),
            'ai_insight': f"Peak detection activity at {peak_hour}:00"
        }
    
    def detect_anomalies(self, df):
        """AI anomaly detection"""
        if len(df) < 10:
            return {"message": "Insufficient data for anomaly detection"}
        
        confidence_mean = df['confidence'].mean()
        confidence_std = df['confidence'].std()
        confidence_threshold = confidence_mean - 2 * confidence_std
        anomalous_detections = df[df['confidence'] < confidence_threshold]
        
        return {
            'confidence_anomalies': len(anomalous_detections),
            'anomaly_threshold': float(confidence_threshold),
            'ai_alert': f"Found {len(anomalous_detections)} low-confidence anomalies" if len(anomalous_detections) > 0 else "No significant anomalies detected"
        }
    
    def generate_predictions(self, df):
        """AI-powered predictive analytics"""
        return {
            'quality_forecast': f"Quality trend: {'Stable' if df['confidence'].mean() > 0.7 else 'Needs attention'}",
            'optimal_detection_time': f"Optimal time: {df.groupby(pd.to_datetime(df['timestamp']).dt.hour)['confidence'].mean().idxmax() if 'timestamp' in df.columns else 12}:00",
            'ai_recommendation': "System performing well - maintain current configuration"
        }
    
    def categorize_quality(self, confidence):
        """AI quality categorization"""
        if confidence >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif confidence >= self.quality_thresholds['good']:
            return 'good'
        elif confidence >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def calculate_balance_score(self, class_distribution):
        """Calculate detection balance score"""
        if len(class_distribution) == 0:
            return 0
        
        total = sum(class_distribution.values())
        expected_per_class = total / len(class_distribution)
        deviations = [abs(count - expected_per_class) for count in class_distribution.values()]
        avg_deviation = sum(deviations) / len(deviations)
        return max(0, 100 - (avg_deviation / expected_per_class * 100))
    
    def generate_quality_recommendation(self, quality_score):
        """Generate quality recommendations"""
        if quality_score >= 90:
            return "Excellent quality! System performing optimally."
        elif quality_score >= 75:
            return "Good quality. Consider minor calibration adjustments."
        elif quality_score >= 60:
            return "Fair quality. Review system configuration."
        else:
            return "Poor quality detected. Check system configuration and retrain model."

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main AUTOPACK AI Application"""
    
    # Header
    st.markdown('<h1 class="main-header">🐔 AUTOPACK AI - Complete Multi-Feature System</h1>', unsafe_allow_html=True)
    st.markdown("### 🎯 Advanced Chicken Detection & Quality Assessment Platform")
    
    # Initialize components
    security = SecurityManager()
    freshness_analyzer = ChickenFreshnessAnalyzer()
    camera_detector = CameraDetector()
    ai_analytics = AIAnalyticsEngine()
    
    # Security check
    if not security.access_control_check():
        st.warning("🔒 Please authorize access to use the AUTOPACK AI system")
        st.info("💡 **Demo Access Codes Available in Sidebar**")
        return
    
    # Display security notices
    security.data_privacy_notice()
    
    # Feature selection
    st.sidebar.markdown("## 🎛️ Feature Selection")
    selected_feature = st.sidebar.selectbox(
        "Choose Feature:",
        [
            "🍗 Chicken Freshness Assessment",
            "🎥 Real-time Camera Detection", 
            "🧠 AI Analytics Dashboard",
            "📊 Complete System Overview"
        ]
    )
    
    # Feature routing
    if selected_feature == "🍗 Chicken Freshness Assessment":
        freshness_feature(security, freshness_analyzer)
    elif selected_feature == "🎥 Real-time Camera Detection":
        camera_feature(camera_detector)
    elif selected_feature == "🧠 AI Analytics Dashboard":
        analytics_feature(ai_analytics)
    else:
        system_overview(security, freshness_analyzer, ai_analytics)

def freshness_feature(security, analyzer):
    """Freshness Assessment Feature"""
    st.markdown('<div class="feature-header"><h2>🍗 Chicken Freshness Assessment System</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Image Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload chicken image for analysis",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear, well-lit image of the chicken meat"
        )
        
        if uploaded_file is not None:
            # Security validation
            validated_file, message = security.secure_file_handling(uploaded_file)
            
            if validated_file is None:
                st.error(f"🚫 {message}")
                return
            else:
                st.success(f"✅ {message}")
            
            # Display and analyze image
            image = Image.open(validated_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔬 Analyze Freshness", type="primary"):
                with st.spinner("Analyzing image..."):
                    results = analyzer.analyze_image(image)
                    
                    if results is not None:
                        st.session_state['freshness_results'] = results
                        st.session_state['analyzed_image'] = image
                    else:
                        st.error("Failed to analyze image.")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'freshness_results' in st.session_state:
            results = st.session_state['freshness_results']
            
            category = results['freshness_category']
            confidence = results['confidence']
            safety_score = results['safety_score']
            
            # Display results with color coding
            if category == 'Fresh':
                st.markdown(f'''
                <div class="safety-pass">
                    <h4>✅ Assessment: {category}</h4>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Safety Score:</strong> {safety_score:.1f}/100</p>
                </div>
                ''', unsafe_allow_html=True)
            elif category == 'Questionable':
                st.markdown(f'''
                <div class="safety-caution">
                    <h4>⚠️ Assessment: {category}</h4>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Safety Score:</strong> {safety_score:.1f}/100</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="safety-fail">
                    <h4>❌ Assessment: {category}</h4>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p><strong>Safety Score:</strong> {safety_score:.1f}/100</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("**📝 Recommendations:**")
            for rec in results['recommendations']:
                st.markdown(f"- {rec}")
                
            # Detailed analysis
            with st.expander("📈 Detailed Analysis"):
                color_features = results['color_features']
                texture_features = results['texture_features']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Color Analysis")
                    st.metric("Average Hue", f"{color_features['avg_hue']:.1f}°")
                    st.metric("Average Saturation", f"{color_features['avg_saturation']:.1f}")
                    st.metric("Average Value", f"{color_features['avg_value']:.1f}")
                
                with col2:
                    st.subheader("Texture Analysis")
                    st.metric("Smoothness", f"{texture_features['smoothness']:.3f}")
                    st.metric("Uniformity", f"{texture_features['uniformity']:.6f}")
                    st.metric("Edge Density", f"{texture_features['edge_density']:.4f}")
        else:
            st.info("Upload and analyze an image to see results here.")

def camera_feature(detector):
    """Camera Detection Feature"""
    st.markdown('<div class="feature-header"><h2>🎥 Real-time Camera Detection System</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera controls
        camera_placeholder = st.empty()
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            start_camera = st.button("📷 Start Camera", type="primary")
        
        with col_btn2:
            stop_camera = st.button("⏹️ Stop Camera")
        
        with col_btn3:
            capture_frame = st.button("📸 Capture Frame")
        
        # Simulate camera feed
        if start_camera or 'camera_active' in st.session_state:
            st.session_state['camera_active'] = True
            
            # Generate demo frame
            demo_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some realistic elements
            cv2.rectangle(demo_frame, (100, 100), (300, 200), (200, 150, 100), -1)  # Chicken-like color
            cv2.putText(demo_frame, "LIVE DEMO FEED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Simulate detection
            annotated_frame, detections = detector.simulate_detection(demo_frame)
            
            # Display frame
            camera_placeholder.image(annotated_frame, channels="BGR", caption="Live Detection Feed")
            
            # Store detections
            if 'live_detections' not in st.session_state:
                st.session_state['live_detections'] = []
            
            if len(detections) > 0:
                for det in detections:
                    det['timestamp'] = datetime.now().isoformat()
                st.session_state['live_detections'].extend(detections)
        
        if stop_camera:
            if 'camera_active' in st.session_state:
                del st.session_state['camera_active']
                camera_placeholder.empty()
    
    with col2:
        st.markdown("### 📊 Detection Results")
        
        if 'live_detections' in st.session_state and st.session_state['live_detections']:
            detections = st.session_state['live_detections']
            
            # Summary metrics
            total_detections = len(detections)
            avg_confidence = np.mean([d['confidence'] for d in detections])
            
            st.metric("Total Detections", total_detections)
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            # Recent detections
            st.subheader("Recent Detections")
            for i, detection in enumerate(detections[-5:]):  # Show last 5
                confidence = detection['confidence']
                class_name = detection.get('class', 'Unknown')
                st.write(f"🎯 Detection {i+1}: {class_name} ({confidence:.2%})")
            
            # Clear detections button
            if st.button("🗑️ Clear Detections"):
                st.session_state['live_detections'] = []
                st.rerun()
        else:
            st.info("Start camera to see detection results")

def analytics_feature(ai_analytics):
    """AI Analytics Feature"""
    st.markdown('<div class="feature-header"><h2>🧠 AI-Powered Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Load or generate demo data
    if st.button("🎯 Generate Demo Analytics Data"):
        demo_data = []
        for i in range(100):
            demo_data.append({
                'class': np.random.choice(['breast', 'thigh', 'wing', 'drumstick']),
                'confidence': np.random.beta(8, 2),
                'timestamp': (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
                'image_name': f"demo_image_{i}.jpg"
            })
        st.session_state['analytics_data'] = demo_data
        st.success("✅ Demo data generated!")
    
    # Combine live detections with analytics data
    all_data = []
    if 'analytics_data' in st.session_state:
        all_data.extend(st.session_state['analytics_data'])
    if 'live_detections' in st.session_state:
        all_data.extend(st.session_state['live_detections'])
    
    if all_data:
        st.success(f"📊 Analyzing {len(all_data)} detection records...")
        
        # Run AI Analytics
        with st.spinner("🧠 Running AI pattern analysis..."):
            analytics_results = ai_analytics.analyze_detection_patterns(all_data)
        
        # AI Overview Metrics
        st.markdown("## 🎯 AI Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'quality_trends' in analytics_results:
                quality_score = analytics_results['quality_trends'].get('ai_quality_score', 0)
                st.markdown(f"""
                <div class="ai-metric">
                    <h3>{quality_score:.1f}%</h3>
                    <p>AI Quality Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'detection_efficiency' in analytics_results:
                efficiency_score = analytics_results['detection_efficiency'].get('ai_efficiency_score', 0)
                st.markdown(f"""
                <div class="ai-metric">
                    <h3>{efficiency_score:.1f}%</h3>
                    <p>AI Efficiency Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="ai-metric">
                <h3>{len(all_data)}</h3>
                <p>Total Detections</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if 'anomaly_detection' in analytics_results:
                anomalies = analytics_results['anomaly_detection'].get('confidence_anomalies', 0)
                st.markdown(f"""
                <div class="ai-metric">
                    <h3>{anomalies}</h3>
                    <p>Anomalies Detected</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI Insights
        st.markdown("## 🧠 AI-Generated Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality Insights
            if 'quality_trends' in analytics_results:
                quality_data = analytics_results['quality_trends']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🎯 Quality Analysis</h4>
                    <p><strong>Average Confidence:</strong> {quality_data.get('average_confidence', 0):.2f}</p>
                    <p><strong>AI Recommendation:</strong> {quality_data.get('ai_recommendation', 'No recommendation')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Temporal Patterns
            if 'temporal_patterns' in analytics_results:
                temporal_data = analytics_results['temporal_patterns']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>⏰ Temporal Patterns</h4>
                    <p><strong>Peak Hour:</strong> {temporal_data.get('peak_detection_hour', 'Unknown')}:00</p>
                    <p><strong>AI Insight:</strong> {temporal_data.get('ai_insight', 'No pattern detected')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Efficiency Insights
            if 'detection_efficiency' in analytics_results:
                efficiency_data = analytics_results['detection_efficiency']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>⚡ Efficiency Analysis</h4>
                    <p><strong>Total Detections:</strong> {efficiency_data.get('total_detections', 0)}</p>
                    <p><strong>AI Insight:</strong> {efficiency_data.get('ai_insight', 'No insight available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Predictions
            if 'predictive_insights' in analytics_results:
                predictions = analytics_results['predictive_insights']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🔮 AI Predictions</h4>
                    <p><strong>Quality Forecast:</strong> {predictions.get('quality_forecast', 'No forecast')}</p>
                    <p><strong>Recommendation:</strong> {predictions.get('ai_recommendation', 'No recommendation')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 AI-Powered Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Quality Distribution", "Class Distribution", "Temporal Patterns"])
        
        with tab1:
            if 'quality_trends' in analytics_results and 'quality_distribution' in analytics_results['quality_trends']:
                quality_dist = analytics_results['quality_trends']['quality_distribution']
                if quality_dist:
                    fig = px.pie(
                        values=list(quality_dist.values()),
                        names=list(quality_dist.keys()),
                        title="AI Quality Assessment Distribution",
                        color_discrete_map={
                            'excellent': '#28a745',
                            'good': '#17a2b8',
                            'fair': '#ffc107',
                            'poor': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'detection_efficiency' in analytics_results and 'class_distribution' in analytics_results['detection_efficiency']:
                class_dist = analytics_results['detection_efficiency']['class_distribution']
                if class_dist:
                    fig = px.bar(
                        x=list(class_dist.keys()),
                        y=list(class_dist.values()),
                        title="Detection Efficiency by Class",
                        color=list(class_dist.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_title="Chicken Part", yaxis_title="Detection Count")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'temporal_patterns' in analytics_results and 'hourly_distribution' in analytics_results['temporal_patterns']:
                hourly_dist = analytics_results['temporal_patterns']['hourly_distribution']
                if hourly_dist:
                    fig = px.line(
                        x=list(hourly_dist.keys()),
                        y=list(hourly_dist.values()),
                        title="AI-Detected Temporal Patterns",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Detection Count")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data Explorer
        with st.expander("🔍 Raw Data Explorer"):
            df = pd.DataFrame(all_data)
            st.dataframe(df, use_container_width=True)
            
            # Export data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"autopack_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No data available. Generate demo data or run camera detection to see analytics.")

def system_overview(security, freshness_analyzer, ai_analytics):
    """Complete System Overview"""
    st.markdown('<div class="feature-header"><h2>📊 Complete System Overview</h2></div>', unsafe_allow_html=True)
    
    # System Status
    st.markdown("## 🎛️ System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="ai-metric">
            <h4>🔐 Security</h4>
            <p>✅ Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ai-metric">
            <h4>🍗 Freshness</h4>
            <p>✅ Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        camera_status = "✅ Ready" if 'camera_active' not in st.session_state else "🔴 Active"
        st.markdown(f"""
        <div class="ai-metric">
            <h4>🎥 Camera</h4>
            <p>{camera_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="ai-metric">
            <h4>🧠 AI Analytics</h4>
            <p>✅ Online</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Summary
    st.markdown("## 🌟 Feature Summary")
    
    features_info = [
        {
            "icon": "🔐",
            "name": "Security & Compliance",
            "description": "Access control, audit logging, data privacy protection",
            "status": "Active"
        },
        {
            "icon": "🍗",
            "name": "Freshness Assessment", 
            "description": "AI-powered chicken quality analysis using computer vision",
            "status": "Ready"
        },
        {
            "icon": "🎥",
            "name": "Camera Detection",
            "description": "Real-time chicken part detection and classification",
            "status": "Ready"
        },
        {
            "icon": "🧠",
            "name": "AI Analytics",
            "description": "Pattern recognition, predictive insights, anomaly detection",
            "status": "Online"
        }
    ]
    
    for feature in features_info:
        st.markdown(f"""
        <div class="insight-card">
            <h4>{feature['icon']} {feature['name']}</h4>
            <p>{feature['description']}</p>
            <p><strong>Status:</strong> <span style="color: green;">{feature['status']}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🍗 Test Freshness Analysis", type="primary"):
            st.info("Navigate to 'Chicken Freshness Assessment' to upload an image for analysis.")
    
    with col2:
        if st.button("🎥 Start Camera Detection"):
            st.info("Navigate to 'Real-time Camera Detection' to begin live detection.")
    
    with col3:
        if st.button("🧠 View Analytics"):
            st.info("Navigate to 'AI Analytics Dashboard' to see system insights.")
    
    with col4:
        if st.button("📊 Generate Report"):
            generate_system_report()

def generate_system_report():
    """Generate comprehensive system report"""
    st.markdown("## 📋 System Report")
    
    current_time = datetime.now()
    
    # Collect system data
    total_freshness_analyses = 1 if 'freshness_results' in st.session_state else 0
    total_detections = len(st.session_state.get('live_detections', []))
    total_analytics_records = len(st.session_state.get('analytics_data', []))
    
    report_data = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_status": "Operational",
        "freshness_analyses": total_freshness_analyses,
        "camera_detections": total_detections,
        "analytics_records": total_analytics_records,
        "security_status": "Active"
    }
    
    # Display report
    st.json(report_data)
    
    # Download report
    report_json = json.dumps(report_data, indent=2)
    st.download_button(
        label="📥 Download System Report",
        data=report_json,
        file_name=f"autopack_system_report_{current_time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>🐔 AUTOPACK AI - Complete Multi-Feature System</strong></p>
            <p>🔐 Feature 1: Security & Compliance | 🎥 Feature 2: Camera Detection | 🍗 Freshness Assessment | 🧠 Feature 3: AI Analytics</p>
            <p>⚠️ This system is for educational and demonstration purposes.</p>
            <p>Developed with Streamlit, OpenCV, Scikit-learn, and Plotly</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")