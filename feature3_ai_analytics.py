#!/usr/bin/env python3
"""
File: feature3_ai_analytics.py
Feature 3: AI-Powered Data Management & Analytics
AI Components: Pattern recognition, quality assessment, predictive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import cv2
from PIL import Image

class AIAnalyticsEngine:
    """AI-powered analytics for chicken detection data"""
    
    def __init__(self):
        self.detection_history = []
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        
    def analyze_detection_patterns(self, detection_data):
        """AI-powered pattern recognition in detection data"""
        
        if not detection_data:
            return {"error": "No data to analyze"}
        
        df = pd.DataFrame(detection_data)
        
        # AI Pattern Analysis
        patterns = {
            'temporal_patterns': self.detect_temporal_patterns(df),
            'quality_trends': self.analyze_quality_trends(df),
            'detection_efficiency': self.calculate_detection_efficiency(df),
            'anomaly_detection': self.detect_anomalies(df),
            'predictive_insights': self.generate_predictions(df)
        }
        
        return patterns
    
    def detect_temporal_patterns(self, df):
        """AI-powered temporal pattern detection"""
        
        if 'timestamp' not in df.columns:
            return {"error": "No timestamp data"}
        
        # Convert timestamps
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # Analyze patterns
        hourly_patterns = df.groupby('hour').size()
        daily_patterns = df.groupby('day_of_week').size()
        
        # AI Insights
        peak_hour = hourly_patterns.idxmax()
        peak_day = daily_patterns.idxmax()
        
        patterns = {
            'peak_detection_hour': int(peak_hour),
            'peak_detection_day': peak_day,
            'hourly_distribution': hourly_patterns.to_dict(),
            'daily_distribution': daily_patterns.to_dict(),
            'ai_insight': f"Peak detection activity at {peak_hour}:00 on {peak_day}s"
        }
        
        return patterns
    
    def analyze_quality_trends(self, df):
        """AI quality assessment and trending"""
        
        if 'confidence' not in df.columns:
            return {"error": "No confidence data"}
        
        # Quality categorization using AI thresholds
        df['quality_category'] = df['confidence'].apply(self.categorize_quality)
        
        # Trend analysis
        quality_distribution = df['quality_category'].value_counts()
        avg_confidence = df['confidence'].mean()
        confidence_trend = df['confidence'].rolling(window=10).mean()
        
        # AI Quality Score (weighted algorithm)
        quality_weights = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        weighted_quality = sum(quality_distribution.get(cat, 0) * weight 
                             for cat, weight in quality_weights.items())
        total_detections = len(df)
        ai_quality_score = (weighted_quality / (total_detections * 4)) * 100 if total_detections > 0 else 0
        
        return {
            'average_confidence': float(avg_confidence),
            'quality_distribution': quality_distribution.to_dict(),
            'ai_quality_score': float(ai_quality_score),
            'trend_direction': 'improving' if confidence_trend.iloc[-1] > confidence_trend.iloc[0] else 'declining',
            'ai_recommendation': self.generate_quality_recommendation(ai_quality_score)
        }
    
    def calculate_detection_efficiency(self, df):
        """AI-powered efficiency analysis"""
        
        # Efficiency metrics
        total_detections = len(df)
        unique_sessions = df['timestamp'].nunique() if 'timestamp' in df.columns else 1
        
        if 'class' in df.columns:
            class_distribution = df['class'].value_counts()
            detection_balance = self.calculate_balance_score(class_distribution)
        else:
            class_distribution = {}
            detection_balance = 0
        
        # AI Efficiency Score
        balance_factor = detection_balance / 100
        volume_factor = min(total_detections / 100, 1.0)  # Normalize to 0-1
        
        ai_efficiency_score = (balance_factor * 0.6 + volume_factor * 0.4) * 100
        
        return {
            'total_detections': total_detections,
            'detections_per_session': total_detections / unique_sessions,
            'class_distribution': class_distribution.to_dict(),
            'detection_balance_score': detection_balance,
            'ai_efficiency_score': float(ai_efficiency_score),
            'ai_insight': self.generate_efficiency_insight(ai_efficiency_score, class_distribution)
        }
    
    def detect_anomalies(self, df):
        """AI anomaly detection in detection patterns"""
        
        if len(df) < 10:
            return {"message": "Insufficient data for anomaly detection"}
        
        # Confidence anomalies
        confidence_mean = df['confidence'].mean()
        confidence_std = df['confidence'].std()
        confidence_threshold = confidence_mean - 2 * confidence_std
        
        anomalous_detections = df[df['confidence'] < confidence_threshold]
        
        # Temporal anomalies (if timestamp available)
        temporal_anomalies = []
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            detection_intervals = df['datetime'].diff().dt.total_seconds()
            interval_mean = detection_intervals.mean()
            interval_std = detection_intervals.std()
            
            unusual_intervals = detection_intervals[
                (detection_intervals > interval_mean + 2 * interval_std) |
                (detection_intervals < interval_mean - 2 * interval_std)
            ]
            temporal_anomalies = unusual_intervals.tolist()
        
        return {
            'confidence_anomalies': len(anomalous_detections),
            'temporal_anomalies': len(temporal_anomalies),
            'anomaly_threshold': float(confidence_threshold),
            'ai_alert': f"Found {len(anomalous_detections)} low-confidence anomalies" if len(anomalous_detections) > 0 else "No significant anomalies detected"
        }
    
    def generate_predictions(self, df):
        """AI-powered predictive analytics"""
        
        if len(df) < 5:
            return {"message": "Insufficient data for predictions"}
        
        # Simple trend prediction based on recent data
        recent_data = df.tail(10)
        
        predictions = {
            'next_hour_detections': self.predict_next_period_volume(df),
            'quality_forecast': self.predict_quality_trend(recent_data),
            'optimal_detection_time': self.predict_optimal_time(df),
            'ai_recommendation': self.generate_actionable_insights(df)
        }
        
        return predictions
    
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
        """Calculate how balanced the detection distribution is"""
        if len(class_distribution) == 0:
            return 0
        
        # Perfect balance would be equal distribution
        total = sum(class_distribution.values())
        expected_per_class = total / len(class_distribution)
        
        # Calculate deviation from perfect balance
        deviations = [abs(count - expected_per_class) for count in class_distribution.values()]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to 0-100 score (lower deviation = higher score)
        balance_score = max(0, 100 - (avg_deviation / expected_per_class * 100))
        
        return balance_score
    
    def generate_quality_recommendation(self, quality_score):
        """AI-generated quality recommendations"""
        if quality_score >= 90:
            return "Excellent quality! System performing optimally."
        elif quality_score >= 75:
            return "Good quality. Consider minor calibration adjustments."
        elif quality_score >= 60:
            return "Fair quality. Review lighting conditions and camera positioning."
        else:
            return "Poor quality detected. Check system configuration and retrain model."
    
    def generate_efficiency_insight(self, efficiency_score, class_distribution):
        """AI-generated efficiency insights"""
        if efficiency_score >= 80:
            return "High efficiency system with balanced detection across all classes."
        elif efficiency_score >= 60:
            return f"Moderate efficiency. Consider improving detection balance for {min(class_distribution, key=class_distribution.get)} class."
        else:
            return "Low efficiency detected. Review system configuration and detection parameters."
    
    def predict_next_period_volume(self, df):
        """Predict detection volume for next period"""
        if len(df) < 3:
            return "Insufficient data"
        
        recent_volumes = df.groupby(df['timestamp'].str[:10]).size().tail(3)
        if len(recent_volumes) > 0:
            avg_volume = recent_volumes.mean()
            return f"Predicted: {int(avg_volume)} detections"
        return "No trend data available"
    
    def predict_quality_trend(self, recent_data):
        """Predict quality trend"""
        if 'confidence' not in recent_data.columns or len(recent_data) < 3:
            return "Insufficient data"
        
        recent_avg = recent_data['confidence'].mean()
        if recent_avg > 0.8:
            return "Quality trend: Stable high performance"
        elif recent_avg > 0.6:
            return "Quality trend: Good performance with room for improvement"
        else:
            return "Quality trend: Performance needs attention"
    
    def predict_optimal_time(self, df):
        """Predict optimal detection times"""
        if 'timestamp' not in df.columns:
            return "No temporal data available"
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_avg_confidence = df.groupby('hour')['confidence'].mean()
        
        if len(hourly_avg_confidence) > 0:
            optimal_hour = hourly_avg_confidence.idxmax()
            return f"Optimal detection time: {optimal_hour}:00 hours"
        
        return "Insufficient temporal data"
    
    def generate_actionable_insights(self, df):
        """Generate AI-powered actionable insights"""
        insights = []
        
        if 'confidence' in df.columns:
            avg_confidence = df['confidence'].mean()
            if avg_confidence < 0.7:
                insights.append("Consider retraining model with more diverse data")
            
            low_confidence_count = len(df[df['confidence'] < 0.5])
            if low_confidence_count > len(df) * 0.2:
                insights.append("High rate of low-confidence detections - check camera setup")
        
        if 'class' in df.columns:
            class_counts = df['class'].value_counts()
            if len(class_counts) > 1:
                min_class = class_counts.idxmin()
                if class_counts[min_class] < class_counts.max() * 0.3:
                    insights.append(f"Underrepresented class: {min_class} - collect more training data")
        
        if not insights:
            insights.append("System performing well - maintain current configuration")
        
        return " | ".join(insights)

def load_detection_data():
    """Load detection data from various sources"""
    
    # Check for existing CSV files
    results_dir = Path("results")
    test_results_dir = Path("test_results")
    
    all_data = []
    
    # Load from results directory
    if results_dir.exists():
        for csv_file in results_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                st.warning(f"Could not load {csv_file}: {e}")
    
    # Load from test results directory
    if test_results_dir.exists():
        for csv_file in test_results_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                st.warning(f"Could not load {csv_file}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df.to_dict('records')
    
    return []

def create_ai_visualizations(analytics_results):
    """Create AI-powered visualizations"""
    
    visualizations = {}
    
    # Quality Trends Chart
    if 'quality_trends' in analytics_results:
        quality_data = analytics_results['quality_trends']
        if 'quality_distribution' in quality_data:
            quality_dist = quality_data['quality_distribution']
            
            fig_quality = px.pie(
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
            visualizations['quality_pie'] = fig_quality
    
    # Detection Efficiency Chart
    if 'detection_efficiency' in analytics_results:
        efficiency_data = analytics_results['detection_efficiency']
        if 'class_distribution' in efficiency_data:
            class_dist = efficiency_data['class_distribution']
            
            fig_efficiency = px.bar(
                x=list(class_dist.keys()),
                y=list(class_dist.values()),
                title="Detection Efficiency by Class",
                color=list(class_dist.values()),
                color_continuous_scale='viridis'
            )
            visualizations['efficiency_bar'] = fig_efficiency
    
    # Temporal Patterns
    if 'temporal_patterns' in analytics_results:
        temporal_data = analytics_results['temporal_patterns']
        if 'hourly_distribution' in temporal_data:
            hourly_dist = temporal_data['hourly_distribution']
            
            fig_temporal = px.line(
                x=list(hourly_dist.keys()),
                y=list(hourly_dist.values()),
                title="AI-Detected Temporal Patterns",
                markers=True
            )
            fig_temporal.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Detection Count"
            )
            visualizations['temporal_line'] = fig_temporal
    
    return visualizations

def main():
    """Feature 3 Main Application"""
    
    st.set_page_config(
        page_title="🧠 AI Analytics Dashboard", 
        page_icon="🧠",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
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
    
    .ai-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🧠 AI-Powered Data Analytics Dashboard")
    st.markdown("### Feature 3: Smart Data Management & Pattern Recognition")
    
    # AI Engine
    ai_engine = AIAnalyticsEngine()
    
    # Load data
    with st.spinner("🔍 Loading detection data..."):
        detection_data = load_detection_data()
    
    if not detection_data:
        st.warning("⚠️ No detection data found. Run some detections first!")
        
        # Demo data option
        if st.button("🎯 Generate Demo Data"):
            # Create sample data for demo
            demo_data = []
            for i in range(50):
                demo_data.append({
                    'class': np.random.choice(['breast', 'thigh', 'wing', 'drumstick']),
                    'confidence': np.random.beta(8, 2),  # Skewed towards higher confidence
                    'timestamp': (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
                    'image_name': f"demo_image_{i}.jpg"
                })
            detection_data = demo_data
            st.success("✅ Demo data generated!")
    
    if detection_data:
        
        st.success(f"📊 Analyzing {len(detection_data)} detection records...")
        
        # Run AI Analytics
        with st.spinner("🧠 Running AI pattern analysis..."):
            analytics_results = ai_engine.analyze_detection_patterns(detection_data)
        
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
            total_detections = len(detection_data)
            st.markdown(f"""
            <div class="ai-metric">
                <h3>{total_detections}</h3>
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
        
        # AI Insights Section
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
            
            # Efficiency Insights  
            if 'detection_efficiency' in analytics_results:
                efficiency_data = analytics_results['detection_efficiency']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>⚡ Efficiency Analysis</h4>
                    <p><strong>Detections per Session:</strong> {efficiency_data.get('detections_per_session', 0):.1f}</p>
                    <p><strong>AI Insight:</strong> {efficiency_data.get('ai_insight', 'No insight available')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
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
            
            # Anomaly Detection
            if 'anomaly_detection' in analytics_results:
                anomaly_data = analytics_results['anomaly_detection']
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🚨 Anomaly Detection</h4>
                    <p><strong>Alert:</strong> {anomaly_data.get('ai_alert', 'No alerts')}</p>
                    <p><strong>Threshold:</strong> {anomaly_data.get('anomaly_threshold', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI Predictions
        if 'predictive_insights' in analytics_results:
            st.markdown("## 🔮 AI Predictions & Recommendations")
            
            predictions = analytics_results['predictive_insights']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>📈 Volume Forecast</h4>
                    <p>{predictions.get('next_hour_detections', 'No prediction available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🎯 Quality Forecast</h4>
                    <p>{predictions.get('quality_forecast', 'No forecast available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>⏰ Optimal Time</h4>
                    <p>{predictions.get('optimal_detection_time', 'No data available')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Actionable Recommendations
            if 'ai_recommendation' in predictions:
                st.markdown(f"""
                <div class="ai-alert">
                    <h4>🤖 AI Recommendations</h4>
                    <p>{predictions['ai_recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 AI-Powered Visualizations")
        
        visualizations = create_ai_visualizations(analytics_results)
        
        # Display charts in tabs
        if visualizations:
            tab1, tab2, tab3 = st.tabs(["Quality Analysis", "Efficiency Metrics", "Temporal Patterns"])
            
            with tab1:
                if 'quality_pie' in visualizations:
                    st.plotly_chart(visualizations['quality_pie'], use_container_width=True)
            
            with tab2:
                if 'efficiency_bar' in visualizations:
                    st.plotly_chart(visualizations['efficiency_bar'], use_container_width=True)
            
            with tab3:
                if 'temporal_line' in visualizations:
                    st.plotly_chart(visualizations['temporal_line'], use_container_width=True)
        
        # Raw Data Explorer
        with st.expander("🔍 Raw Data Explorer"):
            df = pd.DataFrame(detection_data)
            st.dataframe(df, use_container_width=True)
            
            # Data export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"ai_analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()