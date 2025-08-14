#!/usr/bin/env python3
"""
File: feature4_ai_production.py
Feature 4: AI-Enhanced Production System
AI Components: Automated quality control, smart alerts, AI reporting, predictive maintenance
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
import time
import threading
import queue

class AIProductionEngine:
    """AI-enhanced production system with automation"""
    
    def __init__(self):
        self.quality_control_ai = AIQualityController()
        self.alert_system = AIAlertSystem()
        self.reporting_ai = AIReportingEngine()
        self.maintenance_ai = AIPredictiveMaintenance()
        
        # Production metrics
        self.production_metrics = {
            'total_processed': 0,
            'quality_score': 0,
            'efficiency_rate': 0,
            'error_rate': 0,
            'uptime': 0
        }
        
        # AI thresholds
        self.ai_thresholds = {
            'quality_alert': 0.7,
            'efficiency_alert': 0.8,
            'error_rate_alert': 0.05,
            'maintenance_prediction': 0.9
        }

class AIQualityController:
    """AI-powered automated quality control"""
    
    def __init__(self):
        self.quality_history = []
        self.rejection_reasons = []
        
    def automated_quality_check(self, detection_result):
        """AI-powered automated quality assessment"""
        
        quality_assessment = {
            'timestamp': datetime.now().isoformat(),
            'detection_confidence': detection_result.get('confidence', 0),
            'classification': detection_result.get('class', 'unknown'),
            'image_quality': self.assess_image_quality(detection_result),
            'ai_quality_score': 0,
            'pass_fail': False,
            'rejection_reason': None,
            'recommended_action': None
        }
        
        # AI Quality Scoring Algorithm
        confidence_score = detection_result.get('confidence', 0) * 40  # 40% weight
        image_quality_score = quality_assessment['image_quality'] * 30  # 30% weight
        consistency_score = self.calculate_consistency_score() * 20  # 20% weight
        historical_score = self.calculate_historical_performance() * 10  # 10% weight
        
        quality_assessment['ai_quality_score'] = (
            confidence_score + image_quality_score + 
            consistency_score + historical_score
        )
        
        # AI Decision Making
        if quality_assessment['ai_quality_score'] >= 85:
            quality_assessment['pass_fail'] = True
            quality_assessment['recommended_action'] = "PASS - High quality detection"
        elif quality_assessment['ai_quality_score'] >= 70:
            quality_assessment['pass_fail'] = True
            quality_assessment['recommended_action'] = "PASS - Acceptable quality with monitoring"
        else:
            quality_assessment['pass_fail'] = False
            quality_assessment['rejection_reason'] = self.determine_rejection_reason(quality_assessment)
            quality_assessment['recommended_action'] = f"REJECT - {quality_assessment['rejection_reason']}"
        
        # Store for learning
        self.quality_history.append(quality_assessment)
        
        return quality_assessment
    
    def assess_image_quality(self, detection_result):
        """AI assessment of image quality factors"""
        
        # Simulated image quality metrics (in real system, would analyze actual image)
        quality_factors = {
            'brightness': np.random.normal(75, 10),  # 0-100
            'contrast': np.random.normal(80, 15),    # 0-100
            'sharpness': np.random.normal(85, 12),   # 0-100
            'noise_level': np.random.normal(20, 8)   # 0-100 (lower is better)
        }
        
        # AI quality score calculation
        brightness_score = min(100, max(0, 100 - abs(quality_factors['brightness'] - 75)))
        contrast_score = min(100, quality_factors['contrast'])
        sharpness_score = min(100, quality_factors['sharpness'])
        noise_score = min(100, 100 - quality_factors['noise_level'])
        
        overall_quality = (brightness_score + contrast_score + sharpness_score + noise_score) / 4
        
        return overall_quality
    
    def calculate_consistency_score(self):
        """AI consistency analysis"""
        if len(self.quality_history) < 5:
            return 80  # Default for new system
        
        recent_scores = [q['ai_quality_score'] for q in self.quality_history[-10:]]
        consistency = 100 - (np.std(recent_scores) * 2)  # Lower std = higher consistency
        
        return max(0, min(100, consistency))
    
    def calculate_historical_performance(self):
        """AI historical performance analysis"""
        if len(self.quality_history) < 10:
            return 75  # Default for new system
        
        recent_performance = [q['ai_quality_score'] for q in self.quality_history[-20:]]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Positive trend = improving performance
        historical_score = 75 + (trend * 5)  # Scale trend impact
        
        return max(0, min(100, historical_score))
    
    def determine_rejection_reason(self, quality_assessment):
        """AI-powered rejection reason determination"""
        
        if quality_assessment['detection_confidence'] < 0.5:
            return "Low detection confidence - unclear classification"
        elif quality_assessment['image_quality'] < 60:
            return "Poor image quality - inadequate lighting or focus"
        elif quality_assessment['ai_quality_score'] < 50:
            return "Multiple quality issues detected"
        else:
            return "Below quality threshold - requires manual review"

class AIAlertSystem:
    """AI-powered smart alert system"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_rules = {
            'quality_degradation': {'threshold': 0.7, 'window': 10, 'severity': 'high'},
            'efficiency_drop': {'threshold': 0.8, 'window': 15, 'severity': 'medium'},
            'error_spike': {'threshold': 0.05, 'window': 5, 'severity': 'high'},
            'anomaly_detected': {'threshold': 0.9, 'window': 1, 'severity': 'critical'}
        }
    
    def monitor_and_alert(self, production_data):
        """AI monitoring and intelligent alerting"""
        
        alerts = []
        
        # Quality degradation detection
        quality_alert = self.check_quality_degradation(production_data)
        if quality_alert:
            alerts.append(quality_alert)
        
        # Efficiency monitoring
        efficiency_alert = self.check_efficiency_drop(production_data)
        if efficiency_alert:
            alerts.append(efficiency_alert)
        
        # Error rate monitoring
        error_alert = self.check_error_spike(production_data)
        if error_alert:
            alerts.append(error_alert)
        
        # Anomaly detection
        anomaly_alert = self.check_anomalies(production_data)
        if anomaly_alert:
            alerts.append(anomaly_alert)
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def check_quality_degradation(self, data):
        """AI quality degradation detection"""
        
        if len(data) < 10:
            return None
        
        recent_quality = [d.get('quality_score', 80) for d in data[-10:]]
        avg_quality = np.mean(recent_quality)
        
        if avg_quality < self.alert_rules['quality_degradation']['threshold'] * 100:
            return {
                'type': 'quality_degradation',
                'severity': 'high',
                'message': f"Quality degradation detected: {avg_quality:.1f}% average quality",
                'recommendation': "Check camera setup, lighting, and model performance",
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_efficiency_drop(self, data):
        """AI efficiency monitoring"""
        
        if len(data) < 15:
            return None
        
        recent_efficiency = [d.get('efficiency_rate', 0.9) for d in data[-15:]]
        avg_efficiency = np.mean(recent_efficiency)
        
        if avg_efficiency < self.alert_rules['efficiency_drop']['threshold']:
            return {
                'type': 'efficiency_drop',
                'severity': 'medium',
                'message': f"Efficiency drop detected: {avg_efficiency:.1%} average efficiency",
                'recommendation': "Review detection parameters and system load",
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_error_spike(self, data):
        """AI error rate monitoring"""
        
        if len(data) < 5:
            return None
        
        recent_errors = [d.get('error_rate', 0) for d in data[-5:]]
        avg_error_rate = np.mean(recent_errors)
        
        if avg_error_rate > self.alert_rules['error_spike']['threshold']:
            return {
                'type': 'error_spike',
                'severity': 'high',
                'message': f"Error rate spike detected: {avg_error_rate:.1%} error rate",
                'recommendation': "Investigate system errors and check model stability",
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_anomalies(self, data):
        """AI anomaly detection"""
        
        if len(data) < 20:
            return None
        
        # Simple anomaly detection based on detection patterns
        recent_detections = [len(d.get('detections', [])) for d in data[-20:]]
        mean_detections = np.mean(recent_detections)
        std_detections = np.std(recent_detections)
        
        latest_detections = recent_detections[-1]
        
        # Check if latest is significantly different
        if abs(latest_detections - mean_detections) > 2 * std_detections:
            return {
                'type': 'anomaly_detected',
                'severity': 'critical',
                'message': f"Anomalous detection pattern: {latest_detections} vs average {mean_detections:.1f}",
                'recommendation': "Investigate unusual detection patterns - possible system issue",
                'timestamp': datetime.now().isoformat()
            }
        
        return None

class AIReportingEngine:
    """AI-powered automated reporting"""
    
    def __init__(self):
        self.report_templates = {
            'daily': self.generate_daily_report,
            'weekly': self.generate_weekly_report,
            'quality': self.generate_quality_report,
            'efficiency': self.generate_efficiency_report
        }
    
    def generate_ai_report(self, report_type, data):
        """Generate AI-powered reports"""
        
        if report_type in self.report_templates:
            return self.report_templates[report_type](data)
        else:
            return {"error": f"Unknown report type: {report_type}"}
    
    def generate_daily_report(self, data):
        """AI-generated daily production report"""
        
        if not data:
            return {"error": "No data available for report"}
        
        # Calculate daily metrics
        total_processed = len(data)
        avg_quality = np.mean([d.get('quality_score', 80) for d in data])
        avg_efficiency = np.mean([d.get('efficiency_rate', 0.9) for d in data])
        error_rate = np.mean([d.get('error_rate', 0) for d in data])
        
        # AI insights
        quality_trend = self.analyze_trend([d.get('quality_score', 80) for d in data])
        efficiency_trend = self.analyze_trend([d.get('efficiency_rate', 0.9) for d in data])
        
        # AI recommendations
        recommendations = self.generate_daily_recommendations(avg_quality, avg_efficiency, error_rate)
        
        report = {
            'report_type': 'Daily Production Report',
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_processed': total_processed,
                'average_quality': avg_quality,
                'average_efficiency': avg_efficiency,
                'error_rate': error_rate
            },
            'ai_insights': {
                'quality_trend': quality_trend,
                'efficiency_trend': efficiency_trend,
                'performance_grade': self.calculate_performance_grade(avg_quality, avg_efficiency, error_rate)
            },
            'ai_recommendations': recommendations,
            'next_actions': self.prioritize_actions(recommendations)
        }
        
        return report
    
    def generate_quality_report(self, data):
        """AI-powered quality analysis report"""
        
        if not data:
            return {"error": "No quality data available"}
        
        quality_scores = [d.get('quality_score', 80) for d in data]
        
        report = {
            'report_type': 'AI Quality Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'quality_metrics': {
                'average_score': np.mean(quality_scores),
                'median_score': np.median(quality_scores),
                'quality_consistency': 100 - np.std(quality_scores),
                'quality_trend': self.analyze_trend(quality_scores)
            },
            'quality_distribution': self.analyze_quality_distribution(quality_scores),
            'ai_recommendations': self.generate_quality_recommendations(quality_scores),
            'predictive_insights': self.predict_quality_future(quality_scores)
        }
        
        return report
    
    def analyze_trend(self, values):
        """AI trend analysis"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def calculate_performance_grade(self, quality, efficiency, error_rate):
        """AI performance grading"""
        
        # Weighted scoring
        quality_score = (quality / 100) * 40  # 40% weight
        efficiency_score = efficiency * 35    # 35% weight
        error_score = (1 - error_rate) * 25   # 25% weight (inverted)
        
        total_score = quality_score + efficiency_score + error_score
        
        if total_score >= 90:
            return "A+ (Excellent)"
        elif total_score >= 80:
            return "A (Very Good)"
        elif total_score >= 70:
            return "B (Good)"
        elif total_score >= 60:
            return "C (Fair)"
        else:
            return "D (Needs Improvement)"
    
    def generate_daily_recommendations(self, quality, efficiency, error_rate):
        """AI-generated daily recommendations"""
        
        recommendations = []
        
        if quality < 80:
            recommendations.append("Quality below target - review detection model and camera setup")
        
        if efficiency < 0.85:
            recommendations.append("Efficiency below target - optimize processing pipeline")
        
        if error_rate > 0.03:
            recommendations.append("Error rate elevated - investigate system stability")
        
        if not recommendations:
            recommendations.append("System performing well - maintain current configuration")
        
        return recommendations
    
    def prioritize_actions(self, recommendations):
        """AI action prioritization"""
        
        priority_keywords = {
            'error': 1,      # Highest priority
            'quality': 2,    # High priority
            'efficiency': 3, # Medium priority
            'maintain': 4    # Low priority
        }
        
        prioritized = []
        for rec in recommendations:
            priority = 4  # Default low priority
            for keyword, p in priority_keywords.items():
                if keyword in rec.lower():
                    priority = min(priority, p)
                    break
            
            prioritized.append({
                'action': rec,
                'priority': priority,
                'priority_label': ['Critical', 'High', 'Medium', 'Low'][priority-1]
            })
        
        return sorted(prioritized, key=lambda x: x['priority'])

class AIPredictiveMaintenance:
    """AI-powered predictive maintenance"""
    
    def __init__(self):
        self.maintenance_history = []
        self.system_health_metrics = []
    
    def predict_maintenance_needs(self, system_data):
        """AI predictive maintenance analysis"""
        
        # Simulated system health metrics
        health_metrics = {
            'cpu_usage': np.random.normal(45, 10),
            'memory_usage': np.random.normal(60, 15),
            'camera_temperature': np.random.normal(35, 5),
            'detection_latency': np.random.normal(100, 20),
            'model_accuracy': np.random.normal(91, 3)
        }
        
        # AI health scoring
        health_score = self.calculate_health_score(health_metrics)
        
        # Predictive analysis
        maintenance_prediction = {
            'overall_health_score': health_score,
            'health_metrics': health_metrics,
            'maintenance_urgency': self.assess_maintenance_urgency(health_score),
            'predicted_issues': self.predict_potential_issues(health_metrics),
            'recommended_actions': self.generate_maintenance_recommendations(health_metrics),
            'next_maintenance_window': self.predict_next_maintenance(health_score)
        }
        
        return maintenance_prediction
    
    def calculate_health_score(self, metrics):
        """AI system health scoring"""
        
        # Normalize and weight metrics
        cpu_score = max(0, 100 - metrics['cpu_usage']) * 0.2
        memory_score = max(0, 100 - metrics['memory_usage']) * 0.2
        temp_score = max(0, 100 - max(0, metrics['camera_temperature'] - 25) * 2) * 0.15
        latency_score = max(0, 100 - max(0, metrics['detection_latency'] - 50) * 0.5) * 0.25
        accuracy_score = metrics['model_accuracy'] * 0.2
        
        total_score = cpu_score + memory_score + temp_score + latency_score + accuracy_score
        
        return min(100, max(0, total_score))
    
    def assess_maintenance_urgency(self, health_score):
        """AI maintenance urgency assessment"""
        
        if health_score >= 90:
            return "low"
        elif health_score >= 75:
            return "medium"
        elif health_score >= 60:
            return "high"
        else:
            return "critical"
    
    def predict_potential_issues(self, metrics):
        """AI issue prediction"""
        
        issues = []
        
        if metrics['cpu_usage'] > 70:
            issues.append("High CPU usage - potential performance degradation")
        
        if metrics['memory_usage'] > 80:
            issues.append("High memory usage - risk of system instability")
        
        if metrics['camera_temperature'] > 40:
            issues.append("Camera overheating - potential hardware damage")
        
        if metrics['detection_latency'] > 150:
            issues.append("High detection latency - user experience impact")
        
        if metrics['model_accuracy'] < 88:
            issues.append("Model accuracy degradation - retraining may be needed")
        
        return issues if issues else ["No immediate issues predicted"]
    
    def generate_maintenance_recommendations(self, metrics):
        """AI maintenance recommendations"""
        
        recommendations = []
        
        if metrics['cpu_usage'] > 60:
            recommendations.append("Monitor CPU usage and consider process optimization")
        
        if metrics['memory_usage'] > 70:
            recommendations.append("Check for memory leaks and restart services if needed")
        
        if metrics['camera_temperature'] > 35:
            recommendations.append("Improve ventilation around camera equipment")
        
        if metrics['detection_latency'] > 120:
            recommendations.append("Optimize detection pipeline and check network connectivity")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations

def main():
    """Feature 4 Main Application"""
    
    st.set_page_config(
        page_title="🏭 AI Production System",
        page_icon="🏭", 
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .production-metric {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #fd7e14 0%, #e8741b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        padding: 1rem;
        border-radius: 10px;
        color: black;
    }
    
    .maintenance-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #6610f2;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🏭 AI-Enhanced Production System")
    st.markdown("### Feature 4: Automated Quality Control & Smart Operations")
    
    # Initialize AI systems
    production_engine = AIProductionEngine()
    
    # Simulate production data
    if 'production_data' not in st.session_state:
        st.session_state.production_data = []
    
    # Production Dashboard
    st.markdown("## 🎛️ Live Production Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Simulate live metrics
    current_metrics = {
        'processed_today': len(st.session_state.production_data),
        'quality_score': np.random.normal(87, 5),
        'efficiency_rate': np.random.normal(0.92, 0.05),
        'error_rate': np.random.normal(0.02, 0.01),
        'system_health': np.random.normal(88, 8)
    }
    
    with col1:
        st.markdown(f"""
        <div class="production-metric">
            <h3>{current_metrics['processed_today']}</h3>
            <p>Processed Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="production-metric">
            <h3>{current_metrics['quality_score']:.1f}%</h3>
            <p>AI Quality Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="production-metric">
            <h3>{current_metrics['efficiency_rate']:.1%}</h3>
            <p>Efficiency Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="production-metric">
            <h3>{current_metrics['error_rate']:.1%}</h3>
            <p>Error Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="production-metric">
            <h3>{current_metrics['system_health']:.0f}%</h3>
            <p>System Health</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Monitoring Section
    st.markdown("## 🤖 AI Monitoring & Alerts")
    
    # Generate sample alerts
    sample_production_data = [current_metrics] * 20
    alerts = production_engine.alert_system.monitor_and_alert(sample_production_data)
    
    if alerts:
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            st.markdown(f"""
            <div class="{severity_class}">
                <h4>🚨 {alert['type'].replace('_', ' ').title()} Alert</h4>
                <p><strong>Message:</strong> {alert['message']}</p>
                <p><strong>Recommendation:</strong> {alert['recommendation']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ **All Systems Normal** - No alerts detected")
    
    # AI Quality Control Section
    st.markdown("## 🎯 AI Quality Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Automated Quality Assessment")
        
        # Simulate quality check
        sample_detection = {
            'class': 'drumstick',
            'confidence': np.random.beta(8, 2),
            'bbox': [100, 100, 200, 200]
        }
        
        quality_result = production_engine.quality_control_ai.automated_quality_check(sample_detection)
        
        if quality_result['pass_fail']:
            st.success(f"✅ **PASS** - Quality Score: {quality_result['ai_quality_score']:.1f}")
        else:
            st.error(f"❌ **REJECT** - Quality Score: {quality_result['ai_quality_score']:.1f}")
        
        st.write(f"**Recommendation:** {quality_result['recommended_action']}")
        
        if quality_result['rejection_reason']:
            st.warning(f"**Rejection Reason:** {quality_result['rejection_reason']}")
    
    with col2:
        st.markdown("### Quality Metrics")
        
        # Quality metrics chart
        quality_scores = [np.random.normal(85, 10) for _ in range(24)]
        hours = list(range(24))
        
        fig_quality = px.line(
            x=hours, 
            y=quality_scores,
            title="24-Hour Quality Trend",
            labels={'x': 'Hour', 'y': 'Quality Score'}
        )
        fig_quality.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Minimum Threshold")
        
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # AI Reporting Section
    st.markdown("## 📊 AI-Generated Reports")
    
    tab1, tab2, tab3 = st.tabs(["Daily Report", "Quality Analysis", "Efficiency Report"])
    
    with tab1:
        if st.button("🔄 Generate Daily Report"):
            with st.spinner("🧠 AI generating daily report..."):
                sample_data = [current_metrics] * 50
                daily_report = production_engine.reporting_ai.generate_daily_report(sample_data)
                
                st.markdown("### 📋 AI Daily Production Report")
                
                # Summary metrics
                summary = daily_report['summary']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Processed", summary['total_processed'])
                with col2:
                    st.metric("Avg Quality", f"{summary['average_quality']:.1f}%")
                with col3:
                    st.metric("Avg Efficiency", f"{summary['average_efficiency']:.1%}")
                with col4:
                    st.metric("Error Rate", f"{summary['error_rate']:.1%}")
                
                # AI Insights
                insights = daily_report['ai_insights']
                st.markdown("### 🧠 AI Insights")
                st.write(f"**Quality Trend:** {insights['quality_trend']}")
                st.write(f"**Efficiency Trend:** {insights['efficiency_trend']}")
                st.write(f"**Performance Grade:** {insights['performance_grade']}")
                
                # Recommendations
                st.markdown("### 💡 AI Recommendations")
                for rec in daily_report['ai_recommendations']:
                    st.write(f"• {rec}")
                
                # Next Actions
                st.markdown("### 🎯 Prioritized Actions")
                for action in daily_report['next_actions']:
                    priority_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                    st.write(f"{priority_color.get(action['priority_label'], '⚪')} **{action['priority_label']}:** {action['action']}")
    
    with tab2:
        st.markdown("### 🎯 Quality Analysis Dashboard")
        
        # Quality distribution
        quality_categories = ['Excellent', 'Good', 'Fair', 'Poor']
        quality_counts = [45, 30, 20, 5]
        
        fig_quality_dist = px.pie(
            values=quality_counts,
            names=quality_categories,
            title="Quality Distribution",
            color_discrete_map={
                'Excellent': '#28a745',
                'Good': '#17a2b8',
                'Fair': '#ffc107', 
                'Poor': '#dc3545'
            }
        )
        
        st.plotly_chart(fig_quality_dist, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚡ Efficiency Metrics")
        
        # Efficiency by chicken part
        chicken_parts = ['Breast', 'Thigh', 'Wing', 'Drumstick']
        efficiency_rates = [0.95, 0.87, 0.92, 0.89]
        
        fig_efficiency = px.bar(
            x=chicken_parts,
            y=efficiency_rates,
            title="Detection Efficiency by Chicken Part",
            color=efficiency_rates,
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # AI Predictive Maintenance
    st.markdown("## 🔧 AI Predictive Maintenance")
    
    maintenance_prediction = production_engine.maintenance_ai.predict_maintenance_needs({})
    
    col1, col2 = st.columns(2)
    
    with col1:
        health_score = maintenance_prediction['overall_health_score']
        urgency = maintenance_prediction['maintenance_urgency']
        
        if urgency == 'critical':
            urgency_color = 'alert-critical'
        elif urgency == 'high':
            urgency_color = 'alert-high'
        elif urgency == 'medium':
            urgency_color = 'alert-medium'
        else:
            urgency_color = 'production-metric'
        
        st.markdown(f"""
        <div class="{urgency_color}">
            <h4>System Health: {health_score:.1f}%</h4>
            <p>Maintenance Urgency: {urgency.title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Predicted issues
        st.markdown("### 🔍 Predicted Issues")
        for issue in maintenance_prediction['predicted_issues']:
            st.write(f"• {issue}")
    
    with col2:
        # System health metrics
        health_metrics = maintenance_prediction['health_metrics']
        
        metrics_df = pd.DataFrame({
            'Metric': list(health_metrics.keys()),
            'Value': list(health_metrics.values())
        })
        
        fig_health = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title="System Health Metrics",
            color='Value',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_health, use_container_width=True)
    
    # Maintenance Recommendations
    st.markdown("### 🛠️ Maintenance Recommendations")
    for rec in maintenance_prediction['recommended_actions']:
        st.markdown(f"""
        <div class="maintenance-card">
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Controls
    st.markdown("## 🎛️ Production System Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Restart Production", type="primary"):
            st.success("✅ Production system restarted")
    
    with col2:
        if st.button("⏸️ Pause Production"):
            st.warning("⏸️ Production paused")
    
    with col3:
        if st.button("🧹 Clear Alerts"):
            st.info("🧹 Alerts cleared")
    
    with col4:
        if st.button("📊 Export Data"):
            st.success("📊 Data export initiated")

if __name__ == "__main__":
    main()