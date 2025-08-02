# 🐔 Chicken Parts Identifier Using Computer Vision

**Capstone Project 2025 - Autopack**  
*AI-powered chicken part detection system using YOLO and OpenCV*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)](https://streamlit.io/)
[![YOLOv5](https://img.shields.io/badge/yolo-v5-orange.svg)](https://github.com/ultralytics/yolov5)

## 📋 Project Overview

This capstone project develops an AI-powered computer vision system that automatically identifies different chicken parts (breast, thigh, wing, drumstick) using a top-down camera setup. The system provides real-time detection with high accuracy for food processing applications.

### 🎯 Project Goals
- **Real-time Detection**: Process live camera feed for chicken part identification
- **High Accuracy**: Achieve 85%+ detection accuracy across all chicken part classes
- **Industrial Application**: Solve real problems for Autopack food processing
- **User-Friendly Interface**: Intuitive web-based interface for monitoring
- **Scalable Solution**: Modular design for deployment in production environments

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Top-down      │    │   AI Processing  │    │   Web Interface │
│   RGB Camera    │───▶│   YOLO Model     │───▶│   (Streamlit)   │
│   (HD/4K)       │    │   + OpenCV       │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Data Storage   │
                       │   Results & Logs │
                       └──────────────────┘
```

## ⭐ Current Status

### **Week 4 Progress:**
- ✅ **Dataset Collection**: 200+ images collected (50+ per chicken part)
- ✅ **Automated Processing**: Batch image organization system
- ✅ **Basic Detection**: General YOLO model integration
- ✅ **Web Interface**: Streamlit-based user interface
- 🔄 **Next**: Custom model training on chicken-specific dataset

### **Upcoming Milestones:**
- **Week 5**: Custom YOLO model training
- **Week 6**: Camera integration & real-time detection
- **Week 7**: Performance optimization & testing
- **Week 8**: Final presentation & deployment

## 🚀 Quick Start

### **Prerequisites:**
- Python 3.8 or higher
- Webcam or USB camera (for live detection)
- Git

### **Installation:**
```bash
# Clone repository
git clone https://github.com/Hendrick850/Capstone-2025.git
cd Capstone-2025

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install streamlit opencv-python torch ultralytics pandas numpy matplotlib pillow seaborn
```

### **Run the Application:**
```bash
# Dataset collection and organization
streamlit run automated_dataset.py

# Photo testing (current week)
streamlit run "Python Files/Test.py"

# Full application (future weeks)
streamlit run "Python Files/main.py"
```

## 📁 Project Structure

```
Capstone-2025/
├── README.md                           # Project documentation
├── automated_dataset.py                # Main dataset collection tool
├── 📁 Python Files/                    # Development code
│   ├── Test.py                        # Photo testing interface
│   ├── main.py                        # Production application
│   ├── dataset_collector.py           # Manual dataset tools
│   └── cleanup_dataset.py             # Maintenance utilities
├── 📁 dataset/                        # Training data
│   ├── 📁 drop_zone/                  # Batch processing folders
│   │   ├── 📁 breast_batch/           # Drop breast images here
│   │   ├── 📁 thigh_batch/            # Drop thigh images here
│   │   ├── 📁 wing_batch/             # Drop wing images here
│   │   └── 📁 drumstick_batch/        # Drop drumstick images here
│   ├── 📁 images/                     # Organized training images
│   │   ├── 📁 breast/                 # Processed breast images
│   │   ├── 📁 thigh/                  # Processed thigh images
│   │   ├── 📁 wing/                   # Processed wing images
│   │   └── 📁 drumstick/              # Processed drumstick images
│   └── 📁 backup/                     # Safety backups
├── 📁 models/                         # AI models storage
│   ├── 📁 training/                   # Training checkpoints
│   └── 📁 exports/                    # Final trained models
└── 📁 logs/                           # Application logs
```

## 🎯 How to Use

### **1. Dataset Collection (Week 4)**
```bash
# Run the automated dataset organizer
streamlit run automated_dataset.py

# 1. Download chicken part images from Google Images
# 2. Drop them into appropriate batch folders:
#    - dataset/drop_zone/breast_batch/
#    - dataset/drop_zone/thigh_batch/
#    - dataset/drop_zone/wing_batch/
#    - dataset/drop_zone/drumstick_batch/
# 3. Click "Process All Batches" in the web interface
# 4. Images are automatically organized and renamed
```

### **2. Photo Testing (Current)**
```bash
# Test detection with uploaded photos
streamlit run "Python Files/Test.py"

# Upload images and see what the AI detects
# Adjust confidence thresholds
# Document results for training
```

### **3. Future: Live Camera Detection**
```bash
# Will be available after custom training (Week 6+)
streamlit run "Python Files/main.py"
```

## 🔧 Technical Specifications

### **Hardware Requirements:**
- **Minimum**: Intel i5, 8GB RAM, Integrated Graphics
- **Recommended**: Intel i7, 16GB RAM, NVIDIA GTX 1060+
- **Camera**: USB webcam or built-in camera (HD recommended)

### **Software Stack:**
- **AI Framework**: YOLOv5/v8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly

### **Performance Targets:**
- **Detection Accuracy**: 85%+ across all chicken part classes
- **Processing Speed**: 15+ FPS for real-time detection
- **Response Time**: <100ms per detection
- **Uptime**: 24/7 continuous operation capability

## 📊 Dataset Information

### **Current Dataset Status:**
- **Breast**: 50+ images
- **Thigh**: 50+ images  
- **Wing**: 50+ images
- **Drumstick**: 50+ images
- **Total**: 200+ images (Week 4 milestone achieved)

### **Data Collection Sources:**
- Google Images (raw chicken parts)
- Food photography websites
- Self-captured images
- Various lighting conditions and angles

## 🧪 Testing & Validation

### **Current Testing:**
- Photo upload testing with general YOLO model
- Interface usability testing
- Dataset organization validation
- Performance benchmarking

### **Future Testing:**
- Custom model accuracy testing
- Real-time camera performance
- Production environment simulation
- User acceptance testing

## 🤝 Team & Collaboration

### **Team Members:**
- **Hendrick** - Lead Developer & AI Implementation
- **Teck Yang** - Storyboard Design & Visual Documentation

### **Collaboration Workflow:**
- **Main Branch**: Stable, working code (Hendrick)
- **Feature Branches**: Development and experimentation
- **Pull Requests**: Code review and integration
- **Issues**: Bug tracking and feature requests

## 📈 Development Roadmap

### **✅ Completed (Week 1-4):**
- [x] Project planning and requirements analysis
- [x] Basic detection framework development
- [x] Dataset collection system implementation
- [x] Web interface creation
- [x] Initial testing and validation

### **🔄 In Progress (Week 4-5):**
- [ ] Custom YOLO model training
- [ ] Dataset annotation and preparation
- [ ] Model performance optimization

### **📋 Upcoming (Week 5-8):**
- [ ] Real-time camera integration
- [ ] Production deployment preparation
- [ ] Final testing and validation
- [ ] Capstone presentation preparation

## 📄 Documentation

### **Additional Resources:**
- [Installation Guide](docs/installation.md) *(coming soon)*
- [API Documentation](docs/api.md) *(coming soon)*
- [Training Guide](docs/training.md) *(coming soon)*
- [Deployment Guide](docs/deployment.md) *(coming soon)*

## 🙏 Acknowledgments

- **Autopack** - Industry partner and project sponsor
- **Ultralytics** - YOLOv5 framework
- **OpenCV Community** - Computer vision libraries
- **Streamlit** - Web interface framework

## 📞 Contact

- **Project Lead**: Hendrick - [GitHub](https://github.com/Hendrick850)
- **Project Repository**: [Capstone-2025](https://github.com/Hendrick850/Capstone-2025)

---

**Project Status**: 🚧 In Development - Week 4  
**Last Updated**: August 2025  
**Version**: 1.0.0-beta