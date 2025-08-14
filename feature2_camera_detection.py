# 🎥 AUTOPACK AI - Feature 2 Camera Detection
# Webcam-based real-time chicken part detection

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from pathlib import Path

class CameraDetector:
    """Main camera detection class for Feature 2"""
    
    def __init__(self):
        self.model = None
        self.camera = None
        self.is_active = False
        
    def load_model(self, model_path='models/chicken_best.pt'):
        """Load YOLO model"""
        try:
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                print(f"✅ Model loaded: {model_path}")
                return True
            else:
                print(f"❌ Model not found: {model_path}")
                return False
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
    
    def start_camera(self):
        """Start webcam"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                print("✅ Camera started")
                return True
            else:
                print("❌ Camera failed")
                return False
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def detect_frame(self, frame):
        """Run detection on single frame"""
        if self.model is None:
            return frame, []
        
        try:
            results = self.model(frame, conf=0.5, verbose=False)
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Draw detection
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"Detection: {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1-10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detections.append({
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return frame, detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []

if __name__ == "__main__":
    print("🎥 AUTOPACK AI Feature 2 - Camera Detection Test")
    
    detector = CameraDetector()
    
    if detector.load_model():
        if detector.start_camera():
            print("✅ Press 'q' to quit, 's' to save frame")
            
            while True:
                ret, frame = detector.camera.read()
                if ret:
                    annotated_frame, detections = detector.detect_frame(frame)
                    
                    cv2.imshow('AUTOPACK AI - Feature 2', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite('detection_result.jpg', annotated_frame)
                        print("📸 Frame saved")
            
            detector.camera.release()
            cv2.destroyAllWindows()
