#!/usr/bin/env python3
"""
YOLO Training Dataset Preparation
Capstone 2025 - Hendrick
Convert organized chicken images to YOLO training format

Usage: streamlit run yolo_training_prep.py
"""

import streamlit as st
import os
import shutil
from pathlib import Path
import yaml
import random
from PIL import Image
import pandas as pd

class YOLODatasetPrep:
    """Prepare chicken dataset for YOLO training"""
    
    def __init__(self):
        self.dataset_path = Path("dataset/images")
        self.yolo_path = Path("yolo_dataset")
        self.chicken_classes = ["breast", "thigh", "wing", "drumstick"]
        
    def create_yolo_structure(self):
        """Create YOLO dataset folder structure"""
        folders = [
            "yolo_dataset/images/train",
            "yolo_dataset/images/val", 
            "yolo_dataset/labels/train",
            "yolo_dataset/labels/val"
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
            
        return folders
    
    def get_dataset_stats(self):
        """Get current organized dataset statistics"""
        stats = {}
        
        for chicken_class in self.chicken_classes:
            class_path = self.dataset_path / chicken_class
            if class_path.exists():
                # Count only image files (not metadata)
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend(list(class_path.glob(f"*{ext}")))
                stats[chicken_class] = len(image_files)
            else:
                stats[chicken_class] = 0
                
        return stats
    
    def create_simple_annotations(self, image_path, class_id):
        """Create simple full-image annotation for classification"""
        # For initial training, we'll use full image as bounding box
        # Later can be improved with actual bounding box annotation
        
        annotation = f"{class_id} 0.5 0.5 1.0 1.0\n"
        # Format: class_id center_x center_y width height (normalized 0-1)
        # 0.5 0.5 1.0 1.0 = center of image, full width/height
        
        return annotation
    
    def split_dataset(self, train_ratio=0.8):
        """Split dataset into train/validation sets"""
        train_files = []
        val_files = []
        
        for class_id, chicken_class in enumerate(self.chicken_classes):
            class_path = self.dataset_path / chicken_class
            
            if class_path.exists():
                # Get all image files
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend(list(class_path.glob(f"*{ext}")))
                
                # Shuffle and split
                random.shuffle(image_files)
                split_point = int(len(image_files) * train_ratio)
                
                class_train = image_files[:split_point]
                class_val = image_files[split_point:]
                
                # Add class info
                for img in class_train:
                    train_files.append((img, class_id, chicken_class))
                for img in class_val:
                    val_files.append((img, class_id, chicken_class))
        
        return train_files, val_files
    
    def copy_and_annotate(self, file_list, split_type):
        """Copy images and create annotations for train/val split"""
        
        images_dir = self.yolo_path / "images" / split_type
        labels_dir = self.yolo_path / "labels" / split_type
        
        copied_count = 0
        
        for img_path, class_id, class_name in file_list:
            try:
                # Copy image
                img_name = f"{class_name}_{copied_count:04d}.jpg"
                dest_img_path = images_dir / img_name
                
                # Convert and copy image
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dest_img_path, 'JPEG', quality=95)
                
                # Create annotation
                annotation = self.create_simple_annotations(img_path, class_id)
                label_name = img_name.replace('.jpg', '.txt')
                label_path = labels_dir / label_name
                
                with open(label_path, 'w') as f:
                    f.write(annotation)
                
                copied_count += 1
                
            except Exception as e:
                st.error(f"Error processing {img_path}: {e}")
        
        return copied_count
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        
        config = {
            'path': str(Path("yolo_dataset").absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.chicken_classes),
            'names': self.chicken_classes
        }
        
        yaml_path = self.yolo_path / "chicken_dataset.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return yaml_path
    
    def prepare_full_dataset(self):
        """Complete dataset preparation process"""
        
        # Create folder structure
        st.info("Creating YOLO dataset structure...")
        folders = self.create_yolo_structure()
        st.success(f"Created {len(folders)} folders")
        
        # Split dataset
        st.info("Splitting dataset into train/validation...")
        train_files, val_files = self.split_dataset(train_ratio=0.8)
        
        st.write(f"**Training set**: {len(train_files)} images")
        st.write(f"**Validation set**: {len(val_files)} images")
        
        # Copy and annotate training files
        st.info("Processing training set...")
        train_count = self.copy_and_annotate(train_files, "train")
        
        # Copy and annotate validation files  
        st.info("Processing validation set...")
        val_count = self.copy_and_annotate(val_files, "val")
        
        # Create config file
        st.info("Creating YOLO configuration...")
        yaml_path = self.create_yaml_config()
        
        return train_count, val_count, yaml_path

def main():
    """Main Streamlit interface for YOLO training preparation"""
    
    st.set_page_config(
        page_title="YOLO Training Prep",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 YOLO Training Dataset Preparation")
    st.markdown("### Convert organized chicken images to YOLO training format")
    
    # Initialize prep system
    prep = YOLODatasetPrep()
    
    # Show current dataset stats
    st.markdown("## 📊 Current Dataset Status")
    stats = prep.get_dataset_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🥩 Breast", stats["breast"])
    with col2:
        st.metric("🍗 Thigh", stats["thigh"])
    with col3:
        st.metric("🍖 Wing", stats["wing"])
    with col4:
        st.metric("🦴 Drumstick", stats["drumstick"])
    with col5:
        total = sum(stats.values())
        st.metric("📊 Total", total)
    
    # Training readiness check
    min_per_class = 30
    ready_for_training = all(count >= min_per_class for count in stats.values())
    
    if ready_for_training:
        st.success(f"✅ Dataset ready for training! ({total} total images)")
    else:
        st.warning(f"⚠️ Need at least {min_per_class} images per class for good training")
    
    st.divider()
    
    # Dataset preparation
    st.markdown("## 🔄 Prepare YOLO Training Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **What this does:**
        1. **Creates YOLO folder structure** → Proper train/val split
        2. **Converts images** → Standardized format and naming
        3. **Generates annotations** → YOLO format labels
        4. **Creates config file** → Ready for training
        
        **Dataset Split:**
        - **80% Training** → Used to teach the model
        - **20% Validation** → Used to test accuracy
        
        **Output Structure:**
        ```
        yolo_dataset/
        ├── images/
        │   ├── train/ ← 80% of images
        │   └── val/   ← 20% of images
        ├── labels/
        │   ├── train/ ← Training annotations
        │   └── val/   ← Validation annotations
        └── chicken_dataset.yaml ← Config file
        ```
        """)
    
    with col2:
        st.markdown("### ⚙️ Preparation Settings")
        
        train_split = st.slider(
            "Training Split %", 
            min_value=70, 
            max_value=90, 
            value=80,
            help="Percentage of images for training vs validation"
        )
        
        if st.button("🚀 Prepare Dataset for Training", type="primary"):
            if ready_for_training:
                with st.spinner("Preparing YOLO dataset..."):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        train_count, val_count, yaml_path = prep.prepare_full_dataset()
                        
                        progress_bar.progress(100)
                        status_text.text("Dataset preparation complete!")
                        
                        st.balloons()
                        st.success(f"""
                        ✅ **Dataset Ready for Training!**
                        
                        **Training Images**: {train_count}
                        **Validation Images**: {val_count}
                        **Config File**: {yaml_path}
                        
                        **Next Step**: Run YOLO training with this prepared dataset
                        """)
                        
                        # Show next steps
                        st.markdown("### 🎯 Next: Start Training")
                        st.code("""
# Training command (run in terminal):
yolo train data=yolo_dataset/chicken_dataset.yaml model=yolov5s.pt epochs=100 imgsz=640
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during preparation: {e}")
                        
            else:
                st.error("Need more images before training! Collect at least 30 per class.")
    
    # Show existing YOLO dataset if available
    if Path("yolo_dataset").exists():
        st.divider()
        st.markdown("## 📁 Existing YOLO Dataset")
        
        yolo_stats = {
            "train_images": len(list(Path("yolo_dataset/images/train").glob("*.jpg"))),
            "val_images": len(list(Path("yolo_dataset/images/val").glob("*.jpg"))),
            "train_labels": len(list(Path("yolo_dataset/labels/train").glob("*.txt"))),
            "val_labels": len(list(Path("yolo_dataset/labels/val").glob("*.txt")))
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Images", yolo_stats["train_images"])
        with col2:
            st.metric("Val Images", yolo_stats["val_images"])
        with col3:
            st.metric("Train Labels", yolo_stats["train_labels"])
        with col4:
            st.metric("Val Labels", yolo_stats["val_labels"])
        
        if st.button("🗑️ Clear Existing Dataset"):
            shutil.rmtree("yolo_dataset")
            st.success("Existing dataset cleared!")
            st.rerun()

if __name__ == "__main__":
    main()