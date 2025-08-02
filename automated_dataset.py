#!/usr/bin/env python3
"""
Automated Dataset Processor for Chicken Parts
Capstone 2025 - Hendrick
Author: Hendrick
Date: July 2025

Features:
- Batch process images from drop folders
- Auto-clear processed images
- Bulk labeling system
- Watch folder automation
"""

import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import time
import glob

class AutomatedDatasetProcessor:
    """Automated batch processing for chicken part images"""
    
    def __init__(self):
        # Back in main folder, so use direct paths
        self.base_path = Path("dataset")
        self.chicken_parts = ["breast", "thigh", "wing", "drumstick"]
        self.setup_automated_folders()
        
    def setup_automated_folders(self):
        """Create folder structure including drop zones"""
        
        folders_to_create = [
            # Drop zones for batch processing  
            "dataset/drop_zone/unsorted",          # Drop all images here
            "dataset/drop_zone/breast_batch",      # Drop breast images here
            "dataset/drop_zone/thigh_batch",       # Drop thigh images here
            "dataset/drop_zone/wing_batch",        # Drop wing images here
            "dataset/drop_zone/drumstick_batch",   # Drop drumstick images here
            
            # Organized storage
            "dataset/images/breast",
            "dataset/images/thigh", 
            "dataset/images/wing",
            "dataset/images/drumstick",
            
            # Processing folders
            "dataset/processed",
            "dataset/backup",
            "dataset/rejected",
            "dataset/temp",
            
            # Model storage - NEW
            "models",                               # YOLO models go here
            "models/training",                      # Training checkpoints
            "models/exports"                        # Exported models
        ]
        
        for folder in folders_to_create:
            Path(folder).mkdir(parents=True, exist_ok=True)
    
    def scan_drop_zones(self):
        """Scan all drop zone folders for new images"""
        found_images = {}
        
        # Check each batch folder
        batch_folders = {
            "breast": "dataset/drop_zone/breast_batch",
            "thigh": "dataset/drop_zone/thigh_batch", 
            "wing": "dataset/drop_zone/wing_batch",
            "drumstick": "dataset/drop_zone/drumstick_batch"
        }
        
        for part, folder_path in batch_folders.items():
            images = []
            folder = Path(folder_path)
            
            if folder.exists():
                # Find all image files
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images.extend(list(folder.glob(ext)))
                    images.extend(list(folder.glob(ext.upper())))
            
            found_images[part] = images
        
        # Check unsorted folder
        unsorted_folder = Path("dataset/drop_zone/unsorted")
        unsorted_images = []
        if unsorted_folder.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                unsorted_images.extend(list(unsorted_folder.glob(ext)))
                unsorted_images.extend(list(unsorted_folder.glob(ext.upper())))
        
        found_images["unsorted"] = unsorted_images
        
        return found_images
    
    def process_batch_images(self, images_list, chicken_part, auto_clear=True):
        """Process a batch of images for a specific chicken part"""
        processed_count = 0
        failed_count = 0
        results = []
        
        for image_path in images_list:
            try:
                # Generate new filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                original_name = image_path.name
                file_extension = image_path.suffix.lower()
                
                new_filename = f"{chicken_part}_{timestamp}_{processed_count + 1:03d}{file_extension}"
                destination = Path(f"dataset/images/{chicken_part}/{new_filename}")
                
                # Process image
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    max_size = (1024, 1024)
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Save processed image
                    img.save(destination, 'JPEG', quality=95)
                
                # Create metadata
                metadata = {
                    "original_filename": original_name,
                    "original_path": str(image_path),
                    "chicken_part": chicken_part,
                    "processed_timestamp": timestamp,
                    "image_size": img.size,
                    "processing_method": "batch_automated"
                }
                
                # Save metadata
                metadata_path = destination.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results.append({
                    "status": "success",
                    "original": original_name,
                    "new_name": new_filename,
                    "chicken_part": chicken_part
                })
                
                # Move original to processed folder (backup)
                if auto_clear:
                    backup_path = Path(f"dataset/backup/{chicken_part}_{timestamp}_{original_name}")
                    backup_path.parent.mkdir(exist_ok=True)
                    shutil.move(str(image_path), str(backup_path))
                
                processed_count += 1
                
            except Exception as e:
                results.append({
                    "status": "failed",
                    "original": image_path.name,
                    "error": str(e),
                    "chicken_part": chicken_part
                })
                failed_count += 1
        
        return processed_count, failed_count, results
    
    def get_dataset_stats(self):
        """Get current dataset statistics"""
        stats = {}
        
        for part in self.chicken_parts:
            folder_path = Path(f"dataset/images/{part}")
            if folder_path.exists():
                # Count image files
                jpg_count = len(list(folder_path.glob("*.jpg")))
                png_count = len(list(folder_path.glob("*.png")))
                jpeg_count = len(list(folder_path.glob("*.jpeg")))
                
                stats[part] = jpg_count + png_count + jpeg_count
            else:
                stats[part] = 0
                
        return stats
    
    def clear_drop_zones(self):
        """Clear all drop zone folders"""
        drop_folders = [
            "dataset/drop_zone/unsorted",
            "dataset/drop_zone/breast_batch",
            "dataset/drop_zone/thigh_batch", 
            "dataset/drop_zone/wing_batch",
            "dataset/drop_zone/drumstick_batch"
        ]
        
        cleared_count = 0
        for folder in drop_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                for file_path in folder_path.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        cleared_count += 1
        
        return cleared_count

def main():
    """Main Streamlit app for automated dataset processing"""
    
    st.set_page_config(
        page_title="Automated Dataset Processor",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize processor
    processor = AutomatedDatasetProcessor()
    
    # Header
    st.title("🤖 Automated Dataset Processor")
    st.markdown("### Batch process chicken part images - Drop, Label, Done!")
    
    # Instructions section
    with st.expander("📋 **How to Use This System**", expanded=True):
        st.markdown("""
        **🚀 Quick Start:**
        1. **Download 50+ chicken images per type** from Google Images  
        2. **Drop them into the appropriate folder** (see paths below)
        3. **Click "Process All Batches"** 
        4. **Done!** Images are automatically organized and labeled
        
        **📁 Drop Zone Folders:**
        ```
        📂 C:\\Users\\hendr\\Desktop\\Chicken\\dataset\\drop_zone\\
        ├── 🥩 breast_batch\\     ← Drop 50+ breast images here
        ├── 🍗 thigh_batch\\      ← Drop 50+ thigh images here  
        ├── 🍖 wing_batch\\       ← Drop 50+ wing images here
        ├── 🦴 drumstick_batch\\  ← Drop 50+ drumstick images here
        └── ❓ unsorted\\         ← Drop mixed images here (manual sorting)
        ```
        
        **💡 Pro Tips:**
        - **Target: 50+ images minimum per chicken part**
        - **Better: 100+ images per part for good training**
        - **Best: 200+ images per part for professional results**
        - Download using Google Images batch download extensions
        - System auto-resizes and renames all images
        - Mix of raw and cooked chicken for variety
        """)
    
    # Current dataset stats
    st.markdown("## 📊 Current Dataset Status")
    stats = processor.get_dataset_stats()
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
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
    
    # Progress tracking
    target_per_class = 50  # Updated minimum target
    progress = min(total / (target_per_class * 4), 1.0)
    st.progress(progress)
    st.write(f"**Progress:** {total}/{target_per_class * 4} images (Minimum for training)")
    
    # Enhanced progress indicators
    if total < 100:
        st.error("🚨 **Need more images!** Minimum 200 total (50 per class)")
    elif total < 200:
        st.warning("⚠️ **Getting started!** Goal: 400 total (100 per class)")  
    elif total < 400:
        st.info("📈 **Good progress!** Target: 800+ total (200+ per class)")
    else:
        st.success("🎉 **Excellent dataset!** Ready for professional training!")
    
    st.divider()
    
    # Scan drop zones
    st.markdown("## 🔍 Drop Zone Scanner")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        if st.button("🔄 Scan Drop Zones", type="primary"):
            with st.spinner("Scanning drop zone folders..."):
                found_images = processor.scan_drop_zones()
            
            st.session_state.found_images = found_images
        
        # Display scan results
        if hasattr(st.session_state, 'found_images'):
            found_images = st.session_state.found_images
            
            total_found = sum(len(images) for images in found_images.values())
            
            if total_found > 0:
                st.success(f"✅ Found {total_found} images ready for processing!")
                
                # Show breakdown
                for part, images in found_images.items():
                    if images:
                        st.write(f"📁 **{part.title()}:** {len(images)} images")
                        
                        # Show first few filenames
                        if len(images) <= 5:
                            for img in images:
                                st.write(f"   • {img.name}")
                        else:
                            for img in images[:3]:
                                st.write(f"   • {img.name}")
                            st.write(f"   • ... and {len(images)-3} more")
                
            else:
                st.info("📂 No images found in drop zones. Drop some images and scan again!")
    
    with col_right:
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("📁 Open Drop Zone Folder"):
            import subprocess
            import platform
            
            drop_path = str(Path("dataset/drop_zone").absolute())
            if platform.system() == "Windows":
                subprocess.run(f'explorer "{drop_path}"', shell=True)
            st.success("Drop zone folder opened!")
        
        if st.button("🧹 Clear All Drop Zones"):
            cleared = processor.clear_drop_zones()
            st.success(f"Cleared {cleared} files from drop zones")
            if hasattr(st.session_state, 'found_images'):
                del st.session_state.found_images
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()
    
    st.divider()
    
    # Batch processing
    st.markdown("## 🚀 Batch Processing")
    
    if hasattr(st.session_state, 'found_images'):
        found_images = st.session_state.found_images
        
        # Process all batches
        if st.button("🤖 Process All Batches", type="primary"):
            total_processed = 0
            total_failed = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_parts = [part for part in ["breast", "thigh", "wing", "drumstick"] 
                          if found_images.get(part, [])]
            
            for idx, part in enumerate(batch_parts):
                images = found_images[part]
                if images:
                    status_text.text(f"Processing {len(images)} {part} images...")
                    
                    processed, failed, results = processor.process_batch_images(
                        images, part, auto_clear=True
                    )
                    
                    total_processed += processed
                    total_failed += failed
                    
                    # Show results for this batch
                    if processed > 0:
                        st.success(f"✅ **{part.title()}:** {processed} images processed")
                    if failed > 0:
                        st.error(f"❌ **{part.title()}:** {failed} images failed")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(batch_parts))
            
            # Final summary
            status_text.text("Batch processing complete!")
            st.balloons()
            st.success(f"🎉 **Complete!** Processed {total_processed} images, {total_failed} failed")
            
            # Clear session state
            del st.session_state.found_images
            
            # Auto-refresh stats
            time.sleep(1)
            st.rerun()
    
    # Manual sorting for unsorted images
    if hasattr(st.session_state, 'found_images') and st.session_state.found_images.get('unsorted', []):
        st.markdown("## 🏷️ Manual Sorting (Unsorted Images)")
        
        unsorted_images = st.session_state.found_images['unsorted']
        st.write(f"Found {len(unsorted_images)} unsorted images")
        
        # Select images and label
        selected_part = st.selectbox(
            "Label these unsorted images as:",
            ["breast", "thigh", "wing", "drumstick"]
        )
        
        if st.button(f"🏷️ Label All {len(unsorted_images)} Images as '{selected_part.title()}'"):
            with st.spinner(f"Processing {len(unsorted_images)} images as {selected_part}..."):
                processed, failed, results = processor.process_batch_images(
                    unsorted_images, selected_part, auto_clear=True
                )
            
            st.success(f"✅ Processed {processed} images as {selected_part}")
            if failed > 0:
                st.error(f"❌ {failed} images failed")
            
            # Clear unsorted from session state
            st.session_state.found_images['unsorted'] = []
            st.rerun()

if __name__ == "__main__":
    main()