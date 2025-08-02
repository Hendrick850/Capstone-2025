#!/usr/bin/env python3
"""
Dataset Cleanup Tool
Capstone 2025 - Hendrick
Author: Hendrick
Date: July 2025

Removes all dataset files and folders for fresh start
"""

import os
import shutil
from pathlib import Path
import streamlit as st
import time

class DatasetCleaner:
    """Clean up all dataset files and folders"""
    
    def __init__(self):
        self.base_path = Path("dataset")
        self.folders_to_remove = [
            "dataset",           # Main dataset folder
            "logs",             # Log files
            "results",          # Result files
            "captured_images",  # Captured images
            "models"            # Model files (optional)
        ]
        
    def scan_existing_files(self):
        """Scan what files/folders currently exist"""
        existing = {}
        total_files = 0
        total_size_mb = 0
        
        for folder_name in self.folders_to_remove:
            folder_path = Path(folder_name)
            
            if folder_path.exists():
                # Count files and calculate size
                file_count = 0
                folder_size = 0
                
                try:
                    for item in folder_path.rglob("*"):
                        if item.is_file():
                            file_count += 1
                            folder_size += item.stat().st_size
                    
                    existing[folder_name] = {
                        "exists": True,
                        "file_count": file_count,
                        "size_mb": round(folder_size / (1024 * 1024), 2)
                    }
                    
                    total_files += file_count
                    total_size_mb += existing[folder_name]["size_mb"]
                    
                except PermissionError:
                    existing[folder_name] = {
                        "exists": True,
                        "file_count": "Permission Error",
                        "size_mb": 0
                    }
            else:
                existing[folder_name] = {
                    "exists": False,
                    "file_count": 0,
                    "size_mb": 0
                }
        
        return existing, total_files, total_size_mb
    
    def remove_folder_safely(self, folder_path):
        """Safely remove a folder with error handling"""
        try:
            if folder_path.exists():
                if folder_path.is_dir():
                    shutil.rmtree(folder_path)
                    return True, f"Removed folder: {folder_path}"
                else:
                    folder_path.unlink()
                    return True, f"Removed file: {folder_path}"
            else:
                return True, f"Already gone: {folder_path}"
        except PermissionError:
            return False, f"Permission denied: {folder_path}"
        except Exception as e:
            return False, f"Error removing {folder_path}: {str(e)}"
    
    def clean_all_dataset_files(self):
        """Remove all dataset-related files and folders"""
        results = []
        success_count = 0
        error_count = 0
        
        for folder_name in self.folders_to_remove:
            folder_path = Path(folder_name)
            success, message = self.remove_folder_safely(folder_path)
            
            results.append({
                "folder": folder_name,
                "success": success,
                "message": message
            })
            
            if success:
                success_count += 1
            else:
                error_count += 1
        
        return results, success_count, error_count

def main():
    """Main cleanup interface"""
    
    st.set_page_config(
        page_title="Dataset Cleaner",
        page_icon="🧹",
        layout="wide"
    )
    
    # Initialize cleaner
    cleaner = DatasetCleaner()
    
    # Header with warning
    st.title("🧹 Dataset Cleanup Tool")
    st.markdown("### Remove all dataset files for fresh start")
    
    # Warning section
    st.error("⚠️ **WARNING: This will permanently delete all dataset files!**")
    
    with st.expander("🚨 **What will be deleted:**", expanded=True):
        st.markdown("""
        **This tool will remove:**
        - 📂 `dataset/` folder (all organized images)
        - 📂 `logs/` folder (detection logs)
        - 📂 `results/` folder (saved results)
        - 📂 `captured_images/` folder (screenshot images)
        - 📂 `models/` folder (optional - trained models)
        
        **This tool will NOT touch:**
        - ✅ Your Python code files (.py files)
        - ✅ Your main project folder
        - ✅ Any files outside the Chicken folder
        - ✅ Your GitHub repository
        """)
    
    # Scan current state
    st.markdown("## 🔍 Current Dataset Status")
    
    if st.button("🔄 Scan Current Files"):
        with st.spinner("Scanning dataset files..."):
            existing_files, total_files, total_size_mb = cleaner.scan_existing_files()
        
        st.session_state.existing_files = existing_files
        st.session_state.total_files = total_files
        st.session_state.total_size_mb = total_size_mb
    
    # Display scan results
    if hasattr(st.session_state, 'existing_files'):
        existing_files = st.session_state.existing_files
        total_files = st.session_state.total_files
        total_size_mb = st.session_state.total_size_mb
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Files", total_files)
        with col2:
            st.metric("💾 Total Size", f"{total_size_mb:.1f} MB")
        with col3:
            folders_exist = sum(1 for f in existing_files.values() if f["exists"])
            st.metric("📁 Folders Found", folders_exist)
        
        # Detailed breakdown
        st.markdown("### 📊 Folder Details")
        
        for folder_name, info in existing_files.items():
            if info["exists"]:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"📂 **{folder_name}/**")
                with col2:
                    st.write(f"Files: {info['file_count']}")
                with col3:
                    st.write(f"Size: {info['size_mb']} MB")
                with col4:
                    st.write("🔴 Will be deleted")
            else:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"📂 ~~{folder_name}/~~")
                with col4:
                    st.write("✅ Already clean")
        
        # Show what will happen
        if total_files > 0:
            st.warning(f"⚠️ **Ready to delete {total_files} files ({total_size_mb:.1f} MB)**")
        else:
            st.success("✅ **No dataset files found - already clean!**")
    
    st.divider()
    
    # Cleanup section
    st.markdown("## 🗑️ Cleanup Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Safety confirmation
        st.markdown("### ✅ Safety Confirmation")
        
        confirm1 = st.checkbox("☑️ I want to delete all dataset files")
        confirm2 = st.checkbox("☑️ I understand this cannot be undone")
        confirm3 = st.checkbox("☑️ I have backed up anything important")
        
        all_confirmed = confirm1 and confirm2 and confirm3
        
        # Cleanup button
        if st.button(
            "🧹 DELETE ALL DATASET FILES", 
            type="primary", 
            disabled=not all_confirmed,
            help="Must check all confirmations first"
        ):
            if all_confirmed:
                st.warning("🚨 Starting cleanup process...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Execute cleanup
                results, success_count, error_count = cleaner.clean_all_dataset_files()
                
                # Show results
                st.markdown("### 🔄 Cleanup Results")
                
                for i, result in enumerate(results):
                    progress_bar.progress((i + 1) / len(results))
                    
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")
                
                status_text.text("Cleanup complete!")
                
                # Final summary
                if error_count == 0:
                    st.balloons()
                    st.success(f"🎉 **SUCCESS!** All {success_count} folders cleaned!")
                    st.info("💡 **Ready for fresh start!** You can now run your dataset collectors with clean folders.")
                else:
                    st.warning(f"⚠️ **Partial success:** {success_count} cleaned, {error_count} errors")
                
                # Clear session state
                if hasattr(st.session_state, 'existing_files'):
                    del st.session_state.existing_files
                    del st.session_state.total_files
                    del st.session_state.total_size_mb
            
            else:
                st.error("❌ Please confirm all checkboxes first")
    
    with col2:
        # Alternative actions
        st.markdown("### 🛠️ Alternative Actions")
        
        st.info("**Less drastic options:**")
        
        if st.button("📁 Open Dataset Folder"):
            import subprocess
            import platform
            
            dataset_path = str(Path("dataset").absolute())
            try:
                if platform.system() == "Windows":
                    subprocess.run(f'explorer "{dataset_path}"', shell=True)
                    st.success("Dataset folder opened in Explorer")
                else:
                    st.info("Manual path: " + dataset_path)
            except:
                st.info("Manual path: " + dataset_path)
        
        if st.button("📋 Create Fresh Folders Only"):
            # Just create empty folder structure
            folders_to_create = [
                "dataset/drop_zone/breast_batch",
                "dataset/drop_zone/thigh_batch", 
                "dataset/drop_zone/wing_batch",
                "dataset/drop_zone/drumstick_batch",
                "dataset/drop_zone/unsorted",
                "dataset/images/breast",
                "dataset/images/thigh",
                "dataset/images/wing", 
                "dataset/images/drumstick"
            ]
            
            for folder in folders_to_create:
                Path(folder).mkdir(parents=True, exist_ok=True)
            
            st.success("✅ Fresh folder structure created!")
        
        st.markdown("---")
        st.markdown("**💡 Recommendations:**")
        st.write("• Use 'Scan Current Files' first")
        st.write("• Back up important images manually")
        st.write("• Clean start = better organization")
        st.write("• Re-run automated_dataset.py after cleanup")

if __name__ == "__main__":
    main()