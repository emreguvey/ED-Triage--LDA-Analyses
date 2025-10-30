# src/data_loader.py
import pandas as pd
import numpy as np
import os
import glob
import gc
from datetime import datetime
import logging
from src.config import DATA_DIR, COLUMN_GROUP

logger = logging.getLogger(__name__)

def optimize_dataframe(df):
    """
    Optimizes memory usage of a DataFrame by downcasting numeric types
    and converting object types to 'category' where appropriate.
    """
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
            
    for col in df.select_dtypes(include=['float64']).columns:
        if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
            
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    return df

def load_excel_enhanced(filepath, group_name):
    """
    Loads a single Excel file with enhanced optimization and error handling.
    """
    start_time = datetime.now()
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        df = optimize_dataframe(df)
        
        # Add a column to identify the source file/group
        df[COLUMN_GROUP] = group_name
        
        load_time = (datetime.now() - start_time).total_seconds()
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"  ✅ Loaded '{group_name}': {len(df):,} rows in {load_time:.1f}s ({memory_mb:.1f}MB)")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        print(f"  ❌ FAILED to load {filepath}: {e}")
        return None

def load_all_data():
    """
    Finds and loads all Excel files from the DATA_DIR, then concatenates them.
    """
    print(f"🚀 ENHANCED DATA LOADING from: {DATA_DIR}")
    print("-" * 40)
    
    # Find all .xlsx files in the data directory
    search_pattern = os.path.join(DATA_DIR, "*.xlsx")
    found_files = glob.glob(search_pattern)
    
    if not found_files:
        print(f"❌ No Excel files found in {DATA_DIR}")
        print("Please add your .xlsx data files to the 'data' directory.")
        return None
        
    print(f"📁 Found {len(found_files)} file(s):")
    
    all_dfs = []
    for f in found_files:
        # Use the filename (without extension) as the group name
        group_name = os.path.splitext(os.path.basename(f))[0]
        df = load_excel_enhanced(f, group_name)
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        print("❌ All data files failed to load.")
        return None
        
    # Combine datasets
    print("🔗 Combining datasets...")
    df_combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    gc.collect()
    return df_combined