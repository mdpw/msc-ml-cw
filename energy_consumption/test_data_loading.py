"""
Quick test script to verify data loading works correctly.
Run this first before the full pipeline.
"""

import sys
import pandas as pd
import yaml

# Add src to path
sys.path.append('src')

def test_data_loading():
    print("🧪 Testing Data Loading...")
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
        
        # Test basic data loading
        print("\n📊 Loading energy dataset...")
        energy_df = pd.read_csv(config['data']['raw_energy_path'])
        print(f"   Shape: {energy_df.shape}")
        print(f"   Columns: {list(energy_df.columns)[:5]}...")
        
        print("\n🌤️ Loading weather dataset...")
        weather_df = pd.read_csv(config['data']['raw_weather_path'])
        print(f"   Shape: {weather_df.shape}")
        print(f"   Cities: {weather_df['city_name'].unique()}")
        
        # Test target column exists
        target_col = config['target']['column']
        if target_col in energy_df.columns:
            print(f"✅ Target column '{target_col}' found")
            print(f"   Sample values: {energy_df[target_col].dropna().head(3).values}")
        else:
            print(f"❌ Target column '{target_col}' not found!")
            print(f"   Available columns: {list(energy_df.columns)}")
            return False
        
        print("\n✅ Basic data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🚀 Ready to run full pipeline!")
    else:
        print("\n🔧 Fix data issues before proceeding.")