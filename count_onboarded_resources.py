"""
Resource Counter for Onboarding Files
This script helps you count how many resources are onboarded in Excel files.
"""

import pandas as pd
import os
from pathlib import Path

def count_resources_in_excel(file_path):
    """Count resources in an Excel file"""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Get basic statistics
        total_resources = len(df)
        columns = df.columns.tolist()
        
        print(f"\n{'='*60}")
        print(f"📊 ONBOARDING FILE ANALYSIS: {Path(file_path).name}")
        print(f"{'='*60}")
        print(f"\n✅ Total Resources Onboarded: {total_resources}")
        print(f"\n📋 Columns in the file:")
        for i, col in enumerate(columns, 1):
            print(f"   {i}. {col}")
        
        # Display first few rows as preview
        print(f"\n📄 Preview (First 5 rows):")
        print(df.head().to_string())
        
        # Check for specific common columns
        print(f"\n📌 Data Summary:")
        for col in columns:
            non_null_count = df[col].notna().sum()
            print(f"   - {col}: {non_null_count} entries")
        
        return df, total_resources
        
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None, 0

def find_onboarding_files():
    """Find potential onboarding Excel files"""
    base_dir = Path(__file__).parent
    
    # Common locations to check
    locations = [
        base_dir / "data" / "uploaded_files",
        base_dir,
        base_dir / "uploads",
        base_dir / "onboarding"
    ]
    
    excel_files = []
    for location in locations:
        if location.exists():
            excel_files.extend(list(location.glob("*.xlsx")))
            excel_files.extend(list(location.glob("*.xls")))
    
    return excel_files

if __name__ == "__main__":
    print("\n🔍 Searching for onboarding Excel files...\n")
    
    # Find all Excel files
    excel_files = find_onboarding_files()
    
    if not excel_files:
        print("❌ No Excel files found in common locations.")
        print("\n💡 Please specify the file path manually:")
        print("   Example: python count_onboarded_resources.py")
        
        # Allow manual input
        file_path = input("\nEnter the full path to your onboarding Excel file: ").strip()
        if file_path and Path(file_path).exists():
            count_resources_in_excel(file_path)
        else:
            print("❌ Invalid file path or file does not exist.")
    else:
        print(f"✅ Found {len(excel_files)} Excel file(s):\n")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        
        # Process all files
        print("\n" + "="*60)
        for file in excel_files:
            count_resources_in_excel(file)
            print()
