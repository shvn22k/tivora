"""
Script to match images with their descriptions from CSV
"""

import os
import csv
import json
from pathlib import Path

# Paths
IMG_DIR = r"C:\Projects\tivora\recc-mvp\img-data"
CSV_FILE = r"C:\Projects\tivora\recc-mvp\desc-data\data.csv"
OUTPUT_FILE = r"C:\Projects\tivora\recc-mvp\experiments\matched_data.json"

def main():
    print("=" * 60)
    print("Image Data Matcher")
    print("=" * 60)
    
    # Get all image files
    print("\n📁 Reading images from:", IMG_DIR)
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Found {len(image_files)} images")
    
    # Create a set of image names for quick lookup
    image_names = set(image_files)
    
    # Read CSV and match with images
    print("\n📄 Reading CSV data from:", CSV_FILE)
    matched_data = []
    matched_count = 0
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            image_name = row['image']
            
            # Check if this image exists in our image folder
            if image_name in image_names:
                matched_data.append({
                    'image': image_name,
                    'description': row['description'],
                    'display_name': row['display name'],
                    'category': row['category']
                })
                matched_count += 1
    
    print(f"   ✅ Matched {matched_count} images with their data")
    
    # Write to JSON file
    print("\n💾 Writing to JSON:", OUTPUT_FILE)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved {len(matched_data)} items to JSON")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images in folder: {len(image_files)}")
    print(f"Matched with CSV data:  {matched_count}")
    print(f"Output file:            {OUTPUT_FILE}")
    
    # Show sample data
    if matched_data:
        print("\n📋 Sample (first 3 items):")
        for i, item in enumerate(matched_data[:3], 1):
            print(f"\n{i}. {item['image']}")
            print(f"   Display Name: {item['display_name']}")
            print(f"   Category: {item['category']}")
            print(f"   Description: {item['description'][:100]}...")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()

