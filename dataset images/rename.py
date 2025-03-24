import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Fix Unicode error

# Define the root folder
root_folder = r"C:\Users\003sa\OneDrive\Desktop\dataset images"

# List of actual subfolders
subfolders = ["Console", "Fog light", "Air intake", "Tail light", 
              "Gear stick", "Dashboard", "Steering wheel", "Headlight"]

for folder in subfolders:
    folder_path = os.path.join(root_folder, folder)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    # Get all images in sorted order
    images = sorted(os.listdir(folder_path))  

    # Debug print to check if images are found
    print(f"📂 {folder}: Found {len(images)} images")

    # Start numbering from 1 for each folder
    img_counter = 1

    for image in images:
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, f"img{img_counter}.jpg")  # Rename as img1.jpg, img2.jpg, etc.

        # Check if the file is an actual image
        if not os.path.isfile(old_path):
            print(f"⚠️ Skipping (not a file): {old_path}")
            continue

        # Rename the file
        try:
            if old_path != new_path:  # Avoid renaming if the name is already correct
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {old_path} → {new_path}")
            img_counter += 1  
        except Exception as e:
            print(f"❌ Error renaming {old_path}: {e}")

print("🎉 Renaming process completed!")
