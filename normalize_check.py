import cv2
import os

# Folder containing resized images
folder = r"D:\FFPP\resized\real"

# List images in folder
files = os.listdir(folder)
print("Files found:", files[:5])

# Select first image
img_path = os.path.join(folder, files[0])
print("Using image:", img_path)

# Read image
img = cv2.imread(img_path)
print("Image loaded:", img is not None)

# Normalize pixel values (0–1)
img = img / 255.0

# Print min and max pixel values
print("Min:", img.min(), "Max:", img.max())
