import cv2
import os

input_real = r"D:\FFPP\frames\real"
input_fake = r"D:\FFPP\frames\fake"

output_real = r"D:\FFPP\resized\real"
output_fake = r"D:\FFPP\resized\fake"

os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

IMG_SIZE = 224

def resize_images(input_folder, output_folder):
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(output_folder, img_name), resized_img)

resize_images(input_real, output_real)
resize_images(input_fake, output_fake)

print("Image resizing completed successfully.")
