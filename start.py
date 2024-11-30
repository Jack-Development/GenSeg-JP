import datetime
import os
import random
import sys
import easyocr
import json
import jaconv
import cv2
import numpy as np
from numpyencoder import NumpyEncoder
from PIL import Image
from OHTR import get_average_height, average_SWT

def generate_job_number():
    prefix = "JOB"
    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    job_number = f"{prefix}-{date_str}-{random_number}"
    return job_number


def find_borders(img, threshold):
    h, w = img.shape
    # Initialize the borders
    left = 0
    right = w - 1
    top = 0
    bottom = h - 1

    # Compute mean values along axes
    mean_cols = np.mean(img, axis=0)
    mean_rows = np.mean(img, axis=1)

    # Find the borders
    left = np.argmax(mean_cols > threshold)
    right = right - np.argmax(mean_cols[::-1] > threshold)
    top = np.argmax(mean_rows > threshold)
    bottom = bottom - np.argmax(mean_rows[::-1] > threshold)

    return (top, bottom, left, right)


def OCR_easy(input_file, job_number):
    # EasyOCR - Apache License 2.0
    # OpenCV2 - Apache License 2.0
    directories = ['full', 'raw', 'clean']
    base_path = f"Archive/{job_number}"

    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)

    reader = easyocr.Reader(['ja','en'])
    image = Image.open(input_file)
    result = reader.readtext(image)

    mask = np.zeros(np.array(image).shape[:2], dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)
    sml_kernel = np.ones((3,3),np.uint8)
    for i, text in enumerate(result):
        bbox = text[0]
        word = text[1]
        word = jaconv.normalize(word)
        word = word.replace("-", "一")
        print(word)
        conf = text[2]

        # Get bounding box
        left = min(point[0] for point in bbox)
        upper = min(point[1] for point in bbox)
        right = max(point[0] for point in bbox)
        lower = max(point[1] for point in bbox)

        # Modify mask for inpainting
        mask[upper:lower, left:right] = 255

        # Crop image to border
        cropped_word = np.array(image.crop((left, upper, right, lower)))
        raw_word = cropped_word.copy()

        # Binarise and filter word for processing
        cropped_word = cv2.cvtColor(cropped_word, cv2.COLOR_BGR2GRAY)
        cropped_word = cv2.adaptiveThreshold(cropped_word,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,25)

        # Crop image to remove any borders
        borders = find_borders(cropped_word, 85)
        cropped_word = cropped_word[borders[0]:borders[1] + 1, borders[2]:borders[3] + 1]
        raw_word = raw_word[borders[0]:borders[1] + 1, borders[2]:borders[3] + 1]

        # Save images for processing
        cv2.imwrite(f"Archive/{job_number}/raw/output_{word}.png", raw_word)
        cv2.imwrite(f"Archive/{job_number}/clean/output_{word}.png", cropped_word)

    # Inpainting base image
    image = np.array(image)
    mask = cv2.erode(mask, kernel, iterations=1)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=0, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(f"Archive/{job_number}/full/output_base.png", inpainted_image)

if __name__ == "__main__":
    job_number = generate_job_number()
    print(f"Starting job {job_number}...")
    path = f"Archive/{job_number}"
    os.makedirs(path, exist_ok=True)

    input_file = "input_image.jpeg"
    OCR_easy(input_file, job_number)

    print(f"Job {job_number} completed.")
