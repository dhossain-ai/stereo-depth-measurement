"""
Stereo Vision Depth Measurement
================================
Measures depth to objects using a stereo image pair.
Harris-Stephens corner detection + NCC matching + disparity-to-depth.

Course: Image Recognition Systems
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Configuration
# ============================================================
LEFT_IMAGE = "input/aloeL.jpg"
RIGHT_IMAGE = "input/aloeR.jpg"
OUTPUT_DIR = "output"

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.0

def load_stereo_pair(left_path, right_path):
    """Load left and right stereo images."""
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None:
        raise FileNotFoundError(f"Could not load left image: {left_path}")
    if img_right is None:
        raise FileNotFoundError(f"Could not load right image: {right_path}")

    print(f"Left image loaded:  {img_left.shape}")
    print(f"Right image loaded: {img_right.shape}")

    return img_left, img_right

def display_stereo_pair(img_left, img_right):
    """Display left and right images side by side and save to output."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Left Image (aloeL.jpg)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Right Image (aloeR.jpg)")
    axes[1].axis("off")

    plt.suptitle("Stereo Image Pair", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "01_stereo_pair.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def preprocess(img_left, img_right):
    """Convert to grayscale and apply Gaussian blur for noise reduction."""
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale left:  {gray_left.shape}, dtype={gray_left.dtype}")
    print(f"Grayscale right: {gray_right.shape}, dtype={gray_right.dtype}")

    # Apply Gaussian blur to reduce noise
    blur_left = cv2.GaussianBlur(gray_left, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    blur_right = cv2.GaussianBlur(gray_right, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    print(f"Gaussian blur applied: kernel={GAUSSIAN_KERNEL}, sigma={GAUSSIAN_SIGMA}")

    return gray_left, gray_right, blur_left, blur_right

def display_preprocessing(gray_left, gray_right, blur_left, blur_right):
    """Display grayscale and blurred images in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(gray_left, cmap="gray")
    axes[0, 0].set_title("Left — Grayscale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_right, cmap="gray")
    axes[0, 1].set_title("Right — Grayscale")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(blur_left, cmap="gray")
    axes[1, 0].set_title(f"Left — Gaussian Blur {GAUSSIAN_KERNEL}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(blur_right, cmap="gray")
    axes[1, 1].set_title(f"Right — Gaussian Blur {GAUSSIAN_KERNEL}")
    axes[1, 1].axis("off")

    plt.suptitle("Preprocessing: Grayscale + Gaussian Blur", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "02_preprocessing.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def main():
    """Main pipeline for stereo depth measurement."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Stereo Depth Measurement Pipeline")
    print("=" * 40)

    # Step 1: Load stereo image pair
    print("\n[Step 1] Loading stereo image pair...")
    img_left, img_right = load_stereo_pair(LEFT_IMAGE, RIGHT_IMAGE)

    # Step 2: Display stereo pair
    print("\n[Step 2] Displaying stereo pair...")
    display_stereo_pair(img_left, img_right)

    # Step 3: Preprocessing
    print("\n[Step 3] Preprocessing...")
    gray_left, gray_right, blur_left, blur_right = preprocess(img_left, img_right)
    display_preprocessing(gray_left, gray_right, blur_left, blur_right)

    print("\n" + "=" * 40)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()