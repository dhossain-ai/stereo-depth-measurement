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

# Harris corner detection parameters
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01

# Corner refinement parameters
NMS_DISTANCE = 10           # Minimum distance between corners (pixels)
MAX_CORNERS = 1500          # Maximum number of corners to keep
SUBPIX_WIN = (5, 5)         # Sub-pixel refinement window
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale left:  {gray_left.shape}, dtype={gray_left.dtype}")
    print(f"Grayscale right: {gray_right.shape}, dtype={gray_right.dtype}")

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

def detect_harris_corners(blur_img):
    """
    Detect corners using Harris-Stephens corner detector.
    
    The Harris response R is computed as:
        R = det(M) - k * trace(M)^2
    where M is the structure tensor (second moment matrix).
    """
    img_float = np.float32(blur_img)
    harris_response = cv2.cornerHarris(img_float, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)

    print(f"Harris response: min={harris_response.min():.6f}, max={harris_response.max():.6f}")

    threshold = HARRIS_THRESHOLD * harris_response.max()
    corner_mask = harris_response > threshold

    corners_yx = np.argwhere(corner_mask)
    corners_xy = corners_yx[:, ::-1]

    print(f"Corners detected: {len(corners_xy)} (threshold={HARRIS_THRESHOLD} × max)")

    return harris_response, corners_xy

def display_harris_left(img_left, harris_response, corners_xy):
    """Display Harris corner response heatmap and detected corners on left image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(harris_response, cmap="hot")
    axes[0].set_title("Harris Corner Response (Left)")
    axes[0].axis("off")

    img_display = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB).copy()
    axes[1].imshow(img_display)
    axes[1].scatter(corners_xy[:, 0], corners_xy[:, 1],
                    c="red", s=2, alpha=0.6)
    axes[1].set_title(f"Detected Corners: {len(corners_xy)} (Left)")
    axes[1].axis("off")

    plt.suptitle("Harris-Stephens Corner Detection — Left Image", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "03_harris_left.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def display_harris_both(img_left, img_right, corners_left, corners_right):
    """Display Harris corners on both images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB).copy()
    axes[0].imshow(img_left_rgb)
    axes[0].scatter(corners_left[:, 0], corners_left[:, 1],
                    c="red", s=2, alpha=0.6)
    axes[0].set_title(f"Left — {len(corners_left)} corners")
    axes[0].axis("off")

    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB).copy()
    axes[1].imshow(img_right_rgb)
    axes[1].scatter(corners_right[:, 0], corners_right[:, 1],
                    c="cyan", s=2, alpha=0.6)
    axes[1].set_title(f"Right — {len(corners_right)} corners")
    axes[1].axis("off")

    plt.suptitle("Harris-Stephens Corner Detection — Both Images", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "04_harris_both.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def refine_corners(blur_img, harris_response, corners_xy):
    """
    Refine corners using non-maximum suppression and sub-pixel accuracy.
    
    1. NMS: Use cv2.goodFeaturesToTrack with Harris mode to select
       well-spaced, strongest corners (enforces minimum distance).
    2. Sub-pixel: Use cv2.cornerSubPix for sub-pixel localization.
    """
    # Use goodFeaturesToTrack with Harris mode for built-in NMS
    # This selects the top corners with minimum distance between them
    corners_gftt = cv2.goodFeaturesToTrack(
        blur_img,
        maxCorners=MAX_CORNERS,
        qualityLevel=HARRIS_THRESHOLD,
        minDistance=NMS_DISTANCE,
        useHarrisDetector=True,
        k=HARRIS_K
    )

    if corners_gftt is None:
        print("No corners survived refinement!")
        return np.array([])

    print(f"After NMS (minDistance={NMS_DISTANCE}): {len(corners_gftt)} corners")

    # Sub-pixel refinement for better accuracy
    corners_subpix = cv2.cornerSubPix(
        blur_img,
        corners_gftt,
        SUBPIX_WIN,
        SUBPIX_ZERO_ZONE,
        SUBPIX_CRITERIA
    )

    # Reshape from (N, 1, 2) to (N, 2)
    corners_refined = corners_subpix.reshape(-1, 2)
    print(f"After sub-pixel refinement: {len(corners_refined)} corners")

    return corners_refined

def display_refinement(img_left, img_right, corners_left_raw, corners_right_raw,
                        corners_left_ref, corners_right_ref):
    """Display before/after refinement comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Before refinement
    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(img_left_rgb)
    axes[0, 0].scatter(corners_left_raw[:, 0], corners_left_raw[:, 1],
                       c="red", s=2, alpha=0.4)
    axes[0, 0].set_title(f"Left BEFORE — {len(corners_left_raw)} corners")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_right_rgb)
    axes[0, 1].scatter(corners_right_raw[:, 0], corners_right_raw[:, 1],
                       c="red", s=2, alpha=0.4)
    axes[0, 1].set_title(f"Right BEFORE — {len(corners_right_raw)} corners")
    axes[0, 1].axis("off")

    # After refinement
    axes[1, 0].imshow(img_left_rgb)
    axes[1, 0].scatter(corners_left_ref[:, 0], corners_left_ref[:, 1],
                       c="lime", s=8, alpha=0.8, edgecolors="black", linewidths=0.3)
    axes[1, 0].set_title(f"Left AFTER — {len(corners_left_ref)} corners")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_right_rgb)
    axes[1, 1].scatter(corners_right_ref[:, 0], corners_right_ref[:, 1],
                       c="lime", s=8, alpha=0.8, edgecolors="black", linewidths=0.3)
    axes[1, 1].set_title(f"Right AFTER — {len(corners_right_ref)} corners")
    axes[1, 1].axis("off")

    plt.suptitle("Corner Refinement: NMS + Sub-Pixel Accuracy", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "05_refinement.png")
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

    # Step 4: Harris corner detection on left image
    print("\n[Step 4] Harris corner detection (left image)...")
    harris_response_left, corners_left = detect_harris_corners(blur_left)
    display_harris_left(img_left, harris_response_left, corners_left)

    # Step 5: Harris corner detection on right image
    print("\n[Step 5] Harris corner detection (right image)...")
    harris_response_right, corners_right = detect_harris_corners(blur_right)
    display_harris_both(img_left, img_right, corners_left, corners_right)

    # Step 6: Corner refinement (NMS + sub-pixel)
    print("\n[Step 6] Corner refinement...")
    corners_left_refined = refine_corners(blur_left, harris_response_left, corners_left)
    corners_right_refined = refine_corners(blur_right, harris_response_right, corners_right)
    display_refinement(img_left, img_right, corners_left, corners_right,
                       corners_left_refined, corners_right_refined)

    print("\n" + "=" * 40)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()