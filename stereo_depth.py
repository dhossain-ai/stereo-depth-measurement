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
NMS_DISTANCE = 10
MAX_CORNERS = 1500
SUBPIX_WIN = (5, 5)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Feature descriptor parameters
PATCH_SIZE = 15

# NCC matching parameters
NCC_THRESHOLD = 0.8

# Match filtering parameters
EPIPOLAR_TOLERANCE = 5      # Max vertical (y) difference in pixels for epipolar constraint
MIN_DISPARITY = 1           # Minimum horizontal displacement (pixels)
MAX_DISPARITY = 200         # Maximum horizontal displacement (pixels)

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
    """
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

    corners_subpix = cv2.cornerSubPix(
        blur_img,
        corners_gftt,
        SUBPIX_WIN,
        SUBPIX_ZERO_ZONE,
        SUBPIX_CRITERIA
    )

    corners_refined = corners_subpix.reshape(-1, 2)
    print(f"After sub-pixel refinement: {len(corners_refined)} corners")

    return corners_refined

def display_refinement(img_left, img_right, corners_left_raw, corners_right_raw,
                        corners_left_ref, corners_right_ref):
    """Display before/after refinement comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

def extract_patches(blur_img, corners, patch_size=PATCH_SIZE):
    """
    Extract NxN intensity patches around each corner as feature descriptors.
    """
    half = patch_size // 2
    h, w = blur_img.shape
    patches = []
    valid_corners = []

    for (x, y) in corners:
        xi, yi = int(round(x)), int(round(y))

        if yi - half < 0 or yi + half >= h or xi - half < 0 or xi + half >= w:
            continue

        patch = blur_img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)

        mean = patch.mean()
        std = patch.std()
        if std < 1e-6:
            continue

        patch_normalized = (patch - mean) / std

        patches.append(patch_normalized)
        valid_corners.append([x, y])

    patches = np.array(patches)
    valid_corners = np.array(valid_corners)

    print(f"Patches extracted: {len(patches)} (patch_size={patch_size}x{patch_size})")
    print(f"Corners discarded (border/flat): {len(corners) - len(valid_corners)}")

    return patches, valid_corners

def display_patches(blur_img, valid_corners, patches):
    """Display a sample of extracted patches to verify descriptor extraction."""
    num_samples = min(10, len(patches))
    indices = np.random.choice(len(patches), num_samples, replace=False)
    indices = np.sort(indices)

    fig, axes = plt.subplots(2, num_samples, figsize=(16, 5))

    for i, idx in enumerate(indices):
        cx, cy = valid_corners[idx]

        half_view = 30
        xi, yi = int(round(cx)), int(round(cy))
        y1 = max(0, yi - half_view)
        y2 = min(blur_img.shape[0], yi + half_view)
        x1 = max(0, xi - half_view)
        x2 = min(blur_img.shape[1], xi + half_view)
        region = blur_img[y1:y2, x1:x2]

        axes[0, i].imshow(region, cmap="gray")
        axes[0, i].scatter([xi - x1], [yi - y1], c="red", s=20, marker="+")
        axes[0, i].set_title(f"Corner {idx}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(patches[idx], cmap="gray")
        axes[1, i].set_title(f"{PATCH_SIZE}×{PATCH_SIZE}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Region", fontsize=10)
    axes[1, 0].set_ylabel("Patch", fontsize=10)

    plt.suptitle("Sample Feature Descriptors (Normalized Patches)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "06_patches.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def match_features_ncc(patches_left, corners_left, patches_right, corners_right):
    """
    Match features between left and right images using Normalized Cross-Correlation.
    
    NCC formula for normalized patches (zero mean, unit variance):
        NCC(a, b) = sum(a * b) / N
    
    NCC = 1.0 means perfect match, NCC = -1.0 means inverse match.
    """
    n_left = len(patches_left)
    n_right = len(patches_right)
    n_pixels = patches_left[0].size

    print(f"Matching {n_left} left patches against {n_right} right patches...")

    left_flat = patches_left.reshape(n_left, -1)
    right_flat = patches_right.reshape(n_right, -1)

    ncc_matrix = (left_flat @ right_flat.T) / n_pixels

    matches = []
    ncc_scores = []

    for i in range(n_left):
        best_j = np.argmax(ncc_matrix[i])
        best_score = ncc_matrix[i, best_j]

        if best_score >= NCC_THRESHOLD:
            matches.append((i, best_j))
            ncc_scores.append(best_score)

    matches = np.array(matches)
    ncc_scores = np.array(ncc_scores)

    print(f"Matches found: {len(matches)} (NCC threshold={NCC_THRESHOLD})")
    if len(ncc_scores) > 0:
        print(f"NCC scores: min={ncc_scores.min():.4f}, max={ncc_scores.max():.4f}, "
              f"mean={ncc_scores.mean():.4f}")

    return matches, ncc_scores

def filter_matches(matches, ncc_scores, corners_left, corners_right):
    """
    Filter matches using epipolar and disparity constraints.
    
    Epipolar constraint: In a rectified stereo pair, matched points must
    lie on the same horizontal scanline. We allow a small tolerance for
    imperfect rectification.
    
    Disparity constraint: The horizontal displacement must be positive
    (left image point is to the right of right image point) and within
    a reasonable range.
    """
    filtered_matches = []
    filtered_scores = []
    
    rejected_epipolar = 0
    rejected_disparity = 0

    for k, (i, j) in enumerate(matches):
        left_pt = corners_left[i]    # (x, y)
        right_pt = corners_right[j]  # (x, y)

        # Epipolar constraint: y-coordinates must be similar
        y_diff = abs(left_pt[1] - right_pt[1])
        if y_diff > EPIPOLAR_TOLERANCE:
            rejected_epipolar += 1
            continue

        # Disparity: horizontal displacement (left_x - right_x)
        # In a standard stereo setup, objects appear shifted to the right
        # in the left image compared to the right image
        disparity = left_pt[0] - right_pt[0]
        if disparity < MIN_DISPARITY or disparity > MAX_DISPARITY:
            rejected_disparity += 1
            continue

        filtered_matches.append((i, j))
        filtered_scores.append(ncc_scores[k])

    filtered_matches = np.array(filtered_matches)
    filtered_scores = np.array(filtered_scores)

    print(f"Matches before filtering: {len(matches)}")
    print(f"Rejected (epipolar, |dy| > {EPIPOLAR_TOLERANCE}px): {rejected_epipolar}")
    print(f"Rejected (disparity not in [{MIN_DISPARITY}, {MAX_DISPARITY}]): {rejected_disparity}")
    print(f"Matches after filtering: {len(filtered_matches)}")

    return filtered_matches, filtered_scores

def display_filtering(corners_left, corners_right, matches_before, matches_after):
    """Display match filtering results — histogram of y-differences and disparities."""
    # Compute y-differences and disparities for all raw matches
    y_diffs_before = []
    disparities_before = []
    for (i, j) in matches_before:
        lp = corners_left[i]
        rp = corners_right[j]
        y_diffs_before.append(lp[1] - rp[1])
        disparities_before.append(lp[0] - rp[0])

    y_diffs_after = []
    disparities_after = []
    for (i, j) in matches_after:
        lp = corners_left[i]
        rp = corners_right[j]
        y_diffs_after.append(lp[1] - rp[1])
        disparities_after.append(lp[0] - rp[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Y-difference histogram
    axes[0].hist(y_diffs_before, bins=50, alpha=0.5, color="red", label=f"Before ({len(matches_before)})")
    axes[0].hist(y_diffs_after, bins=50, alpha=0.7, color="green", label=f"After ({len(matches_after)})")
    axes[0].axvline(-EPIPOLAR_TOLERANCE, color="blue", linestyle="--", label=f"±{EPIPOLAR_TOLERANCE}px")
    axes[0].axvline(EPIPOLAR_TOLERANCE, color="blue", linestyle="--")
    axes[0].set_xlabel("Y-difference (left_y - right_y)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Epipolar Constraint Filter")
    axes[0].legend()

    # Disparity histogram
    axes[1].hist(disparities_before, bins=50, alpha=0.5, color="red", label=f"Before ({len(matches_before)})")
    axes[1].hist(disparities_after, bins=50, alpha=0.7, color="green", label=f"After ({len(matches_after)})")
    axes[1].axvline(MIN_DISPARITY, color="blue", linestyle="--", label=f"[{MIN_DISPARITY}, {MAX_DISPARITY}]px")
    axes[1].axvline(MAX_DISPARITY, color="blue", linestyle="--")
    axes[1].set_xlabel("Disparity (left_x - right_x)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Disparity Range Filter")
    axes[1].legend()

    plt.suptitle("Match Filtering: Epipolar + Disparity Constraints", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "07_filtering.png")
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

    # Step 7: Feature descriptor extraction
    print("\n[Step 7] Extracting feature descriptors...")
    patches_left, valid_corners_left = extract_patches(blur_left, corners_left_refined)
    patches_right, valid_corners_right = extract_patches(blur_right, corners_right_refined)
    display_patches(blur_left, valid_corners_left, patches_left)

    # Step 8: Feature matching using NCC
    print("\n[Step 8] Feature matching (NCC)...")
    matches, ncc_scores = match_features_ncc(
        patches_left, valid_corners_left,
        patches_right, valid_corners_right
    )

    # Step 9: Match filtering (epipolar + disparity constraints)
    print("\n[Step 9] Filtering matches...")
    filtered_matches, filtered_scores = filter_matches(
        matches, ncc_scores, valid_corners_left, valid_corners_right
    )
    display_filtering(valid_corners_left, valid_corners_right, matches, filtered_matches)

    print("\n" + "=" * 40)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()