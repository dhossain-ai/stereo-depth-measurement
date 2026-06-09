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
from scipy.interpolate import griddata
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
EPIPOLAR_TOLERANCE = 5
MIN_DISPARITY = 1
MAX_DISPARITY = 200

# Depth estimation parameters
# Estimated values for the aloe stereo pair
# focal_length: ~2740px (estimated from image width ~1282px and ~65 degree FOV)
# baseline: ~65mm (typical stereo camera separation)
FOCAL_LENGTH_PX = 2740.0    # Estimated focal length in pixels
BASELINE_MM = 65.0          # Estimated baseline in millimeters

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
    """Display left and right images side by side."""
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
    """Convert to grayscale and apply Gaussian blur."""
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
    """Detect corners using Harris-Stephens corner detector."""
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
    """Display Harris response heatmap and detected corners on left image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(harris_response, cmap="hot")
    axes[0].set_title("Harris Corner Response (Left)")
    axes[0].axis("off")

    img_display = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB).copy()
    axes[1].imshow(img_display)
    axes[1].scatter(corners_xy[:, 0], corners_xy[:, 1], c="red", s=2, alpha=0.6)
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

    axes[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    axes[0].scatter(corners_left[:, 0], corners_left[:, 1], c="red", s=2, alpha=0.6)
    axes[0].set_title(f"Left — {len(corners_left)} corners")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    axes[1].scatter(corners_right[:, 0], corners_right[:, 1], c="cyan", s=2, alpha=0.6)
    axes[1].set_title(f"Right — {len(corners_right)} corners")
    axes[1].axis("off")

    plt.suptitle("Harris-Stephens Corner Detection — Both Images", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "04_harris_both.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def refine_corners(blur_img, harris_response, corners_xy):
    """Refine corners using NMS and sub-pixel accuracy."""
    corners_gftt = cv2.goodFeaturesToTrack(
        blur_img, maxCorners=MAX_CORNERS, qualityLevel=HARRIS_THRESHOLD,
        minDistance=NMS_DISTANCE, useHarrisDetector=True, k=HARRIS_K
    )

    if corners_gftt is None:
        print("No corners survived refinement!")
        return np.array([])

    print(f"After NMS (minDistance={NMS_DISTANCE}): {len(corners_gftt)} corners")

    corners_subpix = cv2.cornerSubPix(
        blur_img, corners_gftt, SUBPIX_WIN, SUBPIX_ZERO_ZONE, SUBPIX_CRITERIA
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
    axes[0, 0].scatter(corners_left_raw[:, 0], corners_left_raw[:, 1], c="red", s=2, alpha=0.4)
    axes[0, 0].set_title(f"Left BEFORE — {len(corners_left_raw)} corners")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_right_rgb)
    axes[0, 1].scatter(corners_right_raw[:, 0], corners_right_raw[:, 1], c="red", s=2, alpha=0.4)
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
    """Extract NxN normalized intensity patches around each corner."""
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

        patches.append((patch - mean) / std)
        valid_corners.append([x, y])

    patches = np.array(patches)
    valid_corners = np.array(valid_corners)

    print(f"Patches extracted: {len(patches)} (patch_size={patch_size}x{patch_size})")
    print(f"Corners discarded (border/flat): {len(corners) - len(valid_corners)}")

    return patches, valid_corners

def display_patches(blur_img, valid_corners, patches):
    """Display a sample of extracted patches."""
    num_samples = min(10, len(patches))
    indices = np.sort(np.random.choice(len(patches), num_samples, replace=False))

    fig, axes = plt.subplots(2, num_samples, figsize=(16, 5))

    for i, idx in enumerate(indices):
        cx, cy = valid_corners[idx]
        half_view = 30
        xi, yi = int(round(cx)), int(round(cy))
        y1, y2 = max(0, yi - half_view), min(blur_img.shape[0], yi + half_view)
        x1, x2 = max(0, xi - half_view), min(blur_img.shape[1], xi + half_view)

        axes[0, i].imshow(blur_img[y1:y2, x1:x2], cmap="gray")
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
    """Match features using Normalized Cross-Correlation."""
    n_left, n_right = len(patches_left), len(patches_right)
    n_pixels = patches_left[0].size

    print(f"Matching {n_left} left patches against {n_right} right patches...")

    left_flat = patches_left.reshape(n_left, -1)
    right_flat = patches_right.reshape(n_right, -1)
    ncc_matrix = (left_flat @ right_flat.T) / n_pixels

    matches, ncc_scores = [], []
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
    """Filter matches using epipolar and disparity constraints."""
    filtered_matches, filtered_scores = [], []
    rejected_epipolar = rejected_disparity = 0

    for k, (i, j) in enumerate(matches):
        left_pt, right_pt = corners_left[i], corners_right[j]

        if abs(left_pt[1] - right_pt[1]) > EPIPOLAR_TOLERANCE:
            rejected_epipolar += 1
            continue

        disparity = left_pt[0] - right_pt[0]
        if disparity < MIN_DISPARITY or disparity > MAX_DISPARITY:
            rejected_disparity += 1
            continue

        filtered_matches.append((i, j))
        filtered_scores.append(ncc_scores[k])

    filtered_matches = np.array(filtered_matches)
    filtered_scores = np.array(filtered_scores)

    print(f"Matches before filtering: {len(matches)}")
    print(f"Rejected (epipolar): {rejected_epipolar}")
    print(f"Rejected (disparity): {rejected_disparity}")
    print(f"Matches after filtering: {len(filtered_matches)}")

    return filtered_matches, filtered_scores

def display_filtering(corners_left, corners_right, matches_before, matches_after):
    """Display match filtering histograms."""
    def get_stats(matches):
        yd, disp = [], []
        for (i, j) in matches:
            lp, rp = corners_left[i], corners_right[j]
            yd.append(lp[1] - rp[1])
            disp.append(lp[0] - rp[0])
        return yd, disp

    yd_b, disp_b = get_stats(matches_before)
    yd_a, disp_a = get_stats(matches_after)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(yd_b, bins=50, alpha=0.5, color="red", label=f"Before ({len(matches_before)})")
    axes[0].hist(yd_a, bins=50, alpha=0.7, color="green", label=f"After ({len(matches_after)})")
    axes[0].axvline(-EPIPOLAR_TOLERANCE, color="blue", linestyle="--", label=f"±{EPIPOLAR_TOLERANCE}px")
    axes[0].axvline(EPIPOLAR_TOLERANCE, color="blue", linestyle="--")
    axes[0].set_xlabel("Y-difference (left_y - right_y)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Epipolar Constraint Filter")
    axes[0].legend()

    axes[1].hist(disp_b, bins=50, alpha=0.5, color="red", label=f"Before ({len(matches_before)})")
    axes[1].hist(disp_a, bins=50, alpha=0.7, color="green", label=f"After ({len(matches_after)})")
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

def display_matches(img_left, img_right, corners_left, corners_right, matches, scores):
    """Draw matched feature pairs using cv2.drawMatches."""
    kp_left  = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=PATCH_SIZE) for pt in corners_left]
    kp_right = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=PATCH_SIZE) for pt in corners_right]

    dmatches = [cv2.DMatch(_queryIdx=int(i), _trainIdx=int(j), _distance=1.0 - s)
                for (i, j), s in zip(matches, scores)]
    dmatches_sorted = sorted(dmatches, key=lambda m: m.distance)

    num_draw = min(50, len(dmatches_sorted))
    match_img = cv2.drawMatches(
        cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB), kp_left,
        cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB), kp_right,
        dmatches_sorted[:num_draw], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.imshow(match_img)
    ax.set_title(f"Top {num_draw} Matched Pairs (of {len(matches)} total, sorted by NCC score)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "08_matches.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def compute_disparities(corners_left, corners_right, matches, scores):
    """Compute disparity (left_x - right_x) for each matched pair."""
    left_points, right_points, disparities = [], [], []

    for (i, j) in matches:
        lp, rp = corners_left[i], corners_right[j]
        left_points.append(lp)
        right_points.append(rp)
        disparities.append(lp[0] - rp[0])

    left_points  = np.array(left_points)
    right_points = np.array(right_points)
    disparities  = np.array(disparities)

    print(f"Disparities computed: {len(disparities)}")
    print(f"Disparity range: {disparities.min():.2f} — {disparities.max():.2f} pixels")
    print(f"Disparity mean: {disparities.mean():.2f}, std: {disparities.std():.2f}")

    return left_points, right_points, disparities

def display_disparities(img_left, left_points, disparities):
    """Display disparity scatter plot and histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_left_rgb)
    sc = axes[0].scatter(left_points[:, 0], left_points[:, 1],
                         c=disparities, cmap="jet", s=15, alpha=0.8,
                         edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=axes[0], label="Disparity (pixels)")
    axes[0].set_title(f"Disparity at Matched Points ({len(disparities)} points)")
    axes[0].axis("off")

    axes[1].hist(disparities, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Disparity (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Disparity Distribution")
    axes[1].axvline(disparities.mean(), color="red", linestyle="--",
                    label=f"Mean = {disparities.mean():.1f}px")
    axes[1].legend()

    plt.suptitle("Disparity Calculation", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "09_disparities.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def create_disparity_map(img_left, left_points, disparities):
    """Interpolate sparse disparities into a dense disparity map."""
    h, w = img_left.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    disp_linear  = griddata(left_points, disparities, (grid_x, grid_y), method="linear")
    disp_nearest = griddata(left_points, disparities, (grid_x, grid_y), method="nearest")

    disparity_map = np.where(np.isnan(disp_linear), disp_nearest, disp_linear)

    print(f"Disparity map shape: {disparity_map.shape}")
    print(f"Disparity map range: {disparity_map.min():.2f} — {disparity_map.max():.2f}")

    return disparity_map

def display_disparity_map(img_left, disparity_map):
    """Display the interpolated disparity map."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    axes[0].imshow(img_left_rgb)
    axes[0].set_title("Original Left Image")
    axes[0].axis("off")

    im = axes[1].imshow(disparity_map, cmap="jet")
    plt.colorbar(im, ax=axes[1], label="Disparity (pixels)")
    axes[1].set_title("Disparity Map (Interpolated)")
    axes[1].axis("off")

    axes[2].imshow(img_left_rgb)
    axes[2].imshow(disparity_map, cmap="jet", alpha=0.5)
    axes[2].set_title("Disparity Overlay")
    axes[2].axis("off")

    plt.suptitle("Disparity Map Visualization", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "10_disparity_map.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

def estimate_depth(disparities, focal_length_px=FOCAL_LENGTH_PX, baseline_mm=BASELINE_MM):
    """
    Estimate depth from disparity using the stereo depth formula:

        depth (mm) = (focal_length_px × baseline_mm) / disparity_px

    Larger disparity → object is closer.
    Smaller disparity → object is farther away.
    """
    # Avoid division by zero
    safe_disparities = np.where(disparities > 0, disparities, 1e-6)
    depths_mm = (focal_length_px * baseline_mm) / safe_disparities
    depths_m  = depths_mm / 1000.0

    print(f"Depth estimation parameters:")
    print(f"  Focal length: {focal_length_px:.1f} px")
    print(f"  Baseline:     {baseline_mm:.1f} mm")
    print(f"Depth range: {depths_mm.min():.1f} — {depths_mm.max():.1f} mm "
          f"({depths_m.min():.3f} — {depths_m.max():.3f} m)")
    print(f"Depth mean:  {depths_mm.mean():.1f} mm ({depths_m.mean():.3f} m)")

    return depths_mm, depths_m

def display_final_output(img_left, left_points, disparities, depths_mm,
                          disparity_map, focal_length_px, baseline_mm):
    """
    Final output figure — 2x3 summary panel showing the full pipeline result.
    """
    h, w = img_left.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Build depth map from disparity map
    safe_disp_map = np.where(disparity_map > 0, disparity_map, 1e-6)
    depth_map_mm  = (focal_length_px * baseline_mm) / safe_disp_map

    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) Original image
    axes[0, 0].imshow(img_left_rgb)
    axes[0, 0].set_title("Original Left Image", fontsize=12)
    axes[0, 0].axis("off")

    # (0,1) Disparity scatter
    axes[0, 1].imshow(img_left_rgb)
    sc = axes[0, 1].scatter(left_points[:, 0], left_points[:, 1],
                             c=disparities, cmap="jet", s=15, alpha=0.9,
                             edgecolors="black", linewidths=0.2)
    plt.colorbar(sc, ax=axes[0, 1], label="Disparity (px)")
    axes[0, 1].set_title(f"Disparity at {len(disparities)} Matched Points", fontsize=12)
    axes[0, 1].axis("off")

    # (0,2) Disparity map
    im1 = axes[0, 2].imshow(disparity_map, cmap="jet")
    plt.colorbar(im1, ax=axes[0, 2], label="Disparity (px)")
    axes[0, 2].set_title("Interpolated Disparity Map", fontsize=12)
    axes[0, 2].axis("off")

    # (1,0) Depth scatter on image
    axes[1, 0].imshow(img_left_rgb)
    sc2 = axes[1, 0].scatter(left_points[:, 0], left_points[:, 1],
                              c=depths_mm, cmap="plasma_r", s=15, alpha=0.9,
                              edgecolors="black", linewidths=0.2)
    plt.colorbar(sc2, ax=axes[1, 0], label="Depth (mm)")
    axes[1, 0].set_title(f"Estimated Depth at Matched Points", fontsize=12)
    axes[1, 0].axis("off")

    # (1,1) Dense depth map
    im2 = axes[1, 1].imshow(depth_map_mm, cmap="plasma_r")
    plt.colorbar(im2, ax=axes[1, 1], label="Depth (mm)")
    axes[1, 1].set_title("Dense Depth Map", fontsize=12)
    axes[1, 1].axis("off")

    # (1,2) Depth overlay
    axes[1, 2].imshow(img_left_rgb)
    axes[1, 2].imshow(depth_map_mm, cmap="plasma_r", alpha=0.5)
    axes[1, 2].set_title("Depth Overlay", fontsize=12)
    axes[1, 2].axis("off")

    plt.suptitle(
        f"Final Depth Estimation  |  f={focal_length_px:.0f}px  |  B={baseline_mm:.0f}mm  |  "
        f"Depth: {depths_mm.min():.0f}–{depths_mm.max():.0f}mm  |  "
        f"Mean: {depths_mm.mean():.0f}mm ({depths_mm.mean()/1000:.2f}m)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "11_final_depth.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

    # Also save depth map as standalone PNG
    depth_map_norm = cv2.normalize(depth_map_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored  = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_PLASMA)
    depth_out_path = os.path.join(OUTPUT_DIR, "12_depth_map_final.png")
    cv2.imwrite(depth_out_path, depth_colored)
    print(f"Saved: {depth_out_path}")

def print_summary(disparities, depths_mm):
    """Print a clean final summary of all pipeline results."""
    print("\n" + "=" * 50)
    print("         STEREO DEPTH MEASUREMENT SUMMARY")
    print("=" * 50)
    print(f"  Matched points (after filtering): {len(disparities)}")
    print(f"  Disparity range:  {disparities.min():.1f} — {disparities.max():.1f} px")
    print(f"  Disparity mean:   {disparities.mean():.1f} px")
    print(f"  Focal length:     {FOCAL_LENGTH_PX:.0f} px")
    print(f"  Baseline:         {BASELINE_MM:.0f} mm")
    print(f"  Depth range:      {depths_mm.min():.0f} — {depths_mm.max():.0f} mm")
    print(f"  Depth mean:       {depths_mm.mean():.0f} mm  ({depths_mm.mean()/1000:.2f} m)")
    print(f"  Output folder:    {OUTPUT_DIR}/")
    print("=" * 50)

def main():
    """Main pipeline for stereo depth measurement."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Stereo Depth Measurement Pipeline")
    print("=" * 40)

    print("\n[Step 1] Loading stereo image pair...")
    img_left, img_right = load_stereo_pair(LEFT_IMAGE, RIGHT_IMAGE)

    print("\n[Step 2] Displaying stereo pair...")
    display_stereo_pair(img_left, img_right)

    print("\n[Step 3] Preprocessing...")
    gray_left, gray_right, blur_left, blur_right = preprocess(img_left, img_right)
    display_preprocessing(gray_left, gray_right, blur_left, blur_right)

    print("\n[Step 4] Harris corner detection (left image)...")
    harris_response_left, corners_left = detect_harris_corners(blur_left)
    display_harris_left(img_left, harris_response_left, corners_left)

    print("\n[Step 5] Harris corner detection (right image)...")
    harris_response_right, corners_right = detect_harris_corners(blur_right)
    display_harris_both(img_left, img_right, corners_left, corners_right)

    print("\n[Step 6] Corner refinement...")
    corners_left_refined  = refine_corners(blur_left,  harris_response_left,  corners_left)
    corners_right_refined = refine_corners(blur_right, harris_response_right, corners_right)
    display_refinement(img_left, img_right, corners_left, corners_right,
                       corners_left_refined, corners_right_refined)

    print("\n[Step 7] Extracting feature descriptors...")
    patches_left,  valid_corners_left  = extract_patches(blur_left,  corners_left_refined)
    patches_right, valid_corners_right = extract_patches(blur_right, corners_right_refined)
    display_patches(blur_left, valid_corners_left, patches_left)

    print("\n[Step 8] Feature matching (NCC)...")
    matches, ncc_scores = match_features_ncc(
        patches_left, valid_corners_left,
        patches_right, valid_corners_right
    )

    print("\n[Step 9] Filtering matches...")
    filtered_matches, filtered_scores = filter_matches(
        matches, ncc_scores, valid_corners_left, valid_corners_right
    )
    display_filtering(valid_corners_left, valid_corners_right, matches, filtered_matches)

    print("\n[Step 10] Visualizing matches...")
    display_matches(img_left, img_right, valid_corners_left, valid_corners_right,
                    filtered_matches, filtered_scores)

    print("\n[Step 11] Computing disparities...")
    left_points, right_points, disparities = compute_disparities(
        valid_corners_left, valid_corners_right, filtered_matches, filtered_scores
    )
    display_disparities(img_left, left_points, disparities)

    print("\n[Step 12] Creating disparity map...")
    disparity_map = create_disparity_map(img_left, left_points, disparities)
    display_disparity_map(img_left, disparity_map)

    print("\n[Step 13] Estimating depth...")
    depths_mm, depths_m = estimate_depth(disparities)
    display_final_output(img_left, left_points, disparities, depths_mm,
                          disparity_map, FOCAL_LENGTH_PX, BASELINE_MM)

    print_summary(disparities, depths_mm)

if __name__ == "__main__":
    main()