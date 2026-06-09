# Stereo Vision Depth Measurement

Stereo vision depth measurement using Harris-Stephens corner detection, patch-based NCC feature matching, and disparity-to-depth conversion on a calibrated stereo image pair.

## Task

Measure the depth to an object from two nearly identical images obtained from different cameras. The Harris-Stephens corner detector is used to detect common points, and their displacement (disparity) allows for the calculation of the depth to the object.

## Images

- `aloeL.jpg` — Left camera view
- `aloeR.jpg` — Right camera view

## Methods

1. **Preprocessing** — Grayscale conversion and Gaussian blur
2. **Harris-Stephens Corner Detection** — Detect interest points in both images
3. **Corner Refinement** — Sub-pixel accuracy refinement
4. **Feature Descriptor Extraction** — Extract NxN patches around corners
5. **Feature Matching (NCC)** — Normalized Cross-Correlation matching
6. **Match Filtering** — Epipolar constraint filtering
7. **Match Visualization** — Draw matched feature pairs
8. **Disparity Calculation** — Compute horizontal pixel displacement
9. **Disparity Map** — Color-coded disparity visualization
10. **Depth Estimation** — Convert disparity to depth using `depth = (f × B) / disparity`

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib