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
LEFT_IMAGE = "aloeL.jpg"
RIGHT_IMAGE = "aloeR.jpg"
OUTPUT_DIR = "output"

def main():
    """Main pipeline for stereo depth measurement."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Stereo Depth Measurement Pipeline")
    print("=" * 40)
    print(f"Left image:  {LEFT_IMAGE}")
    print(f"Right image: {RIGHT_IMAGE}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("=" * 40)
    print("Pipeline steps will be added in subsequent commits.")

if __name__ == "__main__":
    main()