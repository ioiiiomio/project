import os
import cv2
import numpy as np

# -------- PATHS --------
BASE_INPUT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/raw/input"
PREP_INPUT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/preprocessed/input"

POSTER_DIR = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/preprocessed/style/poster"
OUTLINE_DIR = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/preprocessed/style/outline"

RESIZE_TO = (256, 256)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

os.makedirs(PREP_INPUT, exist_ok=True)
os.makedirs(POSTER_DIR, exist_ok=True)
os.makedirs(OUTLINE_DIR, exist_ok=True)


def posterize_bgr(img, levels=6):
    """Mild posterization in BGR."""
    x = img.astype(np.float32) / 255.0
    x = np.floor(x * levels) / levels
    return (x * 255).astype(np.uint8)


def outlines_soft(img, levels=6):
    """
    SOFT OUTLINE VERSION:
    - lighter outlines (dark gray instead of black)
    - thinner lines
    - combines posterization + color smoothing + soft edges
    """

    # 1) Posterize base
    base = posterize_bgr(img, levels=levels)

    # 2) Smooth color (bilateral filter)
    smooth = cv2.bilateralFilter(base, d=9, sigmaColor=50, sigmaSpace=50)

    # 3) Extract edges
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)              # LOWER thresholds → thinner lines

    # 4) Blur edges to make them softer & smoother
    edges_soft = cv2.GaussianBlur(edges, (5,5), 0)

    # 5) Convert edges to RGB
    edges_rgb = cv2.cvtColor(edges_soft, cv2.COLOR_GRAY2BGR)

    # 6) Lighten the outline color (dark gray instead of black)
    #    0 would be pure black → we make edges ~30% black
    light_edges = 255 - edges_rgb  # invert
    light_edges = (light_edges * 0.30).astype(np.uint8) + 180

    # 7) Alpha-blend outlines on top of smooth colors
    #    alpha controls how strong outlines are
    alpha = 0.45  # lower → lighter lines
    output = cv2.addWeighted(smooth, 1.0, light_edges, alpha, 0)

    return output

    """
    Style C: posterize + smooth color + dark outlines.
    Everything stays in BGR.
    """
    # 1) Posterize base
    base = posterize_bgr(img, levels=levels)

    # 2) Smooth colors
    color = cv2.bilateralFilter(base, d=9, sigmaColor=75, sigmaSpace=75)

    # 3) Edges from grayscale
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    # Slightly thicken outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 4) Invert edges so lines are dark on bright
    edges_inv = 255 - edges
    edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # 5) Combine: keep color only where edge mask is bright
    out = cv2.bitwise_and(color, edges_inv)
    return out


# -------------- MAIN LOOP --------------
for fname in sorted(os.listdir(BASE_INPUT)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in VALID_EXT:
        continue

    src_path = os.path.join(BASE_INPUT, fname)

    img = cv2.imread(src_path)
    if img is None:
        print("Skipping unreadable:", src_path)
        continue

    # resize first for consistency
    img = cv2.resize(img, RESIZE_TO)

    # save preprocessed input copy
    cv2.imwrite(os.path.join(PREP_INPUT, fname), img)

    # outlines style C
    outl = outlines_soft(img, levels=6)
    cv2.imwrite(os.path.join(OUTLINE_DIR, fname), outl)

    print("Processed:", fname)

print("\nStyle generation complete.")
