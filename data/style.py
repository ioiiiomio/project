import os
import cv2
import numpy as np

# ------------------------------
# CONFIG
# ------------------------------
RAW_INPUT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/raw/input"
RAW_STYLE_ROOT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/raw/style"

PREP_INPUT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/preprocessed/input"
PREP_STYLE_ROOT = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/data/preprocessed/style"

RESIZE_TO = (256, 256)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

os.makedirs(PREP_INPUT, exist_ok=True)
os.makedirs(PREP_STYLE_ROOT, exist_ok=True)

# ------------------------------
# STYLE FUNCTIONS
# ------------------------------

def posterize_bgr(img, levels=6):
    x = img.astype(np.float32) / 255.0
    x = np.floor(x * levels) / levels
    return (x * 255).astype(np.uint8)


def outlines_soft(img, levels=6):
    base = posterize_bgr(img, levels=levels)
    smooth = cv2.bilateralFilter(base, d=9, sigmaColor=50, sigmaSpace=50)

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges_soft = cv2.GaussianBlur(edges, (5, 5), 0)

    edges_rgb = cv2.cvtColor(edges_soft, cv2.COLOR_GRAY2BGR)

    light_edges = 255 - edges_rgb
    light_edges = (light_edges * 0.30).astype(np.uint8) + 180

    alpha = 0.45
    output = cv2.addWeighted(smooth, 1.0, light_edges, alpha, 0)

    return output


# ------------------------------
# PROCESS RAW INPUT (resize only)
# ------------------------------
print("\n=== Processing RAW INPUT ===")
for fname in sorted(os.listdir(RAW_INPUT)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in VALID_EXT: 
        continue

    path = os.path.join(RAW_INPUT, fname)
    img = cv2.imread(path)

    if img is None:
        print("Skipping:", path)
        continue

    img = cv2.resize(img, RESIZE_TO)
    cv2.imwrite(os.path.join(PREP_INPUT, fname), img)

print("Input preprocessing done.")


# ------------------------------
# PROCESS RAW STYLE FOLDERS
# ------------------------------
print("\n=== Processing RAW STYLE FOLDERS ===")

# detect all style folders automatically
style_folders = [
    f for f in os.listdir(RAW_STYLE_ROOT)
    if os.path.isdir(os.path.join(RAW_STYLE_ROOT, f))
]

print("Found styles:", style_folders)

for style in style_folders:
    raw_style_dir = os.path.join(RAW_STYLE_ROOT, style)
    prep_style_dir = os.path.join(PREP_STYLE_ROOT, style)
    os.makedirs(prep_style_dir, exist_ok=True)

    print(f"\n-- Processing style '{style}' --")

    for fname in sorted(os.listdir(raw_style_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXT:
            continue

        path = os.path.join(raw_style_dir, fname)
        img = cv2.imread(path)

        if img is None:
            print("Skipping unreadable:", path)
            continue

        img = cv2.resize(img, RESIZE_TO)
        cv2.imwrite(os.path.join(prep_style_dir, fname), img)

print("Finished resizing all style folders.")


# ------------------------------
# GENERATE EXTRA STYLE VERSIONS (poster, outline)
# ------------------------------
print("\n=== Generating Poster & Outline Styles ===")

POSTER_OUT = os.path.join(PREP_STYLE_ROOT, "poster")
OUTLINE_OUT = os.path.join(PREP_STYLE_ROOT, "outline")

os.makedirs(POSTER_OUT, exist_ok=True)
os.makedirs(OUTLINE_OUT, exist_ok=True)

for fname in sorted(os.listdir(PREP_INPUT)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in VALID_EXT:
        continue

    img_path = os.path.join(PREP_INPUT, fname)
    img = cv2.imread(img_path)

    poster = posterize_bgr(img, levels=6)
    outline = outlines_soft(img, levels=6)

    cv2.imwrite(os.path.join(POSTER_OUT, fname), poster)
    cv2.imwrite(os.path.join(OUTLINE_OUT, fname), outline)

print("\n=== All preprocessing complete. ===")