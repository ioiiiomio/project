import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

BASE_TEST = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/test/img"
MODEL_DIR = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/models"
RESULT_DIR = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/results/simple_mod"

os.makedirs(RESULT_DIR, exist_ok=True)

STYLE = "poster"   
MODEL_PATH = os.path.join(MODEL_DIR, f"autoencoder_{STYLE}.keras")

from tensorflow.keras.losses import MeanSquaredError

model = load_model(MODEL_PATH)

print("Loaded model:", MODEL_PATH)

IMAGE_SIZE = (256, 256)

def load_preprocessed_test_images(folder):
    imgs = []
    names = []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        path = os.path.join(folder, fname)

        img = cv2.imread(path, cv2.IMREAD_COLOR)            # LOAD COLOR
        if img is None:
            print(f"Skipping unreadable: {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR → RGB
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype("float32") / 255.0                 # Normalize to [0,1]

        imgs.append(img)
        names.append(fname)

    return np.array(imgs), names

test_imgs, test_names = load_preprocessed_test_images(BASE_TEST)

print("Loaded test images:", test_imgs.shape)

preds = model.predict(test_imgs)
print("Generated outputs:", preds.shape)


out_folder = os.path.join(RESULT_DIR, STYLE)
os.makedirs(out_folder, exist_ok=True)

for img_arr, fname in zip(preds, test_names):
    out = (img_arr * 255).astype("uint8")
    out = out.squeeze()  # remove channel dim
    cv2.imwrite(os.path.join(out_folder, fname), out)

print("Saved results to:", out_folder)

def show_results_grid(inputs, outputs, names, rows=6):
    n = min(len(inputs), rows)
    plt.figure(figsize=(8, n * 4))

    for i in range(n):
        inp = inputs[i]
        out = outputs[i]

        # Input image
        plt.subplot(n, 2, i*2 + 1)
        plt.imshow(inp)
        plt.title(f"Input: {names[i]}")
        plt.axis("off")

        # Stylized output
        plt.subplot(n, 2, i*2 + 2)
        plt.imshow(out)
        plt.title(f"Stylized ({STYLE})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


show_results_grid(test_imgs, preds, test_names, rows=3)


# import os

# MODEL_DIR = "/Users/amayakof/Desktop/2025_autumn/deep_learning/SIS/3/project/models"
# print(os.listdir(MODEL_DIR))
