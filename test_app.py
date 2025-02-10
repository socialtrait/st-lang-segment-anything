url = "http://localhost:8000/predict"

import requests
import base64
import cv2
import numpy as np
import os

os.makedirs("outs", exist_ok=True)

fname = "example_images/cadbury.jpg"
with open(fname, "rb") as f:

    files = {"image": f}

    data = {
        "sam_type": "sam2.1_hiera_small",
        "box_threshold": "0.5",
        "text_threshold": "0.25",
        "text_prompt": "cadbury",
    }
    response = requests.post(url, files=files, data=data)

import cv2

img_bytes = response.json()["image"]
img_bytes = base64.b64decode(img_bytes)
img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
cv2.imwrite("outs/cadbury_sam.jpg", img)

results = response.json()["results"]

for mask, box, score, label in zip(
    results["masks"], results["boxes"], results["scores"], results["labels"]
):
    mask = (np.array(mask) * 255).astype(np.uint8)
    mask = mask.reshape(img.shape[0], img.shape[1])
    cv2.imwrite(f"outs/cadbury_sam_{label}.jpg", mask)
