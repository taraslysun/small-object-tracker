import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

INPUT_FOLDER = "data/images/bees_medium/blurred_frames"
OUTPUT_FILE = "detections.npy"

HISTORY = 500
DIST_THRESHOLD = 800
DETECT_SHADOWS = False

ERODE_SIZE = (3, 3)
DILATE_SIZE = (8, 8)
MIN_AREA = 30
MAX_AREA = 1000

files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.png")))
if not files:
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")))

if not files:
    print("Error: No images found.")
else:
    print(f"Generating clean detections from {len(files)} frames...")

    back_sub = cv2.createBackgroundSubtractorKNN(
        history=HISTORY,
        dist2Threshold=DIST_THRESHOLD,
        detectShadows=DETECT_SHADOWS
    )

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ERODE_SIZE)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATE_SIZE)

    sample = cv2.imread(files[0])
    h, w, _ = sample.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    points = np.array([
        [0, 0],
        [int(w*0.65), 0],
        [int(w*0.80), h],
        [0, h]
    ])
    cv2.fillPoly(roi_mask, [points], 255)

    all_detections = []

    for i, file_path in enumerate(files):
        frame = cv2.imread(file_path)

        fg_mask = back_sub.apply(frame)

        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        clean_mask = cv2.erode(fg_mask, kernel_erode, iterations=1)
        clean_mask = cv2.dilate(clean_mask, kernel_dilate, iterations=2)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, wa, ha = cv2.boundingRect(cnt)
                aspect = float(wa)/ha
                if 0.4 < aspect < 2.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        all_detections.append([i, cX, cY])

        if i % 100 == 0:
            print(f"Processed frame {i}...")

    all_detections = np.array(all_detections)
    np.save(OUTPUT_FILE, all_detections)
    print(f"\nSUCCESS! Saved {len(all_detections)} clean bee detections to '{OUTPUT_FILE}'")

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB))
    plt.title("Final Mask: Notice the Right Side is Pure Black (Perfect)")
    plt.axis('off')
    plt.show()