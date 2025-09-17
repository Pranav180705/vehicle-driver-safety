# anpr.py (improved)
import cv2
import pytesseract
import os
import csv
from datetime import datetime
import numpy as np

# --------- CONFIG ---------
VIDEO_FILE = "car_video.mp4"   # change if different
OUTPUT_CSV = "recognized_plates.csv"
SAVE_ROI_DEBUG = False         # set True to save ROI images for inspection
SHOW_ROI_WINDOW = False        # set True to pop-up ROIs while processing
MIN_CONFIDENCE_TO_SAVE = 30    # only save detections with this confidence
# path to tesseract binary (uncomment if needed)
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
# --------------------------

# initialize CSV if not exists
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "plate_text"])

# load cascade
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
if plate_cascade.empty():
    print("Error: couldn't load plate cascade. Check OpenCV installation.")
    exit(1)

# helper: clean OCR text
def clean_plate_text(s):
    if not s:
        return ""
    s = s.strip()
    # uppercase, keep alnum and common separators
    s = s.upper()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
    s = "".join(ch for ch in s if ch in allowed)
    s = s.replace(" ", "")
    return s

def ocr_plate_text(roi):
    """
    Attempt OCR with preprocessing and different configurations.
    Returns (best_text, best_confidence)
    """
    # convert to gray
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # try resizing to width ~400 for better OCR
    h, w = gray.shape
    target_w = 400
    scale = max(1, target_w // max(1, w))
    if scale > 1:
        gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    # morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # create two variants: thresholded and plain
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    candidates = []
    # try multiple OCR configs
    configs = [
        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",   # single text line
        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",   # treat as single word
        "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"    # assume block of text
    ]

    # run OCR on thresholded and adapted images and plain gray
    images = [th, th_adapt, gray]
    for img in images:
        for cfg in configs:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg, lang='eng')
            texts = data.get("text", [])
            confs = data.get("conf", [])
            # pick the highest-conf word (>=0)
            best_text = ""
            best_conf = -1.0
            for t, c in zip(texts, confs):
                if not t or t.strip() == "":
                    continue
                try:
                    conf_val = float(c)
                except:
                    conf_val = -1.0
                t_clean = clean_plate_text(t)
                if len(t_clean) >= 2 and conf_val > best_conf:
                    best_text = t_clean
                    best_conf = conf_val

            # fallback: full string
            if best_text == "":
                full = pytesseract.image_to_string(img, config=cfg, lang='eng').strip()
                full_clean = clean_plate_text(full)
                if full_clean:
                    best_text = full_clean
                    best_conf = 0.0

            candidates.append((best_text, best_conf, cfg))

    # choose best candidate by confidence then length
    best = ("", -1.0, "")
    for t, c, cfg in candidates:
        if not t:
            continue
        # prefer longer strings with reasonable conf
        score = (len(t) * 2) + (c if c is not None else 0)
        best_score = (len(best[0]) * 2) + (best[1] if best[1] is not None else 0)
        if score > best_score:
            best = (t, c, cfg)

    return best[0], (best[1] if best[1] is not None else 0.0)

# open video
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Error: could not open video file '{VIDEO_FILE}'.")
    exit(1)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(60,20))

    for (x, y, w, h) in plates:
        # expand roi
        pad_x = max(2, int(0.06 * w))
        pad_y = max(2, int(0.12 * h))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        roi = frame[y1:y2, x1:x2]

        # debug: show ROI
        if SHOW_ROI_WINDOW:
            cv2.imshow("ROI", cv2.resize(roi, (400, int(400 * roi.shape[0] / roi.shape[1]))))

        plate_text, conf = ocr_plate_text(roi)

        # draw green rectangle around plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = plate_text if plate_text else "Unknown"
        if len(label) > 40:
            label = label[:40] + "..."

        # position label above box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_x = x1
        label_y = max(0, y1 - 10)

        # draw label background rectangle (filled)
        cv2.rectangle(frame, (label_x, label_y - text_h - baseline), (label_x + text_w, label_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # log to console
        print(f"[Frame {frame_idx}] Detected ROI -> OCR='{plate_text}' conf={conf}")

        # save ROI images optionally for offline inspection
        if SAVE_ROI_DEBUG:
            fname = f"roi_{frame_idx}_{x1}_{y1}.png"
            cv2.imwrite(fname, roi)

        # save to CSV if confident
        if plate_text and conf >= MIN_CONFIDENCE_TO_SAVE:
            ts = datetime.utcnow().isoformat()
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, plate_text])

    cv2.imshow("ANPR (Tesseract)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")


