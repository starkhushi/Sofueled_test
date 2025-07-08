import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# ----------------------------
# CONFIGURATION
# ----------------------------
IMAGE_PATH = 'test_image.jpg'  # your test image
MODEL_PATH = 'runs/train/nails_detector/weights/best.pt'  # trained YOLOv5 model
PIXEL_TO_MM = 0.264  # adjust this based on known reference
OUTPUT_PATH = 'final_output.jpg'
CONFIDENCE_THRESHOLD = 0.3

# ----------------------------
# WEIGHT ESTIMATION FORMULA
# ----------------------------
def estimate_weight(height_mm):
    return round(0.002 * (height_mm ** 1.8), 2)  # sample estimation

# ----------------------------
# LOAD YOLOv5 MODEL
# ----------------------------
def load_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)

# ----------------------------
# RUN DETECTION
# ----------------------------
def detect_nails(model, image_path):
    results = model(image_path, size=640)
    detections = results.pandas().xyxy[0]
    return detections

# ----------------------------
# DRAW AND PROCESS
# ----------------------------
def process_detections(image, detections):
    results = []
    for idx, det in detections.iterrows():
        if det['confidence'] < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
        height_px = y2 - y1
        height_mm = round(height_px * PIXEL_TO_MM, 2)
        weight = estimate_weight(height_mm)
        results.append({
            "id": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "height_mm": height_mm,
            "weight_g": weight
        })
    return results

# ----------------------------
# GROUP SIMILAR NAILS
# ----------------------------
def group_nails(results):
    features = np.array([[r['height_mm'], r['weight_g']] for r in results])
    if len(features) >= 2:
        kmeans = KMeans(n_clusters=len(features) // 2 or 1, n_init=10)
        return kmeans.fit_predict(features)
    return [0] * len(features)

# ----------------------------
# DRAW RESULTS ON IMAGE
# ----------------------------
def draw_results(image, results, groups):
    colors = [(255,0,0), (0,255,0), (0,255,255), (255,255,0), (255,0,255), (200,100,100)]
    for r, group in zip(results, groups):
        color = colors[group % len(colors)]
        cv2.rectangle(image, (r['x1'], r['y1']), (r['x2'], r['y2']), color, 2)
        label = f"ID:{r['id']} H:{r['height_mm']}mm W:{r['weight_g']}g G{group}"
        cv2.putText(image, label, (r['x1'], r['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    print("🔍 Loading model...")
    model = load_model(MODEL_PATH)
    print("📸 Reading image...")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("❌ Image not found.")
        return

    print("🚀 Running detection...")
    detections = detect_nails(model, IMAGE_PATH)
    results = process_detections(image, detections)
    groups = group_nails(results)

    print(f"\n✅ Nails detected: {len(results)}")
    for r, g in zip(results, groups):
        print(f"• Nail {r['id']}: Height={r['height_mm']}mm | Weight={r['weight_g']}g | Group={g}")

    print("\n🎯 Drawing results...")
    image = draw_results(image, results, groups)
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"\n📁 Output saved as: {OUTPUT_PATH}")

    cv2.imshow("Final Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
