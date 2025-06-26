import os
import cv2
import csv

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

image_dir = "/Users/macbook/Downloads/division_sequence_Z2_6"
label_dir_green = "/Users/macbook/Downloads/Bachelorarbeit/CodeExampleRayan/yolov5/runs/detect/test_zstack_2_6_run/labels"
label_dir_red = "/Users/macbook/Downloads/Bachelorarbeit/CodeExampleRayan/yolov5/runs/detect/test_zstack_3_6_run/labels"
output_dir = "/Users/macbook/Downloads/Bachelorarbeit/output_combined_6"

os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(image_dir)):
    if not filename.endswith(".png"):
        continue

    print(f"🔄 Verarbeite Bild: {filename}")

    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Bild nicht lesbar: {image_path}")
        continue

    label_path_green = os.path.join(label_dir_green, filename.replace(".png", ".txt"))
    label_path_red = os.path.join(label_dir_red, filename.replace("Z2", "Z3").replace(".png", ".txt"))

    green_boxes = []
    if os.path.exists(label_path_green):
        with open(label_path_green, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x_center, y_center, width, height = map(float, parts[:5])
                    x = int((x_center - width / 2) * img.shape[1])
                    y = int((y_center - height / 2) * img.shape[0])
                    w = int(width * img.shape[1])
                    h = int(height * img.shape[0])
                    green_boxes.append((x, y, w, h))
    else:
        print(f"Keine grünen Labels gefunden: {label_path_green}")

    red_boxes = []
    if os.path.exists(label_path_red):
        with open(label_path_red, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x_center, y_center, width, height = map(float, parts[:5])
                    x = int((x_center - width / 2) * img.shape[1])
                    y = int((y_center - height / 2) * img.shape[0])
                    w = int(width * img.shape[1])
                    h = int(height * img.shape[0])
                    red_boxes.append((x, y, w, h))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        print(f"Keine roten Labels gefunden: {label_path_red}")

    matched_pairs = []
    for gx, gy, gw, gh in green_boxes:
        for rx, ry, rw, rh in red_boxes:
            iou = calculate_iou((gx, gy, gw, gh), (rx, ry, rw, rh))
            if iou > 0.5:
                matched_pairs.append({
                    "filename": filename,
                    "green_box": (gx, gy, gw, gh),
                    "red_box": (rx, ry, rw, rh),
                    "iou": iou
                })
                break
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    for rx, ry, rw, rh in red_boxes:
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    csv_path = os.path.join(output_dir, "matched_boxes.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "green_box", "red_box", "iou"])
        if write_header:
            writer.writeheader()
        for match in matched_pairs:
            writer.writerow(match)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)