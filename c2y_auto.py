import os
import zipfile
import json

from pathlib import Path

# === CONFIGURATION ===
input_dir = "C:/Users/LAMBDA THETA/Downloads/genba_kp"  # Folder containing ZIP files
output_root = "C:/Users/LAMBDA THETA/Downloads/genba_kp"  # Where ZIPs will be extracted

# === FUNCTION TO NORMALIZE KEYPOINTS ===
def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] / image_width
        y = keypoints[i + 1] / image_height
        visibility = keypoints[i + 2]  # If needed later
        normalized_keypoints.extend([x, y])
    return normalized_keypoints

# === MAIN PROCESSING LOOP ===
for zip_file in os.listdir(input_dir):
    if not zip_file.endswith(".zip"):
        continue

    zip_path = os.path.join(input_dir, zip_file)
    unzip_folder_name = os.path.splitext(zip_file)[0]
    unzip_folder_path = os.path.join(output_root, unzip_folder_name)

    # 1. UNZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder_path)
    print(f"Extracted: {zip_file}")

    # 2. FIND ANNOTATION FILE
    annotations_folder = os.path.join(unzip_folder_path, "annotations")
    if not os.path.exists(annotations_folder):
        print(f"No annotations folder in {zip_file}, skipping.")
        continue

    json_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON annotation found in {annotations_folder}, skipping.")
        continue

    annotation_path = os.path.join(annotations_folder, json_files[0])
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # 3. SETUP LABEL OUTPUT FOLDER
    labels_dir = os.path.join(unzip_folder_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # 4. CONVERT COCO TO YOLO
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_name = os.path.splitext(image_info['file_name'])[0]

        yolo_data = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id and 'keypoints' in annotation:
                class_id = annotation['category_id']  # Optional: remap if needed
                keypoints = annotation['keypoints']
                normalized_kpts = normalize_keypoints(keypoints, image_width, image_height)
                yolo_data.append(f"{class_id} " + " ".join(map(str, normalized_kpts)))

        if yolo_data:
            label_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_data))

        else:
            label_path = os.path.join(labels_dir, f"{image_name}.txt")
            open(label_path, 'w').close()

    print(f"Conversion complete for: {zip_file}")

print("All ZIPs processed.")
