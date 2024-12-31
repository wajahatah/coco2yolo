import json
import os

# Paths to COCO annotation JSON file and the directory to save YOLO labels
coco_annotation_path = "C:/Users/LAMBDA THETA/Downloads/a57/annotations/person_keypoints_default.json"
yolo_labels_dir = "C:/Users/LAMBDA THETA/Downloads/a57/annotations/yolo"
os.makedirs(yolo_labels_dir, exist_ok=True)

# Function to normalize keypoints
def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] / image_width
        y = keypoints[i + 1] / image_height
        visibility = keypoints[i + 2]
        
        # Append normalized x and y, ignoring visibility
        normalized_keypoints.extend([x, y])
    return normalized_keypoints

# Load COCO annotations
with open(coco_annotation_path, 'r') as f:
    coco_data = json.load(f)

# Iterate over each image in the COCO dataset
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']

    # Collect all annotations for this image
    yolo_data = []
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id and 'keypoints' in annotation:
            class_id = annotation['category_id']  # Class ID (you may want to map it to YOLO format)
            keypoints = annotation['keypoints']
            
            # Normalize keypoints
            normalized_keypoints = normalize_keypoints(keypoints, image_width, image_height)
            yolo_data.append(f"{class_id} " + " ".join(map(str, normalized_keypoints)))

    # Save YOLO format annotation
    label_filename = os.path.join(yolo_labels_dir, f"{image_info['file_name'].split('.')[0]}.txt")
    with open(label_filename, 'w') as f:
        f.write("\n".join(yolo_data))

print("Conversion complete. YOLO labels saved.")
