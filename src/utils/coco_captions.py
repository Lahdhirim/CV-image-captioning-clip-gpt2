import json
from pathlib import Path

# [HIGH]: if split is val, split it into val and test sets
split = "val"

coco_annotation_path = f"data/captions_{split}2014.json"
image_dir = f"data/{split}2014"

with open(coco_annotation_path, "r") as f:
    coco_data = json.load(f)

image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

output = []
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    filename = image_id_to_filename[image_id]
    output.append({"image": str(Path(image_dir) / filename), "caption": caption})

with open(f"data/coco_{split}_captions_processed.json", "w") as f:
    json.dump(output, f, indent=2)

print("Done, saved processed captions json.")
