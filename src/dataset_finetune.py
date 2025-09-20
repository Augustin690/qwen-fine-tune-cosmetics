from qwen2_vl_2b_instruct_0shot import get_annotation_labels
import os
from datasets import Dataset
import json
import copy
from pathlib import Path

# Define base directories relative to the script location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "clean_dataset")

# Update paths
annotations_path = os.path.join(DATA_DIR, "annotations", "clean_annotations.csv")
product_csv_path = os.path.join(BASE_DIR, "csv_data", "skincare_products_clean.csv")
image_dir = os.path.join(DATA_DIR, "images")

# Update save paths
train_output = os.path.join(DATA_DIR, "train_dataset_cat4.json")
test_output = os.path.join(DATA_DIR, "test_dataset_cat4.json")

# Load annotations
annotations_df_full =  get_annotation_labels(annotations_path, product_csv_path)
#print(annotations_df_full.head())
# Load data once
classes = sorted(annotations_df_full["product_type"].unique().tolist(), key=len, reverse=True)
class_str = ', '.join(classes)

# Function to extract image_id from filename

def extract_image_id(file_name):
    return "_".join(file_name.split("_")[-2:]).replace(".jpg", "")


# Create mapping between image_id and file paths
image_files = os.listdir(image_dir)
image_id_to_path = {extract_image_id(file): os.path.join(image_dir, file) for file in image_files}

annotations_df_full["image_path"] = annotations_df_full["image_id"].map(image_id_to_path)
#print(annotations_df_full.head())
# Filter out missing images
annotations_df_full = annotations_df_full.dropna(subset=["image_path"])

#print(annotations_df_full)

annotations_df_full = annotations_df_full[["product_type", "image_path"]]

dataset = Dataset.from_pandas(annotations_df_full)
# Split dataset
dataset = dataset.train_test_split(test_size=0.2, seed=0)

template_json = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": ""#filename
                    },
                    {"type": "text", "text": "What is the category of the skincare product in the image?"
                     #"'Product(category: write here the product category )' and be concise."
                     } # product type
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "" } # product type
                ]
            }
        ]
    }


print(len(dataset['train']))

def create_json(split:str):
    split_json = []
    for idx in range(len(dataset[split])):
        row = dataset[split][idx]
        # Create a fresh copy of the template JSON for each row
        row_json = copy.deepcopy(template_json)

        row_json['messages'][0]['content'][0]['image'] = row['image_path']
        row_json['messages'][1]['content'][0]['text'] = f"{row['product_type']}"
        split_json.append(row_json)

    return split_json


# save tp json
train_json,test_json = json.dumps(create_json('train'), indent=4), json.dumps(create_json('test'), indent=4)

# save to disk
with open(train_output, "w") as f:
    f.write(train_json)

with open(test_output, "w") as f:
    f.write(test_json)