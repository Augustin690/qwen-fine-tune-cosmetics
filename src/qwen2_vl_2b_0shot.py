"""Script to test Qwen2-VL-2B before fine-tuning on skincare product image classification task."""

import random
from transformers import pipeline
# Load model directly
from transformers import AutoProcessor, AutoTokenizer,  AutoModelForImageTextToText, Qwen2VLProcessor, Qwen2VLForConditionalGeneration
import os
from dotenv import load_dotenv
import torch
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from torchvision import transforms
# Enable mixed precision
from torch.amp import autocast
import numpy as np
from tqdm import tqdm
from qwen2_vl_2b_instruct_0shot import get_annotation_labels


if __name__ == '__main__':
#if False:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    load_dotenv("../credentials.env")
    wandb.login(key= os.getenv("WANDB_API_KEY"))
    QWEN_PATH = "local_qwen2_vl_2b"

    image_classifier = pipeline("image-text-to-text", model=QWEN_PATH,
                     torch_dtype=torch.bfloat16, device="mps")
    # Load annotations
    annotations_path = "../clean_dataset/annotations/clean_annotations.csv"
    product_csv_path = "../csv_data/skincare_products_clean.csv"
    annotations_df =  get_annotation_labels(annotations_path, product_csv_path)
    classes = annotations_df["product_type"].unique().tolist()
    #classes.append("Unknown")
    # Sort classes by length in descending order to prioritize specific matches
    sorted_classes = sorted(classes, key=len, reverse=True)
    class_prompt = "What type of skincare product is shown in this image? Choose one of the following: " + ", ".join(classes) + ("."
                                                                                                                                 "Only output one word, which has to be one of the classes mentioned.")

    def extract_image_id(file_name):
        return "_".join(file_name.split("_")[-2:]).replace(".jpg", "")

    # Create mapping between image_id and file paths
    image_dir = "../clean_dataset/images/"
    image_files = os.listdir(image_dir)
    image_id_to_path = {extract_image_id(file): os.path.join(image_dir, file) for file in image_files}
    annotations_df["image_path"] = annotations_df["image_id"].map(image_id_to_path)
    annotations_df = annotations_df.dropna(subset=["image_path"])     # Filter out missing images
    target_size = (448, 448)
    inference_examples = []  # To store the 10 random examples
    true_labels = []
    predicted_labels = []

    annotations_df = annotations_df.sample(frac=0.2, random_state=seed).reset_index(drop=True) # sampling 20% of dataset
    # Shuffle the dataset and select 10 random examples for inspection
    random.seed(seed)  # For reproducibility
    random_indices = random.sample(range(len(annotations_df)), 10)
    random_examples = annotations_df.iloc[random_indices]

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # Store one batch as an inference example
    inference_example = {
        "image_paths": [],
        "images": [],
        "true_labels": [],
        "generated_texts": [],
        "predicted_labels": []
    }

    # Initialize wandb
    wandb.init(project="Qwen2-VL-2B-Evaluation",entity="Quantaest_Itw", name= f"zeroshot_resize_{target_size}_base_model",config={
        "model": "Qwen2-VL-2B",
        "task": "Image-Text Classification",
        "dataset": "Skincare Products",
        "dataset_sample": "0.2",
        "other": "not instruct, changed chat_template.json"
    })
    # Perform inference on each image
    #conversation_batch = [conversation for _ in range(batch_size)]
    print(f"Images to be processed: {len(annotations_df)}")
    for idx in tqdm(range(0, len(annotations_df)), desc="Processing images"):
        # no batching
        image_path = annotations_df.loc[idx, "image_path"]
        true_label = annotations_df.loc[idx, "product_type"]
        image_idx = Image.open(image_path).resize(target_size, Image.Resampling.LANCZOS)
        true_labels.append(true_label)

        message = [
            {
                "role":"user",
                "content":[
                    {
                        "type": "image", "image": image_idx
                    },
                    {
                        "type":"text",
                        "text": class_prompt
                    }
                ]
            }
        ]
        response = image_classifier(text = message, max_new_tokens=10, return_full_text=False)

        prediction = response[0]['generated_text']
        predicted_label = next((cls for cls in sorted_classes if cls.lower() in prediction.lower()), "Moisturiser")
        predicted_labels.append(predicted_label)

        # Save the first batch as an inference example
        if idx in random_indices :
            inference_example["image_paths"] = image_path
            inference_example["images"] = image_idx
            inference_example["true_labels"] = true_label
            inference_example["generated_texts"] = prediction
            inference_example["predicted_labels"] = predicted_label

            # Create a wandb.Table to log the inference example
            inference_table = wandb.Table(columns=["Image", "True Label", "Generated Text", "Predicted Label"])

            image = inference_example["images"]
            # Add row to the table
            inference_table.add_data(
                wandb.Image(image),
                inference_example["true_labels"],
                inference_example["generated_texts"],
                inference_example["predicted_labels"]
            )

            # Log the table to wandb
            wandb.log({"Inference Example": inference_table})

    print("number of true labels", len(true_labels), "number of predicted labels", len(predicted_labels))
    print("number of predicted labels", len(predicted_labels))
    # Now you have all_predictions and all_true_labels for evaluation
    print("Predictions:", predicted_labels)
    print("True Labels:", true_labels)
    # Ensure true labels and predictions are in the same format
    # For example, if predictions are strings, convert true labels to strings
    true_labels = [str(label) for label in true_labels]
    predicted_labels = [str(pred) for pred in predicted_labels]

    # Compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" , xticklabels=sorted_classes, yticklabels=sorted_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the confusion matrix plot locally
    plt.savefig("../confusion_matrix.png")
    print("Confusion matrix saved to 'confusion_matrix.png'")
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        y_true=true_labels,
        preds=predicted_labels,
        class_names=sorted_classes,
    )})


    # Finish wandb run
    wandb.finish()
