"""Script to test Qwen2-VL-2B-instruct before fine-tuning on skincare product image classification task."""
import random
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


def load_qwen_from_hf(env_path :str, model_path:str, save_path:str) -> None:

    load_dotenv(env_path)
    auto_processor = AutoProcessor.from_pretrained(model_path,
                                              token= os.getenv('HF_TOKEN'))
    auto_model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                        token= os.getenv('HF_TOKEN'))

    # Save to disk
    auto_model.save_pretrained(save_path)
    auto_processor.save_pretrained(save_path)

    print(f"Model loaded to disk in folder: {save_path}")

def load_qwen_from_disk(model_path:str, min_pixels:int=500, max_pixels:int=500)-> tuple[AutoProcessor, Qwen2VLForConditionalGeneration]:

    return AutoProcessor.from_pretrained(model_path), Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

def get_annotation_labels(ann_path:str, product_path:str) -> pd.DataFrame:

    ann_df = pd.read_csv(ann_path, dtype={"product_url": str})

    # product type will be our label here
    products_df = pd.read_csv(product_path, usecols=["product_type", "product_url"],
                              dtype={"product_url": str})

    products_df.drop_duplicates(subset=["product_url"], keep="first", inplace=True)

    return pd.merge(products_df, ann_df, on="product_url", how="inner")


# Predict function
def predict(image, prompt:str, processor: Qwen2VLProcessor, model:Qwen2VLForConditionalGeneration):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    #qwen2b_path = "Qwen/Qwen2-VL-2B-instruct"
    #env_path = "../credentials.env"
    #load_qwen_from_hf(env_path, qwen2b_path,  "local_qwen2_vl_2b-instruct")
    seed = 42

    # Check if MPS is available
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    load_dotenv("../credentials.env")
    wandb.login(key= os.getenv("WANDB_API_KEY"))

    QWEN_PATH = "local_qwen2_vl_2b-instruct"
    # Load the processor and model from disk
    processor, model = load_qwen_from_disk(QWEN_PATH)
    # Enable gradient checkpointing (if available)
    #model.gradient_checkpointing_enable()

    # Move the model to the device (MPS or CPU)
    model.to(device)

    # Load annotations
    annotations_path = "../clean_dataset/annotations/clean_annotations.csv"
    product_csv_path = "../csv_data/skincare_products_clean.csv"
    annotations_df = get_annotation_labels(annotations_path, product_csv_path)
    classes = annotations_df["product_type"].unique().tolist()
    classes.append("Unknown")
    # Sort classes by length in descending order to prioritize specific matches
    sorted_classes = sorted(classes, key=len, reverse=True)
    class_prompt = "What type of skincare product is shown in this image? Choose one of the following: " + ", ".join(classes) + ("."
                                                                                                                                 "Only output one word, which has to be one of the classes mentioned.")

    # Extract image_id from file names
    c = 0
    def extract_image_id(file_name):
        return "_".join(file_name.split("_")[-2:]).replace(".jpg", "")

    # Create mapping between image_id and file paths
    image_dir = "../clean_dataset/images/"
    image_files = os.listdir(image_dir)
    image_id_to_path = {extract_image_id(file): os.path.join(image_dir, file) for file in image_files}
    # Merge annotations with image paths
    annotations_df["image_path"] = annotations_df["image_id"].map(image_id_to_path)
    # Filter out missing images
    annotations_df = annotations_df.dropna(subset=["image_path"])
    target_size = (448, 448)  # You can adjust the resolution as needed
    conversation = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                    },
                    {
                        "type":"text",
                        "text":f"{class_prompt}."
                    }
                ]
            }
        ]

    # Preprocess the inputs
    #text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    #print("text promppt",text_prompt)


    inference_examples = []  # To store the 10 random examples
    true_labels = []
    predicted_labels = []

    annotations_df = annotations_df.sample(frac=0.2, random_state=seed).reset_index(drop=True) # sampling 20% of dataset
    # Shuffle the dataset and select 10 random examples for inspection
    random.seed(seed)  # For reproducibility
    random_indices = random.sample(range(len(annotations_df)), 10)
    random_examples = annotations_df.iloc[random_indices]

    batch_size = 4  # Adjust based on memory constraints

    conversation_batch = [conversation for _ in range(batch_size)]
    # Preparation for batch inference
    #text_prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversation_batch]

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
    wandb.init(project="Qwen2-VL-2B-Evaluation",entity="Quantaest_Itw", name= f"zeroshot_resize_{target_size}_batch",config={
        "model": "Qwen2-VL-2B",
        "task": "Image-Text Classification",
        "dataset": "Skincare Products",
        "batch_size": f"{batch_size}",
        "dataset_sample": "0.2",
        "other": "explicit nograd, bf16 passed for inputs, batched prompts as well"
    })
    # Perform inference on each image
    print(f"Images to be processed: {len(annotations_df)}")
    for idx in tqdm(range(0, len(annotations_df), batch_size), desc="Processing batches"):
        #image_path = annotations_df.loc[idx, "image_path"]
        #true_label = annotations_df.loc[idx, "product_type"]
        # forgot lanczosss
        #image = Image.open(image_path).resize(target_size) #Image.Resampling.LANCZOS)
        batch_indices = range(idx, min(idx + batch_size, len(annotations_df)))
        # Load and preprocess images
        images_batch = [preprocess(Image.open(annotations_df.loc[idx, "image_path"])) for idx in batch_indices]
        image_paths_batch = [annotations_df.loc[idx, "image_path"] for idx in batch_indices]

        print(f"Number of images: {len(images_batch)}")
        #print(f"Number of text prompts: {len(text_prompts)}")
        if len(images_batch) < batch_size:
            print("Padding the last batch to match batch_size...")
            while len(images_batch) < batch_size:
                images_batch.append(images_batch[-1])  # Duplicate the last image

        # Get true labels for the batch
        true_labels_batch = [annotations_df.loc[idx, "product_type"] for idx in batch_indices]
        true_labels.extend(true_labels_batch)

        # Preprocess images and text
        try:
            processor_ = AutoProcessor.from_pretrained(QWEN_PATH)
            text_prompts = [processor_.apply_chat_template(msg, add_generation_prompt=True) for msg in conversation_batch]
            inputs = processor_(
                images=images_batch,
                text=text_prompts,
                padding=True,
                return_tensors="pt",
                #padding="max_length",  # Force token padding
                do_rescale=False,
            )
        except IndexError as e:
            print(f"IndexError during processor call: {e}")
            print(f"Batch indices: {list(batch_indices)}")
            print(f"Images batch size: {len(images_batch)}")
            continue
        # Preprocess image and text (for batch inference)
        #inputs = processor(images=image, text=text_prompt, padding= True, return_tensors="pt")
        #print(f"Tokenized input IDs: {inputs['input_ids']}")
       #print(f"Decoded tokens: {processor.tokenizer.decode(inputs['input_ids'][0])}")
        #print(f"Image token ID: {processor.tokenizer.convert_tokens_to_ids('<image>')}")

        # Move inputs to the correct device (MPS or CPU)
        inputs = inputs.to(torch.bfloat16).to(device)

        # Perform the inference
        #with autocast(device_type="mps"):
        # Perform inference
        #with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(**inputs,max_new_tokens=128)

        # Decode the generated text and determine the predicted label
        print("generating prediction")
        #generated_text = processor.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
        #print(generated_text)
        # Map the generated text to the closest class
        #predicted_label = next((cls for cls in sorted_classes if cls.lower() in generated_text.lower()), "Unknown")
        #predicted_labels.append(predicted_label)
        #true_labels.append(true_label)
        # Decode predictions (assuming outputs are token IDs)
        predictions_batch = processor_.batch_decode(outputs, skip_special_tokens=True)
        predicted_labels_batch = [
        next((cls for cls in sorted_classes if cls.lower() in generated_text.split("assistant")[-1].strip().lower()), "Unknown")
        for generated_text in predictions_batch
    ]
        predicted_labels.extend(predicted_labels_batch)


       # print(f"Accuracy:{c/(idx+1)}")

        # Save the first batch as an inference example
        if idx == random.randint(0, (len(annotations_df) -1)//batch_size) :
            inference_example["image_paths"] = image_paths_batch
            inference_example["images"] = images_batch
            inference_example["true_labels"] = true_labels_batch
            inference_example["generated_texts"] = predictions_batch
            inference_example["predicted_labels"] = predicted_labels_batch

            # Create a wandb.Table to log the inference example
            inference_table = wandb.Table(columns=["Image", "True Label", "Generated Text", "Predicted Label"])

            for i in range(len(inference_example["images"])):
                # Convert tensor image to PIL for visualization
                image = transforms.ToPILImage()(inference_example["images"][i])
                # Add row to the table
                inference_table.add_data(
                    wandb.Image(image),
                    inference_example["true_labels"][i],
                    inference_example["generated_texts"][i],
                    inference_example["predicted_labels"][i]
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
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        y_true=true_labels,
        preds=predicted_labels,
        class_names=sorted_classes,
    )})

    #Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") #, xticklabels=sorted_classes, yticklabels=sorted_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the confusion matrix plot locally
    plt.savefig("../confusion_matrix.png")
    print("Confusion matrix saved to 'confusion_matrix.png'")
    # Finish wandb run
    wandb.finish()

