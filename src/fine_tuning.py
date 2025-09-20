import torch
import json
import datetime
import os
import wandb
import random
import numpy as np
import re
from itertools import islice
from pathlib import Path

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from vision_util import process_vision_info
from logutil import init_logger, get_logger
from torch.amp import autocast, GradScaler
from tqdm import tqdm  # Import tqdm for progress visualization
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define base directories relative to the script location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "clean_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "src", "local_qwen2_vl_2b")
OUTPUT_DIR = os.path.join(BASE_DIR, "train_output")

# Global logger
logger = None

def initialize_global_logger(log_dir):
    global logger
    if logger is None:
        init_logger(log_dir)
        logger = get_logger()
    return logger

class PathConfig:
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "clean_dataset")
        self.model_dir = os.path.join(base_dir, "src", "local_qwen2_vl_2b")
        self.output_dir = os.path.join(base_dir, "train_output")
        
        # Dataset paths
        self.train_dataset = os.path.join(self.data_dir, "train_dataset_cat4.json")
        self.test_dataset = os.path.join(self.data_dir, "test_dataset_cat4.json")
        
        # Create output directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.run_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

class SkinCareDataSet(Dataset): # for toy demo
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'skincare product category: "moisturizer"'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'skincare product category: "moisturizer"<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device, inference:bool=False):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]
    messages = [m['messages'] for m in batch]

    def extract_output_text(messages):
        for message in messages:
            if message["role"] == "assistant":
                for content in message["content"]:
                    if content["type"] == "text":
                        return content["text"]  # Return the structured output
        return None  # Handle cases where no output is found
    desired_outputs = [extract_output_text(m) for m in messages]
    # include only users' messages
    #if msg["role"] == "user"
    texts_user = None
    if inference == True:
        texts_user = [processor.apply_chat_template([msg for msg in m if msg["role"] == "user"], tokenize=False, add_generation_prompt=False) for m in messages] 

    texts = [processor.apply_chat_template([msg for msg in m ], tokenize=False, add_generation_prompt=False) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages, inference)
    #print("textes",texts)
    #print("user texts",texts_user)
   # print("image inputs",image_inputs)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_user = None
    if inference == True:
        #print("infer")
        input_user = processor(
            text=texts_user,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

   # Check for NaN or inf in tensor components of inputs
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):  # Only check tensor values
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise ValueError(f"Inputs contain NaN or inf values in key: {key}")

    inputs = inputs.to(device)
    if input_user is not None:
        input_user = input_user.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    #print("input_ids_lists",input_ids_lists)
    for ids_list in input_ids_lists:
        # -100 causes problems
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)
    #print("labels_list",labels_list)
    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids, input_user, desired_outputs


def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved. 
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    '''
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")

def parse_category(output_str, categories=["Moisturiser", "Exfoliator", "Bath Salts", "Body Wash", "Eye Care", 
                                           "Cleanser", "Bath Oil", "Serum", "Toner", "Mist", "Balm", "Mask", "Peel", "Oil", "Unknown"]):
    """Extract category value from model output string"""
      # Split the output string by newlines
        # Extract assistant message after "assistant\n"
    # Regex to capture text after "assistant\n" until the next blank line
    lines = [line.strip() for line in output_str.lower().split('\n') if line.strip()]
    #match = re.search(r"assistant\n(.*?)(?=\n\n|$)", output_str[2], re.DOTALL & re.IGNORECASE)
    #print(lines)
    if len(lines) >= 3:
        assistant_message = lines[4:]
        # Extract the category value from the assistant message
        for category in categories:
            for line in assistant_message:
                #print("line",line)
                if category.lower() in line:
                    return category       
        return "Unknown"
    return "Unknown"  # Fallback if no valid category is found

def plot_confusion_matrix(y_true, y_pred, categories, save_path):
    """
    Plot and save confusion matrix
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # Create figure and axes
    plt.figure(figsize=(12, 8))
    
    # Plot confusion matrix using seaborn
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def validate(model, val_loader, processor_val, save_dir=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    categories = ["Moisturiser", "Exfoliator", "Bath Salts", "Body Wash", "Eye Care", 
                 "Cleanser", "Bath Oil", "Serum", "Toner", "Mist", "Balm", "Mask", 
                 "Peel", "Oil", "Unknown"]

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
            inputs, labels, inputs_users, desired_outputs = batch

            for key, value in inputs_users.items():
                if isinstance(value, torch.Tensor):
                    assert not torch.any(torch.isnan(value)), f"NaN in {key}"
                    assert not torch.any(torch.isinf(value)), f"Inf in {key}"
            #decoded_text = tokenizer.decode(inputs_users["input_ids"][0])
            #print("decoded text", decoded_text)
            # Compute loss
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()

            generated_ids = model.generate(**inputs_users,
                num_beams=1,  # Use greedy decoding
                max_new_tokens=30,
                pad_token_id=processor_val.tokenizer.pad_token_id
            )
            
            #print("GENERATED IDS",generated_ids)
            if torch.any(generated_ids < 0):
                print("Warning: Negative token IDs detected. Clipping to zero.")
                generated_ids = torch.clamp(generated_ids, min=0)
            preds = processor_val.batch_decode(generated_ids, skip_special_tokens=True)
            
            #print("geenerated text", preds)
            true_texts = desired_outputs
            #print("true texts", true_texts)
            # Extract categories
            batch_pred_cats = [parse_category(p) for p in preds]
            batch_true_cats = [t for t in true_texts]

            if i % 10 == 0:  # Pick a random batch every 10 iterations
                logger.info("generated text: %s", preds)
                logger.info("true texts: %s", true_texts)
                logger.info("batch_pred_cats: %s", batch_pred_cats)
                logger.info("batch_true_cats: %s", batch_true_cats)
                # log batch accuracy
                accuracy = accuracy_score(batch_true_cats, batch_pred_cats)
                logger.info(f"Batch {i} accuracy: {accuracy}")

            all_preds.extend(batch_pred_cats)
            all_labels.extend(batch_true_cats)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Plot confusion matrix if save_dir is provided
    if save_dir:
        plot_confusion_matrix(all_labels, all_preds, categories, save_dir)
        logger.info(f"Confusion matrix saved to {save_dir}")

    return {
        'val_loss': total_loss / len(val_loader),
        'val_accuracy': accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }

class TrainingConfig:
    def __init__(self,
                 batch_size=1,
                 learning_rate=5e-7,
                 num_epochs=20,
                 accumulation_steps=4,
                 gradient_norm=0.5,
                 num_frozen_layers=26,
                 warmup_steps=100,
                 patience=3,
                 reduction_factor=4,
                 weight_decay=0.01,
                 scheduler_t0_div=3,
                 scheduler_t_mult=2,
                 scheduler_min_lr=1e-7):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.gradient_norm = gradient_norm
        self.num_frozen_layers = num_frozen_layers
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.reduction_factor = reduction_factor
        self.weight_decay = weight_decay
        self.scheduler_t0_div = scheduler_t0_div
        self.scheduler_t_mult = scheduler_t_mult
        self.scheduler_min_lr = scheduler_min_lr
        
        # Add categories for classification
        self.categories = ["Moisturiser", "Exfoliator", "Bath Salts", "Body Wash", "Eye Care", 
                          "Cleanser", "Bath Oil", "Serum", "Toner", "Mist", "Balm", "Mask", 
                          "Peel", "Oil", "Unknown"]

def setup_model(config, qwen_path="local_qwen2_vl_2b"):
    global logger
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    
    # Freeze layers
    layer_names = list(model.named_parameters())
    for name, param in layer_names:
        # Freeze all visual layers except the last block and merger
        if 'visual' in name:
            if 'merger' in name:
                param.requires_grad = True
                logger.info(f"Unfrozen merger layer: {name}")
            elif 'visual.blocks.31' in name:  # Last visual block
                param.requires_grad = True
                logger.info(f"Unfrozen visual layer: {name}")
            else:
                param.requires_grad = False
        
        # Freeze language model layers except last one
        if 'layer' in name and int(name.split('.')[2]) <= config.num_frozen_layers:
            param.requires_grad = False
        
        logger.info(f"{name} - {param.requires_grad}")
    
    model.float()
    return model

def setup_processor(processor_path="local_qwen2_vl_2b", min_pixels = 64*28*28,
                     max_pixels=128*28*28, padding_side="right"):
    return AutoProcessor.from_pretrained(processor_path, min_pixels=min_pixels, max_pixels=max_pixels, 
                                         padding_side=padding_side)

def setup_optimizer(model, config):
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        #correct_bias=True
    )
    return optimizer

def setup_scheduler(optimizer, total_steps, config):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // config.scheduler_t0_div,
        T_mult=config.scheduler_t_mult,
        eta_min=config.scheduler_min_lr
    )

def setup_data_loaders(config, processor, processor_val, paths):
    # Training data setup
    train_dataset = SkinCareDataSet(paths.train_dataset)
    train_dataset = torch.utils.data.Subset(
        train_dataset, 
        random.sample(range(len(train_dataset)), 
        len(train_dataset)//config.reduction_factor)
    )
    
    # Test data setup
    test_dataset = SkinCareDataSet(paths.test_dataset)
    test_dataset = torch.utils.data.Subset(
        test_dataset, 
        random.sample(range(len(test_dataset)), 
        len(test_dataset)//config.reduction_factor*2)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor, device=torch.device("cuda")),
        pin_memory=False
    )
    
    val_loader = DataLoader(
        test_dataset,  # Use the reduced test_dataset
        batch_size=config.batch_size,
        collate_fn=partial(collate_fn, processor=processor_val, device=torch.device("cuda"), inference=True),
        pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset

def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch):
    model.train()
    accumulated_avg_loss = 0
    steps = 0
    
    # Add classification loss
    #classification_criterion = torch.nn.CrossEntropyLoss()
    
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", unit="batch") as t_loader:
        for batch_idx, batch in enumerate(t_loader):
            steps += 1
            inputs, labels, _, _ = batch

            if batch_idx % config.accumulation_steps == 0:
                optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels)
                
                # Get logits from the model output
                logits = outputs.logits[:, -1, :]  # Get logits for the last token
                
                # Calculate both losses
                generation_loss = outputs.loss
                
                # Convert text labels to class indices
                #class_indices = torch.tensor([config.categories.index(label) for label in batch[3]], 
                #                          device=logits.device)
                
                # Calculate classification loss
                #lassification_loss = classification_criterion(logits, class_indices)
                
                # Combine losses
                loss = generation_loss / config.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected - skipping batch")
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                wandb.log({
                    "batch_loss": loss.item() * config.accumulation_steps,
                    #"generation_loss": generation_loss.item(),
                    #"classification_loss": classification_loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": steps,
                    "epoch": epoch + 1
                })

    return steps

def train(seed=0):
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize configs
    paths = PathConfig()
    config = TrainingConfig()
    
    # Initialize global logger
    global logger
    logger = initialize_global_logger(paths.run_dir)
    
    # Setup model and processors
    model = setup_model(config, paths.model_dir)
    processor = setup_processor(paths.model_dir)
    processor_val = setup_processor(paths.model_dir, padding_side="left")
    
    # Setup data loaders with paths
    train_loader, val_loader, train_dataset = setup_data_loaders(
        config, processor, processor_val, paths
    )
    
    # Setup training components
    optimizer = setup_optimizer(model, config)
    total_steps = len(train_dataset) * config.num_epochs // (config.batch_size * config.accumulation_steps)
    scheduler = setup_scheduler(optimizer, total_steps, config)
    scaler = GradScaler()
    
    # Initialize wandb
    wandb.init(
        project="Qwen2-VL-2B-Evaluation",
        entity="Quantaest_Itw",
        name="fine_tuning_base_freeze26_valmetrics",
        config={
            "model": "Qwen2-VL-2B",
            "task": "Image-Text Classification",
            "dataset": f"Skincare Products, reduced by {config.reduction_factor}",
            "epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "accumulation_steps": config.accumulation_steps,
            "frozen_visual_blocks": 30,
            "frozen_layers": config.num_frozen_layers,
            "GradScaler": True if scaler is not None else False,
            "clip_grad_norm": config.gradient_norm,
        }
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        wandb.log({"epoch": epoch + 1})
        
        # Validation before training
        if epoch == 0:
            val_metrics = validate(model, val_loader, processor_val, save_dir=paths.run_dir + f'/cm_before_training.png')
        wandb.log({**val_metrics, "epoch": epoch})
        logger.info(f"Epoch {epoch} Validation Metrics: {val_metrics}")
        
        # Train epoch
        steps = train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch)
        
        # Validation and early stopping
        val_metrics = validate(model, val_loader, processor_val, save_dir=paths.run_dir + f'/cm_epoch_{epoch}.png')
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            model.save_pretrained(os.path.join(paths.run_dir, "best_model"))
            write_chat_template(processor, paths.run_dir) 
            logger.info(f"New best model saved at {paths.run_dir}/best_model")
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Cleanup and save
    wandb.finish()
    os.makedirs(paths.run_dir, exist_ok=True)
    model.bfloat16()
    model.save_pretrained(os.path.join(paths.run_dir, "best_model"))
    processor.save_pretrained(paths.run_dir)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train()
