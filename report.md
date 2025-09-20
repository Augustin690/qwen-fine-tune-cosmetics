# Fine-tuning Qwen2-VL-2B for Skincare Product Recognition: Summary Report

## Task Description

The objective is to fine-tune the Qwen2-VL-2B model to accurately classify skincare products into their respective categories based on product images. This is formulated as a vision-language task where the model:

1. **Input**:
   - An image of a skincare product
   - A text prompt: "What type of skincare product is shown in this image?"

2. **Output**:
   - One of 14 predefined categories: Moisturiser, Exfoliator, Bath Salts, Body Wash, Eye Care, Cleanser, Bath Oil, Serum, Toner, Mist, Balm, Mask, Peel, Oil

This is a first step towards a more general image parsing task, where the model would be able to parse a prodcut and return a structured output such as the product name, the brand, the price, the category, etc.:

```python
Product(
    brand: "BrandX",
    category: "Cleanser",
    country: "USA",
    details: "Gentle night cream cleanser",
    howToUse: "Apply to face and rinse off",
    ingredients: "Water, Glycerin, etc.",
            manufacturer: "ManufacturerX",
            name: "Cream Cleanser",
            origin: "USA",
            size: "200ml",
            forNightOrMorning: "Night"
        )
``` 

3. **Dataset Structure**:
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "path/to/product/image.jpg"
                },
                {
                    "type": "text",
                    "text": "What type of skincare product is shown in this image?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Moisturiser"  // Example category
                }
            ]
        }
    ]
}
```

This format leverages Qwen2-VL-2B's multimodal capabilities while maintaining its conversation-style interface. The model must:
- Process and understand visual features from product images
- Interpret the question in context
- Map the combined understanding to the correct product category

The challenge lies in distinguishing between visually similar products (e.g., Serum vs Oil) and handling variations in product presentation (different angles, lighting conditions, packaging styles).

## 1. Data Processing Strategies and Steps

### 1.1 Data Collection
The data collection process was implemented through a custom web crawler specifically designed for LookFantastic's website, targeting high-quality skincare product images.

#### Web Inspection and Image Location
Initial analysis of the website's structure revealed that high-resolution product images were stored in a JavaScript constant within a specific script tag. The images were located using the XPath:
```javascript
[@id="view-item-container"]/div[1]/div/script/text()
```
This location consistently contained an array of image URLs for each product, including multiple angles and detail shots.



#### Crawler Implementation
The crawler was implemented in Python with several key features:

1. **Robust Error Handling**:
   - Implemented retry mechanism with exponential backoff using the `@retry` decorator
   - Maximum 5 retry attempts with delays ranging from 1 to 10 seconds
   - Comprehensive exception handling for network issues and parsing errors

2. **Parallel Processing**:
   - Utilized `ThreadPoolExecutor` for concurrent downloads
   - Configured with 10 worker threads to balance speed and server load
   - Progress tracking using `tqdm` for real-time monitoring

3. **Data Extraction Pipeline**:
   ```python
   def scrape_product_images(url):
       # Fetch webpage with retry mechanism
       response = fetch_url_with_retries(url)
       
       # Parse HTML using BeautifulSoup
       soup = BeautifulSoup(response.text, 'lxml')
       
       # Extract image URLs using regex pattern
       pattern = r'const\s+images\s*=\s*(\[[^\]]*\])'
       # ...
   ```

4. **Organized Data Storage**:
   - Images unique identifiers combining product ID and image number saved in 'csv_data/images_id_full.csv', along with the product URL.
   - Images are saved in the 'product_images' folder, with the filename being the product name (for readability) concatenated with the image ID.


#### Dataset Statistics
- Initial dataset composition:
  - Number of raw images: [3410]
  - Number of products: [990] vs [1126] origninally. We have a coverage of 88% of the products, which I deemed sufficient given the time
        constraints. To go further, we could inspect the log 'log/scraper.log' to see which products were not scraped, why, and eventually 
        retry the scraping process for those.
  - Source: LookFantastic e-commerce platform
  - Image formats: JPG

The crawler implementation includes built-in logging functionality to track the download process and any potential issues, storing detailed logs in a dedicated log file for monitoring and debugging purposes.

### 1.2 Data Cleaning Pipeline

The data cleaning process was implemented with a focus on image quality and dataset integrity, utilizing multiple validation steps:

#### 1. Deduplication Process
- **Method**: Perceptual hashing implementation using PIL
- **Process**:
  ```python
  def get_image_hash(file_path):
      img = Image.open(file_path)
      hash_size = 8
      img = img.convert("L").resize((hash_size+1, hash_size))
      pixels = list(img.getdata())
      diff = [pixels[i*hash_size + j] > pixels[i*hash_size + j + 1]
              for i in range(hash_size) for j in range(hash_size)]
      return hashlib.md5(str(diff).encode()).hexdigest()
  ```
- This approach:
  - Converts images to grayscale
  - Resizes to 9x8 pixels
  - Computes difference hash based on adjacent pixels
  - Generates MD5 hash of the difference pattern
- Duplicates are identified by matching hash values
- Original images are preserved while duplicates are removed

#### 2. Quality Control
Implemented comprehensive quality checks with multiple criteria:

1. **Blur Detection**:
   - Used Laplacian variance method
   - Threshold set to 60 (empirically determined), compromise between quality and amount of images selected:
        Tried with 100, but this removed too many images (more than 1200 versus 700 now). I also checked manually and found that the images with variance below 100 were still good, and noticed a significant drop in quality below 60, even some images might still have been good
        below that threshold..
   - Images with variance below threshold marked as blurry
   ```python
   laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
   ```

2. **Size Validation**:
   - Minimum dimension threshold: 300 pixels
   - Ensures sufficient resolution for model training
   - Removes thumbnail and low-resolution images

3. **Corruption Check**:
   - Validates image file integrity
   - Ensures proper loading in OpenCV
   - Removes unreadable or corrupted files

#### 3. Annotation Processing
- **Format Standardization**:
  - Created a new csv file, 'clean_dataset/annotations/clean_annotations.csv', same as the original one, but only containing the clean images.
  - Copied clean images to 'clean_dataset/images', keeping the original image folder untouhed in case we need to go back to the original dataset and re-run the cleaning process.
  - Used 'src/dataset_fintune.py' to produce the final dataset in JSON format, with the following structure:
    ```json
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "../clean_dataset/images/alpha-h-beauty-sleep-power-peel-50ml_11282237_7.jpg"
                    },
                    {
                        "type": "text",
                        "text": "What type of skincare product is shown in this image? "
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Peel"
                    }
                ]
            }
        ]
    ```
    - Train/test split: 80/20

#### Dataset Statistics After Cleaning
- Initial dataset size: [3410] images
- Duplicates removed: [590] images
- Quality issues detected:
  - Blurry images: [781]
  - Under-sized images: [0]
  - Corrupted files: [0]
- Final clean dataset: [2038] images

The cleaning process is fully automated and logged, with detailed reports generated for each cleaning run, including specific reasons for image rejection and statistics about the cleaning process. You can find the log in the `log/cleaning_report.log` file.

## 2. Fine-tuning Technical Details

### 2.1 Model Architecture and Capabilities

#### Overview
Qwen2-VL-2B is a multimodal large language model that combines vision and language processing capabilities. Its architecture consists of three main components:

1. **Visual Encoder** (31 blocks):
   - Processes image inputs into high-dimensional feature representations
   - Each block contains self-attention and feed-forward layers
   - Progressive feature abstraction from low-level visual features to high-level semantic concepts

2. **Vision-Language Merger**:
   - Bridges visual and textual modalities
   - Aligns visual features with language embeddings
   - Creates a unified representation space

3. **Language Model** (27 layers):
   - Processes text and generates responses
   - Handles both visual context and textual input
   - Enables conversation-style interactions

```ascii
┌─────────────────┐
│   Image Input   │
└────────┬────────┘
         ▼
┌────────────────────┐
│   Visual Encoder   │
│    (31 blocks)     │
│                    │      ┌─────────────┐
│ Block 1  [Frozen]  │      │             │
│     ⋮              │      │    Text     │
│ Block 30 [Frozen]  │      │    Input    │
│ Block 31 [Trained] │      │             │
└────────┬───────────┘      └──────┬──────┘
         │                         │
         ▼                         ▼
    ┌────────────────────────────────┐
    │      Vision-Language Merger    │
    │           [Trained]            │
    └───────────────┬────────────────┘
                    │
                    ▼
    ┌────────────────────────────────┐
    │       Language Model           │
    │        (27 layers)            │
    │                               │
    │  Layers 1-26    [Frozen]      │
    │  Layer 27       [Trained]     │
    └───────────────┬────────────────┘
                    │
                    ▼
         ┌────────────────────┐
         │  Text Generation   │
         └────────────────────┘
```

#### Fine-tuning Strategy Rationale
Our layer freezing strategy is directly informed by the model's architecture:

1. **Visual Encoder**:
   - Froze first 30 blocks to preserve general visual understanding
   - Fine-tuned only the last block (31) for domain-specific features
   - This maintains robust visual processing while allowing adaptation to skincare products

2. **Merger Module**:
   - Kept fully trainable to optimize vision-language alignment
   - Critical for domain-specific associations between visual and textual features
   - Helps bridge the gap between general visual understanding and skincare-specific terminology

3. **Language Model**:
   - Froze first 26 layers to maintain general language capabilities
   - Fine-tuned only the final layer for task-specific outputs
   - Preserves the model's conversation abilities while specializing for product classification

This selective fine-tuning approach:
- Reduces training parameters from 2B to approximately 100M
- Prevents catastrophic forgetting of pre-trained capabilities
- Focuses adaptation on task-relevant components
- Optimizes memory usage and training efficiency

### 2.2 Environment Setup
- Framework: PyTorch with HuggingFace Transformers
- Training optimizations:
  - Mixed precision training (bfloat16)
  - Gradient checkpointing
  - Gradient accumulation
  - Gradient clipping
- Monitoring: Weights & Biases for experiment tracking

Loading the model is done using the following code (from 'src/qwen2_vl_2b_instruct_0shot.py'):
```python
auto_processor = AutoProcessor.from_pretrained(model_path,
                                              token= os.getenv('HF_TOKEN'))
auto_model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                        token= os.getenv('HF_TOKEN'))
```
using 'Qwen/Qwen2-VL-2B' as the model path. I also used 'Qwen/QWENVL-2B-Instruct' when I was testing the model with the 0-shot approach, but i did the fine-tuning with the base model with no instruction tuning. However, I was encountering some issues with the base model when running inference, so I ended up uding the chat_template.json from the instruct model, which I later modified by augmenting the system prompt with classification instructions.

You can load my fine-tuned model from the '' folder, using the same code as above but without need for HuggingFace token.
### 2.3 Model Configuration
- Base model: Qwen2-VL-2B
- Hardware: NVIDIA GeForce RTX 4080 (16GB VRAM)
- Memory optimizations:
  - Gradient checkpointing enabled
  - Batch size optimization
  - Strategic layer freezing

### 2.4 Training Strategy

#### Layer Freezing Approach
```python
# Freeze visual layers except last block and merger
if 'visual' in name:
    if 'merger' in name or 'visual.blocks.31' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Freeze first 26 language model layers
if 'layer' in name and int(name.split('.')[2]) <= 26:
    param.requires_grad = False
```
This approach:
- Preserves pre-trained visual understanding
- Allows fine-tuning of high-level visual features
- Reduces training parameters while maintaining model capacity

#### Hyperparameters
- Learning rate: 5e-7 with cosine annealing warm restarts
- Batch size: 1 (limited by VRAM)
- Gradient accumulation steps: 4
- Training epochs: 20 with early stopping
- Gradient norm clipping: 0.5
- Weight decay: 0.01

#### Data Processing
- Input image resizing:
  - Minimum pixels: 64×28×28
  - Maximum pixels: 128×28×28
- Dataset reduction factor: 4 (for faster iteration)
- Train/test split: 80/20

### Loss and Metrics Calculation

#### Training Loss
The training process utilized the pre-implemented loss function in the Qwen2-VL-2B model, which is a standard cross-entropy loss adapted for sequence generation:

```python
with autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss / config.accumulation_steps
```

This loss function:
- Computes the negative log-likelihood of the correct token at each position
- Automatically handles the teacher forcing during training
- Masks out loss computation for padding tokens
- Is scaled by the accumulation steps for gradient accumulation

#### Validation Strategy
For validation, we implemented a two-step process:

1. **Generation Step**:
```python
generated_ids = model.generate(
    **inputs_users,
    num_beams=1,  # Greedy decoding
    max_new_tokens=30,
    pad_token_id=processor_val.tokenizer.pad_token_id
)
```

2. **Category Parsing**:
```python
def parse_category(output_str, categories=[...]):
    """Extract category value from model output string"""
    lines = [line.strip() for line in output_str.lower().split('\n') 
            if line.strip()]
    
    if len(lines) >= 3:
        assistant_message = lines[4:]
        for category in categories:
            for line in assistant_message:
                if category.lower() in line:
                    return category       
    return "Unknown"
```

This approach:
- Allows the model to generate free-form text responses
- Uses a custom parser to extract category predictions
- Handles variations in response formatting
- Falls back to "Unknown" for uncertain predictions

The validation metrics (accuracy, precision, recall, F1) are then computed by comparing the parsed categories with the ground truth labels. This method preserves the model's natural language generation capabilities while enabling structured classification evaluation.

### 2.5 Model Input Format
The model expects inputs in a specific conversation format:
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image"},
                {"type": "text", "text": "What type of skincare product is shown?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Product Category"}
            ]
        }
    ]
}
```

This structured format ensures consistent training and inference behavior while maintaining the model's conversation capabilities.

## 3. Results Analysis

### Challenges
Due to limited GPU resources, I had to make some compomises to have a decent training time (around 1h/10 epochs) and being able to test different
hyperparameters. To achieve faster training, I used gradient checkpointing, gradient accumulation, and mixed precision training. I also ended up 
freezing more layers than initially planned to fasten the backward propagation and gradient computation. In addition, I settled for a batch size of 1, which is not the best for the model, but it was the only way to fit the model in the GPU memory. To compensate, I used a 4 gradient accumulations steps, which emulate a batch size of 4 regarding the gradient computation.

During early trainings, I started with a learning rate of 1e-5, which was too high and made the model diverge. I then tried with a learning rate of 5e-7, which was too low and made the model converge very slowly. I finally settled for a learning rate of 5e-7 with cosine annealing warm restarts, which seemed to work the best. I also used a only a fraction of the training data and testing data, again to save time and be able to run more trials, which in turn might have cause the model to overfit, having a training set of 400 images and a testing set of 200 images at each epoch. To confirm the results, it would be needed to run the model on the full dataset, which I have not been able to do due to limited GPU resources and time constraints.

With that being said, let's dive into the results!

### 3.1 Performance Metrics Analysis

| Metric    | Before Fine-tuning | After Fine-tuning | Change |
|-----------|-------------------|-------------------|---------|
| Accuracy  | 29.4%             | 50.0%            | +20.6%  |
| Precision | 77.4%             | 59.5%            | -17.9%  |
| Recall    | 29.4%             | 50.0%            | +20.6%  |
| F1 Score  | 36.0%             | 51.0%            | +15.0%  |

The results show interesting patterns:

1. **Accuracy Improvement**:
   - Significant increase from 29.4% to 50.0% (+20.6%)
   - Model is now correctly classifying half of all test samples
   - This represents a substantial improvement in overall classification performance

2. **Precision-Recall Trade-off**:
   - Initial high precision (77.4%) but low recall (29.4%) indicates the zero-shot model was:
     - Very conservative in its predictions
     - Highly confident when making correct predictions
     - Missing many correct classifications (false negatives)
   - After fine-tuning:
     - Precision decreased to 59.5%
     - Recall increased to 50.0%
     - This suggests the model became more balanced in its predictions

3. **F1 Score Evolution**:
   - Improved from 36.0% to 51.0%
   - Despite the precision drop, the F1 score increased by 15 points
   - This confirms that the overall classification performance is better balanced

4. **Interpretation**:
   - The zero-shot model was overly cautious, making few but highly confident predictions
   - Fine-tuning made the model more willing to make predictions across all categories
   - The trade-off between precision and recall resulted in a more practical model for real-world use

These metrics suggest that while the fine-tuned model might make more mistakes when it makes a prediction (lower precision), it is much better at catching correct classifications across all categories (higher recall), resulting in a more balanced and practical model for real-world applications.

### 3.2 Training Progress

The model was trained for a total of 17 epochs, with the best performing checkpoint saved at epoch 14. Let's analyze the evolution of different metrics:

![alt text](./plots/lr.png)
**Learning Rate Schedule**:
- Cosine annealing pattern clearly visible
- Initial learning rate of 5e-7
- Regular warm restarts helping escape local minima
- Gradual decrease allowing fine-grained parameter updates

![alt text](./plots/val_loss_15.png)
**Validation Loss**:
- Sharp initial decrease in the first 3 epochs
- Several plateaus followed by improvements after learning rate restarts
- Best validation loss achieved around epoch 14
- Slight increase in later epochs indicating potential overfitting

![alt text](./plots/val_accuracy_15.png)
**Validation Accuracy**:
- Steady improvement from initial 29.4% to peak of 50.0%
- Most significant gains in first 8 epochs
- Stabilization around epoch 14
- Minor fluctuations in later epochs

![alt text](./plots/val_precision.png)
**Validation Precision**:
- Initial high precision of 77.4% drops significantly in early epochs
- Stabilizes around 60% by epoch 10
- Trade-off between precision and recall becomes evident
- Final value of 59.5% at chosen checkpoint

![alt text](./plots/val_recall_15.png)
**Validation Recall**:
- Steady increase from 29.4% to around 50%
- Mirrors accuracy improvements
- Most gains achieved by epoch 14
- Plateaus in later epochs

![alt text](./plots/val_f1.png)
**F1 Score Evolution**:
- Consistent improvement from 36% to 51%
- Best balance between precision and recall at epoch 14
- Slight degradation after epoch 15

**Key Observations**:
1. **Optimal Checkpoint Selection** (Epoch 14):
   - Best balance across all metrics
   - Peak F1 score achievement
   - Stable performance before potential overfitting

2. **Training Dynamics**:
   - Learning rate warm restarts effectively prevented stagnation
   - Clear evidence of model convergence
   - Good balance between exploration and exploitation phases

3. **Stopping Decision**:
   - Training continued to epoch 17 to confirm no further improvements
   - Early stopping would have correctly identified epoch 14 as optimal
   - Additional epochs showed slight performance degradation

This analysis supports the decision to use the epoch 14 checkpoint as the final model version, as it represents the optimal trade-off between model performance metrics and generalization capability. However, we could probably improve the model by training it for more epochs, using the whole dataset, and more hyperparameter tuning. It is also to be noted that the GPU did not support flash attention, slowing down the training process, and did not allow for larger batch sizes as well as higher image resolutions. Theses results are promising, but they are only a starting point, and for sure far to production level accuracy.

#### Pre-Fine-tuning Confusion Matrix (Zero-shot Performance)

![alt text](./plots/cm_before_training.png)

### Analysis of zero-shot classification:

1. **Strong Categories**:
   - Cleanser (13 correct) and Serum (13 correct) show good performance
   - Mask category performs well (14 correct predictions)

2. **Key Confusion Patterns**:
   - Very high number of samples classfied as Unkown (around 100), showing the lack of uderstanding and specific knowledge of the base model.
   - Eye Care products often confused with Serum (8 cases)
   - Cleanser predictions overlap with several categories, showing some uncertainty in distinguishing cleansing products

3. **Problematic Categories**:
   - Most categories fail to be classified, in particular Moisturiser with 22/26 Unkowns and only 1 correct prediction.
   - Toner shows no correct predictions (0/6 cases)
   - Bath Salts poorly recognized (0/4 cases)
   - Mist often confused with other categories (only 1/5 correct)

4. **Class Imbalance**:
   - Moisturiser has highest representation (28 samples)
   - Some categories like Peel and Bath Salts are underrepresented (3-4 samples)
   - This imbalance affects the model's ability to learn rare categories

#### Post-Fine-tuning Confusion Matrix

![alt text](./plots/cm_epoch_13.png)

### Improvements and remaining challenges after fine-tuning:

1. **Major Improvements**:
   - Overall diagonal pattern is stronger, indicating better classification across categories
   - Moisturiser classification improved significantly
   - Better distinction between Serum and Cleanser categories
   - Mask category maintains strong performance
   - Far less Unkowns, model improves understanding of the dataset

2. **Persistent Challenges**:
   - Some confusion remains between similar product types:
     - Toner/Mist distinction still challenging
     - Bath products (Bath Oil, Bath Salts) show room for improvement
   - Underrepresented categories still show lower performance

3. **Business Impact Analysis**:
   - The model is most reliable for common skincare categories (Moisturiser, Cleanser, Serum)
   - Care should be taken when deploying for specialized products (Peels, Bath products)
   - Consider grouping similar categories (e.g., Toner/Mist) for production use

4. **Recommendations Based on Confusion Patterns**:
   - Collect more training data for underrepresented categories
   - Consider hierarchical classification for similar product types
   - Implement confidence thresholds for critical classifications
   - Use product metadata to help disambiguate similar categories

## 4. Performance Optimization Suggestions

### 4.1 Data-level Improvements

1. **Dataset Expansion**:
   - Collect additional samples for underrepresented categories (Bath Salts, Peel, Toner)
   - Include more product angles and lighting conditions
   - Consider multi-source data collection beyond LookFantastic

2. **Data Augmentation Strategies**:
   - Implement brightness and contrast adjustments
   - Add random cropping and rotation
   - Apply color jittering to simulate different lighting conditions
   - Use modern augmentation techniques like RandAugment or AugMix

3. **Data Quality Enhancements**:
   - Refine blur detection threshold with expert validation
   - Implement object detection to focus on product packaging
   - Consider background removal for cleaner training samples

### 4.2 Model-level Optimizations

1. **Architecture Improvements**:
   - Experiment with unfreezing more visual encoder blocks
   - Test different layer freezing combinations
   - Consider adding classification head instead of pure generation
   - Implement attention visualization for better interpretability

2. **Training Optimizations**:
   - Utilize flash attention when hardware permits
   - Increase batch size with better GPU resources
   - Experiment with different learning rate schedules
   - Test alternative optimizers (e.g., AdaFactor, Lion)

3. **Memory Efficiency**:
   - Implement gradient checkpointing more aggressively
   - Use 8-bit quantization during training
   - Optimize input image resolution vs. model performance
   - Consider model pruning techniques

### 4.3 Hardware and Infrastructure

1. **GPU Requirements**:
   - Upgrade to GPU with flash attention support
   - Minimum 24GB VRAM for larger batches
   - Consider multi-GPU training setup
   - Test cloud GPU alternatives (A100, H100)

2. **Storage and Processing**:
   - Implement efficient data pipeline
   - Use distributed storage for dataset
   - Set up proper backup and versioning
   - Consider data streaming for large datasets

## 5. Conclusion

This project demonstrated the potential of fine-tuning Qwen2-VL-2B for specialized product classification while highlighting several key insights:

### Key Achievements
- Successfully developed an end-to-end pipeline from data collection to model deployment
- Improved classification accuracy from 29.4% to 50.0% through targeted fine-tuning with limited GPU resources.
- Achieved better model understanding evidenced by significant reduction in "Unknown" predictions
- Maintained model's conversation capabilities while adding domain-specific knowledge

### Technical Insights
1. **Architecture Adaptation**:
   - Strategic layer freezing proved effective for transfer learning
   - Vision-language merger fine-tuning crucial for domain adaptation
   - Memory optimization techniques enabled training on limited hardware

2. **Performance Trade-offs**:
   - Precision-recall balance shifted from high-precision/low-recall to more balanced performance
   - F1 score improved by 15 points despite precision decrease
   - Model gained confidence in predictions across all categories

### Limitations and Challenges
- Hardware constraints limited batch size and training scope
- Class imbalance affected performance on rare categories
- Similar product types remain challenging to distinguish
- Generative model is not exploited to its full potential by predicting single category at a time, it would be better to predict a structured output with multiple predictive variables.
- Limited dataset size and reduction factor may impact generalization

### Future Directions
1. **Immediate Improvements**:
   - Train on full dataset 
   - Implement suggested data augmentation strategies
   - Address class imbalance 

2. **Long-term Vision**:
   - Expand to structured product information extraction
   - Develop hierarchical classification system
   - Create more robust evaluation metrics for complex outputs
   - Build production-ready deployment pipeline


I really enjoyed working on this project and I learned a lot about the potential of large vision-language models. I am aware of the limitations of my work, and I am eager to receive feedback and suggestions to improve the whole pipeline.

Thank you for your time and for reading this report.