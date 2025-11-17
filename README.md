# Skincare Product Classification with Qwen2-VL-2B
This project implements a fine-tuned version of Qwen2-VL-2B for classifying skincare products into 14 categories using product images. The model achieves 50% accuracy, improving from 29.4% in zero-shot performance.

```
├── clean_dataset/
│   ├── images/          # Cleaned and processed product images
│   └── annotations/     # Product category annotations
├── csv_data/
│   └── images_id_full.csv   # Image IDs and URLs mapping
├── log/
│   ├── cleaning_report.log  # Data cleaning process logs
│   └── scraper.log         # Web scraping logs
├── product_images/      # Raw scraped images
├── plots/              # Training metrics visualizations
└── src/
    ├── crawler.py      # Web scraping implementation
    ├── dataset_finetune.py  # Dataset preparation
    ├── vision_util.py  # Vision util functions
    ├── qwen2_base.py   # Model training and evaluation
    ├── qwen2_vl_2b_instruct_0shot.py  # Zero-shot testing instruct model
    ├── qwen2_vl_2b_0shot.py  # Zero-shot testing base model
    ├── fine_tuning.py  # Fine-tuning pipeline and utilities
    ├── logutil.py      # Logging configuration and utilities
    ├── analyse_log.py  # Scraping Log analysis
    └── cleaning.py     # Data cleaning and preprocessing
```

## Getting Started
Install dependencies:
pip install -r requirements.txt
## Set up environment variables:
"export HF_TOKEN="your_huggingface_token"
## Load the fine-tuned model:
from transformers import AutoProcessor, AutoModelForImageTextToText

auto_processor = AutoProcessor.from_pretrained("path/to/base-model")
auto_model = AutoModelForImageTextToText.from_pretrained("path/to/fine-tuned-model")

The processor will be the one of the base model, the model will be the fine-tuned model.
## Input wandb and HF credentials in the .env file

# Results
The fine-tuned model shows significant improvements:
Accuracy: 50.0% (+70%)
F1 Score: 51.0% (+42%)
Better handling of product categories with reduced "Unknown" classifications
For detailed analysis and future improvement suggestions, refer to the full report.
