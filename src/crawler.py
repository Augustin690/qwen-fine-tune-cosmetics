"""Crawler to download skin-care product images on LookFantastic's website."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from typing import Tuple
from retrying import retry

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import logging
import time


# Define a retry decorator with exponential backoff
@retry(
    wait_exponential_multiplier=1000,  # Start with 1 second delay
    wait_exponential_max=10000,        # Max 10 seconds delay
    stop_max_attempt_number=5,         # Max 5 retries
    retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException,))
)
def fetch_url_with_retries(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response

# Function to scrape product images
def scrape_product_images(url:str)-> list|None:
    try:
        logging.info(f"Processing URL: {url}")
        # Fetch the webpage
        response =  fetch_url_with_retries(url)  # Use the retry function
        if response.status_code != 200:
            print(f"Failed to fetch URL: {url} (Status Code: {response.status_code})")
            logging.error(f"Failed to fetch URL: {url} (Status Code: {response.status_code})")
            return None

        # Parse the webpage
        soup = BeautifulSoup(response.text, 'lxml')

        # Extract product name or ID (adjust selector as needed)
        #product_name = soup.find("h1").get_text(strip=True)  # Assuming the product name is in an <h1> tag
        product_name = url.split("/")[-2]  # Example: Use the second-to-last part of the URL as an ID
        if not product_name:
            logging.warning(f"Failed to find product name: {url}")
            product_name = 'unknown_product'
        #product_key = f"{product_name}_{product_id}"

        #print(f"Product ID: {product_id}")
        print(f"Product name: {product_name}")

        # Locate the script tag containing image URLs
        script_tag = soup.select_one("#view-item-container > div:nth-of-type(1) > div > script")
        if not script_tag:
            print(f"No script tag found for URL: {url}")
            logging.error(f"No script tag found for URL: {url}")
            return None

        # Extract the script content
        script_content = script_tag.string
        if not script_content:
            print(f"No content in script tag for URL: {url}")
            logging.error(f"No content in script tag for URL: {url}")
            return None

        #print(script_content)

        # Regex to match `const images` specifically
        pattern = r'const\s+images\s*=\s*(\[[^\]]*\])'
        # Search for the `images` array
        match = re.search(pattern, script_content)
        if not match:
            print(f"No const declaration found in script for URL: {url}")
            logging.error(f"No const declaration found in script for URL: {url}")
            return None

        # Extract and parse the JSON-like object
        data = match.group(1)
        image_urls = re.findall(r'"(https?:\/\/[^"]*?\.(?:jpg|jpeg|png|webp))"', data)
        if not image_urls:
            print(f"No image URLs found for URL: {url}")
            logging.error(f"No image URLs found for URL: {url}")
            return None

        # Return product name and image URLs
        print(f"Found {len(image_urls)} images")
        return image_urls

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch URL: {url} (Exception: {e})")
        logging.error(f"Failed to fetch URL after retries: {url} - Error: {e}")
        return None

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        logging.error(f"Error processing URL {url}: {e}")
        return None

# Function to download images and save to product-specific folders
def download_image(image_filename: str, image_url:str, save_dir:str) -> str|None:
   # product_folder = os.path.join(save_dir, product_key)
   # os.makedirs(product_folder, exist_ok=True)

    image_path = os.path.join(save_dir, f"{image_filename}.jpg")

    if os.path.exists(image_path):
        print(f"Image {image_filename} already exists. Skipping download.")
        logging.info(f"Image {image_filename} already exists. Skipping download.")
        return image_filename
    try:
        img_response = fetch_url_with_retries(image_url)
        if img_response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in img_response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {image_path}")
            logging.info(f"Downloaded: {image_path}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Failed to download: {image_url}")
        logging.error(f"HTTP error occurred: {http_err} - Failed to download: {image_url}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image {image_url} after retries: {e}")
        logging.error(f"Failed to download image {image_url} after retries: {e}")
    except Exception as e:
        # connection error
        print(f"Error downloading image {image_url}: {e}")
        logging.error(f"Error downloading image {image_url}: {e}")


def process_product(row, save_images_dir):
    product_url = row['product_url']
    image_urls = scrape_product_images(product_url)
    product_id = product_url.split('/')[-1].replace('.html', '')
    product_name = product_url.split('/')[-2]
    image_data = []

    if image_urls:
        for i, image_url in enumerate(image_urls):
            image_id = f"{product_id}_{i+1}"
            image_file = f"{product_name}_{image_id}"
            image_data.append({
                "product_url": product_url,
                "image_id": image_id,
            })
            # Download images
            download_image(image_file, image_url, save_images_dir)
            time.sleep(0.1)  # Avoid overloading the server

    return image_data

def run(products_csv_path:str, save_images_dir:str, images_csv_path:str) -> None:
    logging.basicConfig(
        filename="../log/scraper.log",  # Log file
        level=logging.INFO,      # Log everything from INFO and above
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Load the product csv_data from CSV
    df = pd.read_csv(products_csv_path)
    # Initialize list for storing image ids and product urls
    all_image_data = []
    print(f"Set to scrape {len(df)} skincare products")

    # Use ThreadPoolExecutor to parallelize the processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_product, row, save_images_dir) for _, row in df.iterrows()]

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URLs", colour="white"):
            try:
                all_image_data.extend(future.result())
            except Exception as e:
                logging.error(f"Error processing a product: {e}")

    pd.DataFrame(all_image_data).to_csv(images_csv_path, index=False)  # Save image-level info


if __name__ == "__main__":
    # XPath of URL images : [@id="view-item-container"]/div[1]/div/script/text()
    # Directory to save images
    save_dir = "../product_images"
    image_csv_path = "../csv_data/images_id_full.csv"  # New CSV for image-level csv_data
    skincare_products_path = "../csv_data/skincare_products_clean.csv"

    run(skincare_products_path, save_dir, image_csv_path)


