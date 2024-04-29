import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Define the URL of the website to scrape
url = "https://shop.rema1000.dk/kolonial/chips-og-snacks"

# Set up Selenium WebDriver (make sure you have the appropriate driver installed)
driver = webdriver.Chrome()  # You can use other drivers like Firefox or Edge

# Load the webpage
driver.get(url)

# Wait for some time to let the page load completely (you can adjust the time as needed)
time.sleep(5)

# Find all image elements
image_elements = driver.find_elements(By.CSS_SELECTOR, "div.image.product-grid-image img")

# Create a directory to store the images
if not os.path.exists("images"):
    os.makedirs("images")

# Initialize a counter to keep track of the number of images saved
image_count = 0

# Iterate over each image element and download the image
for index, img in enumerate(image_elements):
    # Get the source URL of the image
    img_url = img.get_attribute("src")
    
    # Get the image content
    img_content = requests.get(img_url).content
    
    # Save the image to the images directory
    image_path = os.path.join("images", f"image_{index}.jpg")
    with open(image_path, "wb") as f:
        f.write(img_content)
    
    # Increment the image count
    image_count += 1
    print(f"Image {index} saved successfully.")

### PRINTING PRICE AND TITLE OF EACH IMAGE
# Find the price elements
price_elements = driver.find_elements(By.CLASS_NAME, "price-normal")

# Find the title elements
title_elements = driver.find_elements(By.CLASS_NAME, "title")

# Combine the image, price, and title elements into pairs
combined_list = zip(image_elements, price_elements, title_elements)

# Print the combined list with comma-separated elements
for index, (image_element, price_element, title_element) in enumerate(combined_list):
    image_path = os.path.join("images", f"image_{index}.jpg")
    print(f"Image Path: {image_path}, Price: {price_element.get_attribute('outerHTML')}, Title: {title_element.get_attribute('outerHTML')}")

# Print the total number of images found and saved
print(f"Total number of images found and saved: {image_count}")

# Quit the driver
driver.quit()
