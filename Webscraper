import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Define the URL of the website to scrape
url = "https://shop.rema1000.dk/slik/mixposer"

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
    with open(os.path.join("images", f"image_{index}.jpg"), "wb") as f:
        f.write(img_content)
        # Increment the image count
        image_count += 1
        print(f"Image {index} saved successfully.")

# Print the total number of images found and saved
print(f"Total number of images found and saved: {image_count}")

# Quit the driver
driver.quit()
