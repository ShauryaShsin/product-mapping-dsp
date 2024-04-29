import requests
from bs4 import BeautifulSoup
import os

def download_images(url, folder_path):
    # Create folder if not exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Send request to the URL
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all image tags with class 'nem-packshot__img'
    img_tags = soup.find_all('img', class_='nem-packshot__img')
    
    # Download each image
    for img_tag in img_tags:
        img_url = img_tag.get('src')
        # Extract image name
        image_name = img_url.split('/')[-1].split('?')[0]
        # Download the image
        with open(os.path.join(folder_path, image_name), 'wb') as f:
            f.write(requests.get(img_url).content)
        print(f"Downloaded {image_name}")

# Example usage
url = 'https://www.nemlig.com/?search=chips'
folder_path = 'images'
download_images(url, folder_path)
