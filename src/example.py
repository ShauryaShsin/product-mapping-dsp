import pandas as pd

# Define columns
columns = ["SKU", "name", "brand", "price", "kg", "image_url"]

# Example data as a list of lists
salling_product_data = [
    ["1", "favorit", "kims", 1.99, 0.150, "product_images/favorit/image.jpg"],
    ["2", "havsalt", "kims", 2.50, 0.150, "product_images/havsalt/image.jpg"],
    ["3", "sour", "kims", 2.25, 0.150, "product_images/sour/image.jpg"],
]

# Create a DataFrame with this data
df = pd.DataFrame(salling_product_data, columns=columns)

print(df.head(10))
