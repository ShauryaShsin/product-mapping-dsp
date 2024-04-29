import pandas as pd
from model_inference import predict_product

# Create dataframe (table) to store Sallings products
columns = ["SKU", "name", "brand", "price", "kg", "image_url"]

salling_product_data = [
    ["100", "favorit", "kims", 1.99, 0.150, "src/product_images/favorit/image.jpg"],
    ["200", "havsalt", "kims", 2.50, 0.150, "src/product_images/havsalt/image.jpg"],
    ["300", "sour", "kims", 2.25, 0.150, "src/product_images/sour/image.jpg"],
]

# Create a DataFrame with this data
df_salling = pd.DataFrame(salling_product_data, columns=columns)
df_salling.to_csv("src/data/salling_products.csv")

# print(df_salling.head(10))

# Create dataframe (table) to store retailers scraped products
columns = [
    "Retailer_SKU",
    "Salling_SKU",
    "Retailer",
    "name",
    "brand",
    "price",
    "kg",
    "image_url",
]

# Assuming the scraper has found the following products
scraped_product_data = [
    [
        "48",
        None,
        "Nemlig.com",
        "favorit",
        "kims",
        1.99,
        0.150,
        "src/product_images/favorit/image.jpg",
    ],
    [
        "41",
        None,
        "Nemlig.com",
        "havsalt",
        "kims",
        2.50,
        0.150,
        "src/product_images/havsalt/image.jpg",
    ],
    [
        "23",
        None,
        "Nemlig.com",
        "sour",
        "kims",
        2.25,
        0.150,
        "src/product_images/sour/image.jpg",
    ],
]

df_retailer = pd.DataFrame(scraped_product_data, columns=columns)
df_retailer.to_csv("src/data/retailer_products.csv")

# Use the trained model to predict which Salling product the scraped product corresponds to
for index, row in df_retailer.iterrows():

    predicted_salling_product = predict_product(img_path=row["image_url"])

    salling_match = df_salling[df_salling["name"] == predicted_salling_product]

    salling_sku = salling_match["SKU"]

    print(salling_sku)

    df_retailer.at[index, "Salling_SKU"] = salling_sku

    df_retailer.to_csv("src/data/retailer_products.csv")
