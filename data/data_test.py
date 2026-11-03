import pandas as pd
import urllib.request
import os

# check if data exists, if not download it
if not os.path.exists("data/ridership_headline.parquet"):
    # download the data
    URL_DATA = 'https://storage.data.gov.my/transportation/ridership_headline.parquet'

    os.makedirs("data", exist_ok=True)

    print(f"Downloading {URL_DATA}...")

    urllib.request.urlretrieve(URL_DATA, "data/ridership_headline.parquet")

    print("Download complete.")

df = pd.read_parquet("data/ridership_headline.parquet")

print (df.shape)
print (df.columns.tolist())