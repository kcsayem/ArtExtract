import pandas as pd
import requests
from tqdm import tqdm

data = pd.read_csv("data/published_images.csv")

def download_image(url, save_path):
    try:
        # Send a GET request to the URL to fetch the image
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Open the file in binary write mode
            with open(save_path, 'wb') as file:
                # Write the image content to the file in chunks
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            # print(f"Image downloaded successfully and saved as '{save_path}'")
            return 1
        else:
            return 0
            # print(f"Failed to download image: Status code {response.status_code}")
    except Exception as e:
        # print(f"An error occurred: {e}")
        return 0

counter = 0
for i in tqdm(range(len(data))):
    url = data.loc[i, "iiifthumburl"]
    save_path = f"data/images/{data.loc[i, 'uuid']}.jpg"
    count = download_image(url, save_path)
    counter += count

print(f"Downloaded {counter} images")
