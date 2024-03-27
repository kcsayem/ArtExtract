import os
from PIL import Image, ImageFile
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
class WikiArtDataset(Dataset):
    def __init__(self, csv_dir, root_dir, transform=None):
        self.annotations = json.load(open(csv_dir, "r"))
        self.root_dir = root_dir
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations[index][0])
        image = Image.open(img_path)
        y_label = self.annotations[index][1:]
        y_label = torch.tensor(y_label)

        if self.transform:
            image = self.transform(image)

        return image, y_label
    

class Transform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = np.array(image)
        image = image / 255
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        return image
        
    
def get_dataloader(csv_dir, root_dir, transform, batch_size):
    dataset = WikiArtDataset(csv_dir, root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    csv_dir = 'train.json'
    root_dir = 'wikiart/'
    # transform = WikiArtDataset.Transform(128)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataloader = get_dataloader(csv_dir, root_dir, transform, 32, train=True)
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
        break

