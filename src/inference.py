# imports torch
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader

# stats
import glob
import pandas as pd
import numpy as np

# matplotlib and PIL
import matplotlib.pyplot as plt
import PIL
# local
from HPA_transforms import get_transform
from HPA_dataset import HPAImageDataset
from HPA_model_fetch import resnet_50_4_channel as resnet50
# model
model_path = '../models/resnet50_14_04_21.pth'
df_path = '../input/image_subset_example/example.csv'
ROOT_DIR = '../input/image_subset_example/'

def show_one(loader):
    single_batch = next(iter(loader))
    print(single_batch[0][0].shape)
    im = (transforms.ToPILImage(mode='RGB')(single_batch[0][0][:3, :, :]))
    im.show()


def fetch_dataset():
    def collate_fn(batch):
        anno = []
        tiles = []
        for b in batch:
            for label in b[1]:
                anno.append(label)
            for tile in b[0]:
                tiles.append(tile)
        return tuple(zip(tiles, anno))


    df = pd.read_csv(df_path)
    example_dataset = HPAImageDataset(root=ROOT_DIR,
                                      transforms=get_transform(),
                                      df=df)
    test_loader = DataLoader(example_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)
    return(test_loader)


def do_inference(loader, model):
    batch = next(iter(loader))
    model.eval()
    img = batch[0][0]
    label = batch[0][1]

    with torch.no_grad():
        X = img.unsqueeze(0).float()
        y = torch.reshape(label, (1,19))
        pred = model(X)

    print(pred)
    print(y)

if __name__ == '__main__':
    loader = fetch_dataset()
    model = resnet50(model_path)
    do_inference(loader, model)
    #show_one(loader)
