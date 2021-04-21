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

# local
from HPA_transforms import get_transform
from HPA_dataset import HPAImageDataset

# model
model_path = '../models/resnet50_14_04_21.pth'
df_path = '../input/image_subset_example/example.csv'

# model
def model_fetch(PATH):
    """
    Function for fetching custom resnet model
    """
    model = torchvision.models.resnet50()
    model.conv1 = nn.Conv2d(4, 64, (7, 7),
                            stride=(2,2),
                            padding=(3,3),
                            bias=False)

    # output features to 3 classification
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=19
        ),
        torch.nn.Sigmoid())

    model.load_state_dict(torch.load(PATH))
    model.eval()
    return(model)


model = model_fetch(model_path)

ROOT_DIR = '../input/image_subset_example/'
df = pd.read_csv(df_path)
example_dataset = HPAImageDataset(root=ROOT_DIR,
                                  transforms=get_transform(),
                                  df=df)

def collate_fn(batch):
    anno = []
    tiles = []
    for b in batch:
        for label in b[1]:
            anno.append(label)
        for tile in b[0]:
            tiles.append(tile)
    return tuple(zip(tiles, anno))


test_loader = DataLoader(example_dataset,
                         batch_size=1,
                         shuffle=False,
                         collate_fn=collate_fn)


if __name__ == '__main__':
    single_batch = next(iter(test_loader))
    print(single_batch[0][0].shape)
