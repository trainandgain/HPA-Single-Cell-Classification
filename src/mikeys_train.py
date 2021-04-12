'''Train small network on cell level images'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.datasets import DataLoader

from cell_dataset import CellDataset

IMG_ROOT = '../input/train_cells'
TRAIN_CSV = '../input/train_cells/train.csv'
RANDOM_STATE = 42

def extract_as_array(str_):
    list_ = str_.strip('][').split(', ')
    return np.array([int(i) for i in list_])


if __name__ == '__main__':
    
    # Read training csv and init train/test data loaders
    df = pd.read_csv(train_csv, index_col=0)
    images = df['cell_id'].values
    targets = df['Label'].apply(extract_as_array).values
    # TODO: stratify this split
    train_images, test_images, train_targets, test_targets = train_test_split(
    images, targets, random_state=RANDOM_STATE
        )
    
    train_cells = CellDataset(train_images, train_targets, IMG_ROOT)
    test_cells = CellDataset(test_images, test_targets, IMG_ROOT)
    
    train_loader = DataLoader(train_cells)
    test_loader = DataLoader(test_cells)
    