'''Dataset class for pre-segmented images in PyTorch format'''

import torch
import pandas as pd
import numpy as np
from PIL import Image
import os

#from config import CHANNELS
CHANNELS = ['red', 'green', 'blue', 'yellow']

class CellDataset(object):
    '''Dataset class to fetch HPA cell-level images
    and corresponding weak labels
    '''
    def __init__(self, images, targets, img_root, augmentations=None):
        self.images = images
        self.targets = targets
        self.img_root = img_root
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_id = self.images[idx] 
        img_channels = self._fetch_channels(img_id)
        features = self._channels_2_array(img_channels)
        # Adjust to channel first indexing for pytorch (speed reasons)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        # Grab target vector
        target = self.targets[idx]
       
        return {'image': torch.tensor(features),
                'target': torch.tensor(target)
                }
    
    def _fetch_channels(self, img_id: str, channel_names=CHANNELS):
        'Return absolute path of segmentation channels of a given image id'
        base = os.path.join(self.img_root, img_id)
        return [base + '_' + i  + '.png' for i in channel_names]
                                         
    def _channels_2_array(self, img_channels):
        'Return 3D array of pixel values of input image channels'
        # Init and reshape single channel array so we can concat other channels
        channel_1 = np.array(Image.open(img_channels[0]))
        shape = channel_1.shape + (1,)  
        pixel_arr = channel_1.reshape(shape)
        # Lay out 4 channels in 3D array for model input
        for channel in img_channels[1:]:
            channel_values = np.array(Image.open(channel)).reshape(shape)
            pixel_arr = np.concatenate([pixel_arr, channel_values], axis=2)
        return pixel_arr
