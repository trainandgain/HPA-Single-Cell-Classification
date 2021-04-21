import os
from PIL import Image
import pandas as pd
class HPAImageDataset(object):


    def __init__(self, root, transforms, df):
        self.root = root
        self.df = df
        self.transforms = transforms


    def __len__(self):
        return(len(self.df))


    def __getitem__(self, idx):
        # load channels and masks for idx
        r_path = os.path.join(self.root, "image", 
                              self.df.ID[idx]+'_red.png')
        g_path = os.path.join(self.root, "image", 
                              self.df.ID[idx]+'_green.png')
        b_path = os.path.join(self.root, "image", 
                              self.df.ID[idx]+'_blue.png')
        y_path = os.path.join(self.root, "image", 
                              self.df.ID[idx]+'_yellow.png')

        mask_path = os.path.join(self.root, 
                                 "mask", 
                                 self.df.ID[idx]+'_predictedmask.png')

        # open imgs in PIL
        imgs = {
            'red': Image.open(r_path),
            'green': Image.open(g_path),
            'blue': Image.open(b_path),
            'yellow': Image.open(y_path),
            'mask': Image.open(mask_path)
        }

        #define target
        target = {
            'image_id': idx,
            'labels': self.df.iloc[0, 2:].to_numpy(dtype='float16')
        }

        if self.transforms is not None:
            # torchvision compose only accepts single input
            sample = self.transforms((imgs, target))
            imgs, target = sample
        # return((imgs, target))  if we want to include imge ID
        return((imgs, target['labels'])) 
