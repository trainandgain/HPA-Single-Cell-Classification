# imports
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np

# transformers
from torchvision import transforms

# dataset imports
import os
import torch
import torchvision
from torch.utils.data import DataLoader

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # check output size is 
                                                     # right format
        self.output_size = output_size

    def __call__(self, sample):
        
        image, target = sample['image'], sample['target']
        
        w, h = image.size 
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        #resize iamge
        img = image.resize((new_h, new_w))
        #resize mask image
        target['masks'] = target['masks'].resize((new_h, new_w))
        return {'image': img, 'target': target}
    


class ToTensor(object):
    """Custom to tensor class, does not accept dictionary."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        # torch image: C X H X W
        image = transforms.ToTensor()(image)
        return {'image': image, 'target': target}

    
class split_mask_bounding_box_area(object):
    """Custom to tensor class, does not accept dictionary."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        
        # load target
        mask = target['masks']
        # convert to array for individual segments
        mask = np.array(mask)
        # split into different segmentations
        obj_ids = np.unique(mask)
        # 0 is background, get rid
        obj_ids = obj_ids[1:]
        
        # split color-encoded mask into
        # a set of binary masks
        
        masks = mask == obj_ids[:, None, None] # HOW DOES THIS WORK
        
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
             
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class in picture
        # this is how the dataset is defined
        label = target['labels']
        # tensor of labels
        labels = torch.tensor([int(label)]*num_objs, dtype=torch.int64)
        
        # tensor masks
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # area of box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # define target
        target["masks"] = masks
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area

        return {'image': image, 'target': target}


# define transforms
def get_transform():
    custom_transforms = [Rescale(512),
                         split_mask_bounding_box_area(),
                         ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)

# define labels as a dictionary
map_labels={
    0: 1,
    14: 2,
    16: 3
}

class HPAImageDataset(object):
    
    
    def __init__(self, root, transforms, labels):
        self.root = root
        self.transforms = transforms
        # sort in order to make sure they
        # are aligned
        self.labels = labels # dictionary {'ID': 'label'}
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'image'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'mask'))))
                          
                          
    def __len__(self):
        return(len(self.imgs))
                          
    
    def __getitem__(self, idx):
        # load images and masks for idx
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        # open img in PIL
        img = Image.open(img_path)
        # load mask
        mask = Image.open(mask_path)
        
        #define target
        target = {}
        # set image id and mask
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        target["masks"] = mask
        # label there is only one class in picture
        # this is how the dataset is defined
        label = self.labels[os.path.basename(img_path[:-4])]
        target["labels"] = label
        
        if self.transforms is not None:
            # torchvision compose only accepts single input
            sample = self.transforms({'image': img, 'target': target})
            img, target = sample['image'], sample['target']

        return(img, target)

X = pd.read_csv('C:/Users/Admin/Git/HPA-Single-Cell-Classification/input/image_subset/subset_train.csv')
X.Label = X['Label'].astype('int')

ROOT_DIR = 'C:/Users/Admin/Git/HPA-Single-Cell-Classification/input/image_subset/'
# create own Dataset
labels = dict(zip(X.ID, X.Label.apply(lambda x: map_labels[x])))

transformed_dataset = HPAImageDataset(root=ROOT_DIR,
                                   transforms=get_transform(),
                                   labels=labels,
                                                )

sub_dataset = torch.utils.data.Subset(transformed_dataset, list(range(5)))

def collate_fn(batch):
    return(tuple(zip(*batch)))

# batch size
train_batch_size = 64


data_loader = DataLoader(sub_dataset, 
                                        batch_size=train_batch_size, 
                                        shuffle=True, 
                                        collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in
#                    annotations]
#     print(annotations)
for i_batch, sample_batched in enumerate(data_loader):
    print(i_batch, sample_batched)
    # observe 4th batch and stop.
    if i_batch == 3:
        break