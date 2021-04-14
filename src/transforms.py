class Tile(object):
    """
    Takes in:
    imgs = {
            'red': Image.open(r_path),
            'green': Image.open(g_path),
            'blue': Image.open(b_path),
            'yellow': Image.open(y_path),
            'mask': Image.open(mask_path)
        }
    
    target = {
        'image_id': torch.tensor([idx]),
        'labels': np.array(df.iloc[idx, 3:])
    }
    Outputs:
    
    imgs = {
            'tiles': tiles
        }
        
    target = {
        'image_id': torch.tensor([idx]),
        'labels': np.array(df.iloc[idx, 3:])
    }
    """
    @staticmethod
    def tiled_img(r, g, b, y, box):
        """
        Find center of box. Crop the image using PIL
        at center, using desired tile size.
        """
        # (xmax-xmin) / 2 , (ymax - ymin) /2
        center = ( (box[2]-box[0])/2, (box[3]-box[1])/2 )
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        
        # crop channels
        r = r.crop((left, top, right, bottom))
        g = g.crop((left, top, right, bottom))
        b = b.crop((left, top, right, bottom))
        y = y.crop((left, top, right, bottom))
        
        # return cropped channels using PIL 
        rgby = np.dstack((r,g,b,y))
    
        return(rgby)
    
    
    @staticmethod
    def split_mask(mask):
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
        return boxes
        
        
    @staticmethod
    def gen_labels(labels, num_objs):
        """
        Take in numpy array of one hot encoded labels
        output:
        [N, OHE labels]
        where N is the number of objects or tiles
        """
        return [labels for i in range(num_objs)]
        
        
    def __call__(self, sample):
        imgs, target = sample
        
        # get masks
        boxes = self.split_mask(imgs['mask'])
         
        # list of tiled imgs
        tiles = list()
        for box in boxes:
            tiles.append(self.tiled_img(imgs['red'],
                                        imgs['green'],
                                        imgs['blue'],
                                        imgs['yellow'],
                                        box))
            
        # target
        new_target = {
        'image_id': target['image_id'],
        'labels': self.gen_labels(target['labels'], len(boxes))
        }
        
        
        return ((tiles, new_target))


class ImageToTensor(object):
    """
    Custom to tensor class, does not accept dictionary.
    """

    @staticmethod
    def image_tensor(imgs):
        return [torch.from_numpy(im).permute(2,0,1) for im in imgs]
    
    
    @staticmethod
    def idx_tensor(idx):
        return torch.tensor(idx)
    
    
    @staticmethod
    def labels_tensor(labels):
        return[torch.from_numpy(l) for l in labels]
    
    
    def __call__(self, sample):
        imgs, target = sample
        # torch image: C X H X W
        t_tiles = self.image_tensor(imgs)
        idx = self.idx_tensor(target['image_id'])
        labels = self.labels_tensor(target['labels'])
        
        new_target = {
            'image_id': idx,
            'labels': labels
        }
        return ((t_tiles, new_target))


class Rescale(object):
    """
    Rescale the tiles received.
    Return rescaled 4 channel images (numpy)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)) # check output size is 
                                                     # right format
        self.output_size = output_size

    def __call__(self, sample):
        
        imgs, target = sample
        for i in range(len(imgs)):
            # numpy.shape responds with (H, W, C)
            h = imgs[i].shape[0] 
            w = imgs[i].shape[1] 

            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            #resize iamge
            imgs[i] = resize(imgs[i], 
                             (new_h, new_w))
        return ((imgs, target))