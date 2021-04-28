'''Segment images into individual cells with image-level weak labels'''

import os
from typing import List
from argparse import ArgumentParser
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hpacellseg.cellsegmentator import CellSegmentator
from hpacellseg.utils import label_cell

from config import CHANNELS, NUC_MODEL, CELL_MODEL

def read_training_csv(csv_path):
    'Return dataframe containing ids and labels for training set images'
    train = pd.read_csv(csv_path)
    # One-hot encode label column
    def encoder(row):
        # Init vector of zeros
        encoded_label = [0] * 19
        labels = row['Label'].split('|')
        for label in labels:
            # Replace zeros with 1's based on labels present
            encoded_label[int(label)] = 1
        row['Label'] = encoded_label
        return row
    
    train = train.apply(encoder, axis=1)
    
    return train

def fetch_channels(img_id: str, dir_path: str, channel_names=CHANNELS) -> List[str]:
    'Return absolute path of segmentation channels of a given image id'
    base = os.path.join(dir_path, img_id)
    return [base + '_' + i  + '.png' for i in channel_names]

def get_segmentation_mask(ref_channels: List[str], segmentator: CellSegmentator):
    'Return cell segmentation mask for single image using paths to reference channels'
    # Ref channels must be in order red, yellow, blue
    input_ = [[i] for i in ref_channels]  # Segmentator only accepts list of lists input
    nuc_segmentation = segmentator.pred_nuclei(input_[2])[0]
    cell_segmentation = segmentator.pred_cells(input_)[0]
    mask = label_cell(nuc_segmentation, cell_segmentation)[1]
    return mask

def is_image_edge(img_shape, y_min, y_max, x_min, x_max):
    'Return true if cell bounding box is touching edge of parent image'
    if (y_min == 0) or (x_min == 0): 
        return True
    elif ((y_max + 1) >= img_shape[0]) or ((x_max + 1) >= img_shape[1]):
        return True
    else:
        return False

def record_metadata(array_img, cell_id, cell_num, parent_id, edge_of_image, df):
    'Return updated dataframe with image metadata'
    metadata = {'cell_id': cell_id,
                'parent_image_id': parent_id,
                'cell_number': cell_num,
                'size_y': array_img.shape[0],
                'size_x': array_img.shape[1],
                'edge_of_img': edge_of_image # is on the edge of parent image?
                }
    return df.append(metadata, ignore_index=True)
    
def save_img(destination_dir, filename, array_img):
    'Save numpy array to png file'
    path = os.path.join(destination_dir, filename)
    plt.imsave('{}.png'.format(path), array_img)
    
def extract_and_save(parent_id, channels, mask, df, destination, visualise=False):
    'Save individual cells as channel images and record metadata in df'
    for label in np.unique(mask):
        # Get values from where image == label
        if label == 0:
            continue  # ignore background
        temp_mask = mask.copy()
        temp_mask[temp_mask != label] = 0
        temp_mask[temp_mask == label] = 1
        # Get temp mask bounding box coords
        idxs = np.asarray(temp_mask == 1).nonzero()
        y_min, y_max = idxs[0].min(), idxs[0].max()
        x_min, x_max = idxs[1].min(), idxs[1].max()
        edge_of_image = is_image_edge(temp_mask.shape, y_min, y_max, x_min, x_max)
        
        for channel in channels:
            channel_arr = np.array(plt.imread(channel))
            channel_colour = channel.split('_')[-1][:-4]  # grab colour from end of channel path
            # Zero pad and square off
            single_cell = temp_mask * channel_arr
            single_cell = single_cell[y_min:(y_max + 1), x_min:(x_max + 1)]
            #plt.imshow(single_cell)
            cell_id = f'{parent_id}_cell_{label}'  # Overall cell id
            cell_channel_id = f'{cell_id}_{channel_colour}'  # Append colour channel when saving
            save_img(destination, cell_channel_id, single_cell)
            
        df = record_metadata(single_cell, cell_id, label, parent_id, edge_of_image, df)
    
    return df

def segment_image(img_id, input_dir, output_dir, seg):
    'Segment cells in img_id and save these to a directory, return metadata on img'
    # arg pass the dir through from terminal
    channels = fetch_channels(img_id, input_dir)
    ref_channels = channels[:3]
    df = pd.DataFrame()
    
    mask = get_segmentation_mask(ref_channels, seg)
    df = extract_and_save(img_id, channels, mask, df, output_dir, visualise=False)
    
    return df

def save_progress(train_df, output_df, output_dir):
    'Save progress to csv file in output directory'
    # Append weak labels to cells in parent image 
    labels = train_df.set_index('ID')
    output_df = output_df.join(labels, on='parent_image_id')
    # Save final csv with segmented cell images
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(os.path.join(output_dir, 'train.csv'))
    
def run(input_dir, train_csv, output_dir, nuc_model_path, cell_model_path):
    'Run segment_image function on all images in training csv'
    # Init seg models
    seg = CellSegmentator(
        nuc_model_path,
        cell_model_path,
        scale_factor=0.25,
        device="cuda",
        padding=False,  # RUN W/ PADDING=TRUE???
        multi_channel_model=True
        )
    # Get image ids and their targets from provided training csv
    train_df = read_training_csv(train_csv)
    output_df = pd.DataFrame()  # init output_df to record metadata
    # Iterate through each image_id in train.csv
    for count, img_id, target in train_df.itertuples():
        df = segment_image(img_id, input_dir, output_dir, seg)
        output_df = output_df.append(df)
        # Checkpoint every 5 images saving progress to csv
        if count % 5 == 0:
            save_progress(train_df, output_df, output_dir)
        print(f'Image {count} processed!')
    
    # Final save at end of training dataframe
    save_progress(train_df, output_df, output_dir)
    

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    run(args.input_dir, args.train_csv, args.output_dir, NUC_MODEL, CELL_MODEL)
    