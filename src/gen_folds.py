'''Generate multi-label stratified folds for model training'''

import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from argparse import ArgumentParser

from config import RANDOM_STATE

def import_df(train_csv):
    'Import and shuffle training cells csv'
    df = pd.read_csv(train_csv, index_col=0)
    # Shuffle df to mix up single cell images before stratification
    df = df.sample(frac=1, random_state=RANDOM_STATE)
    df.reset_index(drop=True, inplace=True)
    return df

def extract_imgs_targets(df):
    'Extract and return features and targets as numpy arrays'
    images = df['cell_id'].values
    
    def extract_as_array(str_):
            list_ = str_.strip('][').split(', ')
            return np.array([int(i) for i in list_])
    targets = df['Label'].apply(extract_as_array).values

    # Reshape to single arrays
    img_mat = images.reshape(-1, 1)
    target_mat = np.vstack(targets)
    return img_mat, target_mat

def gen_folds(df, img_mat, target_mat, n_folds):
    'Return dataframe with folds column'
    k_fold = IterativeStratification(n_splits=n_folds, order=1)
    splits = k_fold.split(img_mat, target_mat)
    df['fold'] = 0  # Generate folds column
    # Grab fold number and img indexes from splits, adjust fold column accordingly
    for fold, (_, fold_idxs) in enumerate(splits):
        valid_imgs = img_mat[fold_idxs]
        df.loc[df['cell_id'].isin(valid_imgs.reshape(-1)), 'fold'] = fold  
    return df

def run(train_csv, output_csv, n_folds):
    df = import_df(train_csv)
    img_mat, target_mat = extract_imgs_targets(df)
    df = gen_folds(df, img_mat, target_mat, n_folds)
    df.to_csv(output_csv)
    

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--n_folds', type=int)
    args = parser.parse_args()
    
    run(args.train_csv, args.output_csv, int(args.n_folds))
    