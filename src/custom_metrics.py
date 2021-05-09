'''A set of custom metrics for monitoring network training with sparse class labels'''

import numpy as np
from sklearn.metrics import auc, average_precision_score
from statistics import mean


def calc_prec_rec(y_pred_, y_target, thresh):
    'Return precision and recall for given threshold in tuple'
    y_pred = y_pred_.copy()
    # Convert to binary predictions
    super_idxs = y_pred >= thresh
    y_pred[super_idxs] = 1
    sub_idxs = y_pred < thresh
    y_pred[sub_idxs] = 0
    
    tp = (y_pred.T @ y_target).item()
    fp = sum((y_pred - y_target) > 0).item()
    fn = sum((y_pred - y_target) < 0).item()
    
    if (tp + fp) == 0:
        precision = None
    else:
        precision = tp / (tp + fp)
    
    if (tp + fn) == 0:
        recall = None
    else:
        recall = tp / (tp + fn)
    
    return precision, recall

def mAUC(y_preds, y_targets, n_iters=10):
    'Return mean precision recall auc across valid labels'
    ap_rec = []
    # Calc avg precision score one label at a time
    for lab in range(y_targets.shape[1]):
        y_pred = y_preds[:, lab]
        y_target = y_targets[:, lab]
        
        prec_rec = []
        for thresh in np.linspace(0, 0.9, n_iters):
            prec, rec = calc_prec_rec(y_pred, y_target, thresh)
            if prec is None or rec is None:
                continue
            prec_rec.append([prec, rec])
        if len(prec_rec) <= 1:
            continue
        # Extract precision recall for given thresholds and estimate auc
        prec_rec = np.array(prec_rec)
        prec, rec = prec_rec[:, 0], prec_rec[:, 1]
        ap = auc(rec, prec)
        ap_rec.append(ap)
    if len(ap_rec) == 0:
        return np.nan
    mean_AP = mean(ap_rec)
    return mean_AP

def skl_mAP(y_preds, y_targets):
    'Return sklearn mean average precision across valid labels'
    ap_rec = []
    # Calc avg precision score one label at a time
    for lab in range(y_targets.shape[1]):
        y_pred = y_preds[:, lab]
        y_target = y_targets[:, lab]
        # If no targets present, skip label to avoid /0 runtime warning
        if y_target.sum() == 0:
            continue
        ap = average_precision_score(y_target, y_pred)
        ap_rec.append(ap)
    if len(ap_rec) == 0:
        return np.nan
    mean_AP = mean(ap_rec)
    return mean_AP

def mF1():
    pass
