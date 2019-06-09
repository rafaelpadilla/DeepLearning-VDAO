import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix


# Metrics
def _fpr(tn, fp, fn, tp, eps=1e-12):
    return fp / (fp + tn + eps)

def _fnr(tn, fp, fn, tp, eps=1e-12):
    return fn/ (fn + tp + eps)

def _tpr(tn, fp, fn, tp, eps=1e-12):
    return 1 - fnr(tn, fp, fn, tp)

def _tnr(tn, fp, fn, tp, eps=1e-12):
    return 1 - fpr(tn, fp, fn, tp)

def _f1(tn, fp, fn, tp, eps=1e-12):
    return (2*tp)/(2*tp + fn + fp)

def _dis(tn, fp, fn, tp, eps=1e-12):
    return np.sqrt(_fpr(tn, fp, fn, tp, eps)**2 + _fnr(tn, fp, fn, tp, eps)**2)

def compute_results(args):
    # Load the prediction results
    preds = pd.read_pickle(args.file)

    if args.kernel_size > 1:
        for layer in preds.columns.levels[0]:
            preds.loc[:, (layer, 'predictions')] = \
                preds.loc[:, (layer, 'predictions')].agg(
                    medfilt, kernel_size=args.kernel_size)

    res_layer = {}
    for layer in preds.columns.levels[0]:
        predictions = preds.loc[:, (layer, 'predictions')]
        labels = preds.loc[:, (layer, 'labels')]
        vids = []

        for idx in range(len(predictions)):
            tn, fp, fn, tp = confusion_matrix(
                labels[idx], predictions[idx], labels=[0,1]).ravel()
            vids += [{meter: eval('_' + meter)(tn, fp, fn, tp) for meter in ['fpr', 'fnr', 'f1']}]

        res_layer[layer] = pd.concat([pd.Series(vid) for vid in vids],
                                     keys=preds.index, axis=1)

    final_res = pd.concat(res_layer, axis=0)

    # Compute avg results across videos in the (test) set
    mean_results = final_res.mean(axis=1).unstack()
    mean_results['dis'] = mean_results[['fpr', 'fnr']].apply(np.linalg.norm, axis=1)

    # Save results to file
    mean_results.to_csv(args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute final results for dataset')

    parser.add_argument(
        '--file',
        dest='file',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to the prediction results'
    )
    parser.add_argument(
        '--save',
        dest='save_dir',
        type=str,
        metavar='PATH',
        help='path to save the global metric results'
    )
    parser.add_argument(
        '--med-filter',
        dest='kernel_size',
        default=1,
        type=int,
        metavar='N',
        help='kernel size of the temporal median filter'
    )

    args = parser.parse_args()
    print(args)
    compute_results(args)
