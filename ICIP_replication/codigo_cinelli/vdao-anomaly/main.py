import argparse
import os
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K

import archs
import datasets.hdf5_vdao as vdao
import metrics
import utils
from archs.networks import optimizers
from datasets.hdf5_vdao import VDAO

np.random.seed(0)
tf.set_random_seed(0)

arch_names = sorted(name for name in archs.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(archs.__dict__[name]))


def _common(args, mode, validation=False, val_ratio=0, aloi_file=None, **kwargs):
    data_mode = 'test' if mode == 'test' else 'train'
    database = vdao.VDAO(args.dataset_dir, args.file, mode=data_mode,
                         val_set=validation, val_ratio=val_ratio,
                         aloi_file=aloi_file)

    # Set tensorflow session configurations
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ''
    K.set_session(tf.Session(config=config))
    print('save results: {}'.format(args.save_dir))

    # Useful metrics to record
    metrics_list = [metrics.fnr, metrics.fpr, metrics.distance, metrics.f1,
                    metrics.tp, metrics.tn, metrics.fp, metrics.fn]
    meters = {func.__name__: func for func in metrics_list}

    thresholds = kwargs.pop('thresholds', 0.5)
    arch = archs.__dict__[args.arch.lower()]
    arch_params = utils.parse_kwparams(args.arch_params)

    logger = {}
    # Apply func to data comming from all specified layers
    for layer in VDAO.LAYER_NAME:
        print('layer: {}'.format(layer))
        database.set_layer(layer)
        cross_history = utils.History()
        outputs = []

        roc = metrics.ROC() if validation is True else None

        # Apply func to each partition of the data
        for group_idx, (samples, set_size) in enumerate(
                database.load_generator(**utils.parse_kwparams(args.cv_params))):

            # Load old model or create a new one
            if args.load_model is not None:
                try:
                    model = arch(load_path=args.load_model, save_path=args.save_dir,
                                 layer=layer, group_idx=group_idx)
                except FileNotFoundError:
                    print('file not found for layer {}'.format(layer))
                    continue
            else:
                model = arch(load_path=args.load_model, save_path=args.save_dir,
                             layer=layer, group_idx=group_idx,
                             #  input_shape=samples[0][0].shape[1:],
                             input_shape=next(iter(samples.values()))[
                                 0].shape[1:],
                             weight_decay=args.weight_decay, **arch_params)

            if mode == 'train':
                output = _train(args, model, samples, set_size, meters,
                                cross_history,  roc=roc)
                print('\nFinished training {}'.format(group_idx+1))
            else:
                if type(thresholds) is dict:
                    group_thresholds = thresholds[layer][group_idx]
                else:
                    group_thresholds = thresholds

                output = _eval(args, model, samples, set_size[data_mode], meters,
                               threshold=group_thresholds)

            outputs += [output]

        if mode == 'train':
            logger[layer] = {'history': cross_history}
            if roc is not None:
                logger[layer].update({'roc': roc})
        else:
            logger[layer] = {'output': outputs}

        print('\n' + '* ' * 80 + '\n\n')

    return logger


def _eval(args, model, test_samples, set_size, meters, threshold=0.5):
    measures = []
    preds = []
    set_size = np.hstack(
        (np.zeros(1, dtype='int64'), np.cumsum(set_size)))
    (data, labels), vid_names = test_samples

    try:
        model.set_batch(args.batch_size)
    except AttributeError:
        pass
    for start, stop in zip(set_size[:-1], set_size[1:]):
        measurements, _, predictions = _evaluate(
            model, (data[start:stop], labels[start:stop]), meters,
            tune_threshold=False, batch_size=args.batch_size,
            thresholds=threshold, mode='test', verbose=0)
        predictions['predictions'] = (predictions['probas'] >
                                      threshold).astype(predictions['labels'].dtype)
        measures += [measurements]
        preds += [predictions]
    return {'meters': measures, 'nb_vids': len(set_size)-1,
            'results': preds, 'video_names': vid_names}


def eval(args):
    checkpoint = pickle.load(open(os.path.join(
        args.load_model, 'summary.pkl'), 'rb'))
    thresholds = {}
    if args.optim_thres is True:
        for key, val in checkpoint.items():
            if key == 'config':
                continue
            try:
                thres = val['history'].history['val']['thresholds']
                # thres = val['val']['thresholds']
                thresholds.update({key: np.asarray(thres)[:, 1]})
            except KeyError:
                print('No optimized threshold found')
                thresholds.update({key: 0.5})
    else:
        thresholds = 0.5

    logger = _common(args, 'test', thresholds=thresholds)

    all_results = {}
    for layer, results in logger.items():
        res = pd.concat([pd.DataFrame(group['meters'])
                         for group in results['output']]).reset_index(drop=True)
        res = res.applymap(lambda x: x[0])
        all_results.update({layer: res})

    pd.concat(all_results, axis=1).to_csv(
        os.path.join(args.load_model, args.save_dir), index=False)
    # read w/ header: pd.read_csv('models/bb/test-results.csv', header=[0,1])


def predict(args):
    checkpoint = pickle.load(open(os.path.join(
        args.load_model, 'summary.pkl'), 'rb'))
    thresholds = {}
    if args.optim_thres is True:
        for key, val in checkpoint.items():
            if key == 'config':
                continue
            try:
                thres = val['history'].history['val']['thresholds']
                thresholds.update({key: np.asarray(thres)[:, 1]})
            except KeyError:
                print('No optimized threshold found')
                thresholds.update({key: 0.5})
    else:
        thresholds = 0.5
    mode = 'pred' if args.train_file is True else 'test'
    logger = _common(args, mode, thresholds=thresholds)

    all_results = {}
    for layer, results in logger.items():
        res = pd.concat([pd.DataFrame(group['results'],
                                      index=group['video_names']
                                      )
                         for group in results['output']])
        all_results.update({layer: res})

    df = pd.concat(all_results, axis=1)
    df.to_pickle(os.path.join(args.load_model, args.save_dir + '.pkl'))
    df.to_csv(os.path.join(args.load_model, args.save_dir + '.csv'), index=True)
    # read w/ header: pd.read_csv('models/bb/test-results.csv', header=[0,1])


def _train(args, model, samples, set_size, meters, cross_history, roc=None, **kwargs):

    # train_samples, val_samples = samples
    train_samples = samples.pop('train', None)
    val_samples = samples.pop('val', None)
    for mode, size in set_size.items():
        if size == 0:
            continue
        cross_history.update('nb_vids', size, mode=mode)

    if hasattr(model, 'compile'):
        model.set_batch(args.batch_size)
        model.set_epochs(args.epochs)
        model.compile(args.optim, args.lr, args.lr_span, args.lr_factor,
                      ['accuracy',
                       metrics.FalseNegRate(),
                       metrics.FalsePosRate(),
                       metrics.Distance(),
                       metrics.FBetaScore(beta=1),
                       metrics.TruePos(),
                       metrics.TrueNeg(),
                       metrics.FalsePos(),
                       metrics.FalseNeg()])

    # Train the model
    history = model.fit(X=train_samples[0], y=train_samples[1],
                        val_data=val_samples)

    if hasattr(history, 'history'):
        # TODO: log the best value and not the latest
        for name, meter in history.history.items():
            mode = 'val' if name.startswith('val') else 'train'
            cross_history.update(name, meter[-1], mode)

    if val_samples is not None:
        measurements, thres_vals, _ = _evaluate(
            model, (val_samples, set_size['val']), meters, mode='val',
            batch_size=args.batch_size, tune_threshold=True, roc=roc)

        for name, measures in measurements.items():
            cross_history.update(
                'pp_' + name, [measure for measure in measures], mode='val')

        cross_history.update('thresholds', thres_vals, mode='val')
        return {'history': cross_history, 'roc': roc}

    return {'history': cross_history}


def _evaluate(model, data, meters, batch_size=32, verbose=1, mode='val',
              tune_threshold=False, thresholds=[0.5], roc=None, history=None):
    if mode not in ['val', 'test']:
        raise ValueError('mode should be either val or test')
    if mode is 'test' and tune_threshold is True:
        raise ValueError(
            'You cannot tune the threshold on the test data silly, that is cheating')
    if tune_threshold is True and (roc is None or isinstance(roc, metrics.ROC) is False):
        # Optionally I can create a metrics.ROC object here
        raise ValueError(
            'roc should be an instance of metrics.ROC to optimize threshold')
    if history is not None and isinstance(history, utils.History) is False:
        raise ValueError('history should be an instance of utils.History')

    try:
        (samples, labels), set_size = data
    except ValueError:
        samples, labels = data
        set_size = None

    probas_ = model.predict_proba(samples).squeeze()

    # Compute ROC curve and find thresold that minimizes dist
    if tune_threshold:
        best_threshold, _ = roc(labels, probas_)
        thresholds = thresholds + [best_threshold]

    # Evaluate on VAL set (threshold @ 50%. @ `best_threshold`)
    meters_val = metrics.compose(meters.values(), (labels, probas_),
                                 threshold=thresholds)
    if history:
        for name, measures in zip(meters.keys(), meters_val):
            history.update(
                'pp_' + name, [measure for measure in measures], mode=mode)
        # history.update('thresholds', thresholds, mode=mode)

    if verbose:
        msg = ['\n{}:: thresholds @ {}\n'.format(mode.upper(), thresholds)]
        msg += ['{0}: {1}\n'.format(name, measures) for name, measures in
                zip(meters.keys(), meters_val)]
        print(''.join(msg))

        # utils.print_result(history.history[mode],
        #     '{}:: thresholds @ {}'.format(mode.upper(), thresholds), exclude=['thresholds'])

    return {key: val for key, val in zip(meters.keys(), meters_val)},\
        thresholds, {'probas': probas_, 'labels': labels}


def train(args):
    aloi_file = args.aloi_file
    val_ratio = args.val_ratio
    logger = _common(args, 'train', validation=args.val_roc, aloi_file=aloi_file,
                     val_ratio=val_ratio)
    for layer in logger.keys():
        logger[layer]['history'].averages(weights_key='nb_vids',
                                          exclude=['thresholds'])
    # Compute ROC if data available
    if args.val_roc is True:
        roc_stats = {'tprs': {}, 'auc': {}}
        plt.figure('all_layers')
        for layer, results in logger.items():
            roc = results['roc']

            for name, stats in zip(('mean', 'std'), (roc.mean(), roc.std())):
                roc_stats['tprs'][name], roc_stats['auc'][name] = stats

            logger[layer]['roc'] = roc_stats
            roc.plot(os.path.join(os.path.join(args.save_dir, layer),
                                  'roc-crossval.eps'))

            # Plot mean ROC curve for each layer
            plt.figure('all_layers')
            auc_mean, auc_std = roc_stats['auc']['mean'], roc_stats['auc']['std']
            plt.plot(roc.mean_fpr, roc_stats['tprs']['mean'], lw=2, alpha=0.8,
                     label=r'ROC - {} (AUC = {:.2f} $\pm$ {:0.2f})'.format(
                layer.split('_branch')[0], auc_mean, auc_std))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1,
                 color='r', label='Identidade', alpha=.8)
        metrics.ROC.label_plot()
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(os.path.join(args.save_dir, 'mean-roc.eps'),
                    bbox_inches='tight')
        plt.close()

    # Save history
    del args.func
    logger['config'] = vars(args)
    utils.save_data(logger, os.path.join(args.save_dir, 'summary'),
                    json_format=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection in VDAO')
    subparsers = parser.add_subparsers()
    # Loaders
    parser.add_argument(
        '--dataset-dir',
        '--dir',
        default='/home/bruno.afonso/datasets/article_HDF5',
        type=str,
        metavar='PATH',
        help='The path to the dataset dir')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=128,
        type=int,
        metavar='N',
        help='mini-batch size (default: 32)')
    # Optimizer
    parser.add_argument(
        '--optim',
        default='adamax',
        type=str,
        help='optimizers: ' + ' | '.join(sorted(optimizers.keys())) +
        ' (default: adamax)')
    parser.add_argument(
        '--save',
        dest='save_dir',
        type=str,
        default='models/',
        help='path to save results to')
    parser.add_argument(
        '--file',
        type=str,
        metavar='PATH',
        # default='59_videos_test_batch.h5', -- test
        # default='train_batch_VDAO.h5',     -- train
        help='Path to hdf5 data file')
    parser.add_argument(
        '--load',
        dest='load_model',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to trained model dir')

    # Train parser
    tr_parser = subparsers.add_parser('train', help='Network training')

    tr_parser.add_argument(
        '--epochs',
        default=20,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    tr_parser.add_argument(
        '--val-roc',
        action='store_true',
        help='Whether to split train set on train/val')
    tr_parser.add_argument(
        '--val-ratio',
        default=0.1,
        type=float,
        metavar='N',
        help='Val set size relative to train size (ratio)')
    tr_parser.add_argument(
        '--aloi-file',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to ALOI-augmented imgs file'
    )
    parser.add_argument(
        '--cv-params',
        metavar='PARAMS',
        default=[],
        nargs='+',
        type=str,
        help='cross validation params, methods: ' + ' | '.join(
            list(vdao.group_fetching.keys())) + ' (default method: leave_one_out)'
    )

    # Architecture
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='mlp',
        choices=arch_names,
        help='model architecture: ' + ' | '.join(arch_names) +
        ' (default: mlp)')
    parser.add_argument(
        '--arch-params',
        metavar='PARAMS',
        default=[],
        nargs='+',
        type=str,
        help='model architecture params')

    # Hyperparameters
    tr_parser.add_argument(
        '--lr',
        '--learning-rate',
        default=2e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    tr_parser.add_argument(
        '--lr-factor',
        '--learning-decay',
        default=1,
        type=float,
        metavar='LRF',
        help='learning rate decay factor')
    tr_parser.add_argument(
        '--lr-span',
        '--lr-time',
        default=10,
        type=float,
        metavar='LRS',
        help='time span for each learning rate step')
    tr_parser.add_argument(
        '--weight-decay',
        '--wd',
        default=0,
        type=float,
        metavar='W',
        help='weight decay (default: 0)')
    tr_parser.set_defaults(func=train)

    # Eval parser
    eval_parser = subparsers.add_parser('eval', help='Network training')
    eval_parser.add_argument(
        '--optim-thres',
        action='store_true',
        help='whether to evaluate data using optimized thresholds or 0.5')
    eval_parser.set_defaults(func=eval)

    # Predict parser
    pred_parser = subparsers.add_parser('predict', help='Network training')
    pred_parser.add_argument(
        '--optim-thres',
        action='store_true',
        help='whether to evaluate data using optimized thresholds or 0.5')
    pred_parser.add_argument(
        '--train-file',
        action='store_true',
        help='whether a file with the train file specs is being used')
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    print(args)
    args.func(args)
