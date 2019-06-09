import json
import os
from itertools import product

import h5py
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     LeaveOneGroupOut, LeavePGroupsOut,
                                     train_test_split)

class ManualGroupSplit(object):
    """ Manually defines a cross-validation group split specified by a .json
    file containing. Compatibility class due to Bruno's group partitioning.
    """

    def __init__(self, filename=None, n_splits=None):
        """
        Keyword Arguments:
            filename {string} -- Path to file containing the splits and their
                videos (default: {None})
            n_splits {int} -- Number of different groups to consider
                (default: {None})

        Raises:
            ValueError -- Incorrect number of splits requested (max: 17)
        """
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'r') as fp:
            self.test_folds = json.load(fp)

        self.nb_groups = n_splits or len(self.test_folds)
        if self.nb_groups > len(self.test_folds):
            raise ValueError('n_splits should be not greater than {}'.format(
                len(self.test_folds)))

    def split(self, X, y, groups):
        seen_vids = []
        for test_idx in self.test_folds[:self.nb_groups]:

            test_groups = [groups[idx] for idx in test_idx]
            train_idx = [idx for idx, obj in enumerate(
                groups) if obj not in test_groups]

            test_idx = [idx for idx in test_idx if idx not in seen_vids]
            seen_vids += test_idx

            yield train_idx, test_idx


group_fetching = {'k_fold': GroupKFold,
                  'leave_p_out': LeavePGroupsOut,
                  'leave_one_out': LeaveOneGroupOut,
                  'shuffle_split': GroupShuffleSplit,
                  'manual': ManualGroupSplit
                  }


class VDAO(object):
    """Loader for ResNet50 (Keras pretrained model) features extracted from the
     Video Database of Abandoned Objects (VDAO) in a Cluttered Industrial
     Environment. Features from 59 videos across 17 different layers are
     available.
    """

    # List of available layers from which to get the output feature maps
    LAYER_NAME = [
        'res2a', 'res2b', 'res2c', 'res3a', 'res3b', 'res3c', 'res3d', 'res4a',
        'res4b', 'res4c', 'res4d', 'res4e', 'res4f', 'res5a', 'res5b', 'res5c',
        'avg_pool'
    ]
    LAYER_NAME = [
        name + '_branch2a' for name in LAYER_NAME if name.startswith('res')
    ]

    # Specifies which object appears in each one of the 59 vids (in order)
    VIDS_OBJS = [
        ['Dark-Blue_Box'] * 2, ['Shoe'], ['Camera_Box'], ['Towel'],
        ['White_Jar'], ['Pink_Bottle'], ['Shoe'] * 2, ['Dark-Blue_Box'] * 2,
        ['Camera_Box'] * 2, ['White_Jar'] * 3, ['Brown_Box'] * 3,
        ['Pink_Bottle'] * 3, ['Towel'] * 2, ['Black_Coat'] * 3,
        ['Black_Backpack'] * 6, ['Shoe'] * 3,
        ['Dark-Blue_Box'] * 2, ['Camera_Box'] * 3,
        ['White_Jar'] * 2, ['Brown_Box'] * 3, ['Pink_Bottle'] * 3,
        ['Towel'] * 3, ['Black_Coat'] * 3, ['Black_Backpack'] * 4
    ]
    VIDS_OBJS = [item for sublist in VIDS_OBJS for item in sublist]

    # Specifies the (two) possible illumination settings available for each vid
    VDAO_ILLU = ['NORMAL-Light', 'EXTRA-Light']

    # Specifies among the (three) available positions for each object
    VDAO_POS = ['POS1', 'POS2', 'POS3']

    # Lists all different objects in dataset
    # (each multiple obj vid is considered as a diff obj since those objs are
    # distinct from the one in the single obj vids)
    VDAO_OBJS = set([
        name for name in VIDS_OBJS + ['Mult_Objs1', 'Mult_Objs2', 'Mult_Objs3']
    ])

    def __init__(self, dataset_dir, filename, val_ratio=0,
                 aloi_file=None, val_set=True, mode='train'):
        """
        Arguments:
            dataset_dir {str} -- name of the dir containing the HDF5 files
            filename {str} -- name of the HDF5 file desired

        Keyword Arguments:
            val_ratio {int} -- Ratio ([0,1]) of samples to be used as validation
                (default: {0})
            aloi_file {str} -- name of the HDF5 cointaining the artificial
                samples (ALOI). Necessary only if val_set is `True` (default:
                {None})
            val_set {bool} -- Whether to use a separate validation set
                (default: {True})
            mode {str} -- Whether to get the train/test data (default: {'train'})
        """
        self.dataset_dir = dataset_dir
        self.files = {mode: filename, 'aloi': aloi_file}
        self.data = {mode: None, 'aloi': None}
        if mode == 'train':
            self.data['val'] = None

        self.train_entries = list(product(
            self.VDAO_ILLU, self.VDAO_OBJS, self.VDAO_POS))
        self.val_ratio = val_ratio
        self.val_set = val_set
        self.mode = mode

    def set_layer(self, layer_name):
        self.data = {key: None for key in self.data.keys()}
        self.out_layer = layer_name

    def load_generator(self,
                       method='leave_one_out',
                       **kwargs):
        """Generator over the folds created by the method chosen in `group_fetching`

        Keyword Arguments:
            method {str} -- Which method from `group_fetching` dict to use
                (default: {'leave_one_out'})

        Returns:
            If self.mode is 'test'
                tuple -- data and label pairs for the test set, name of test vids
                dict -- size (value) of the test set (key)
            Else
                dict -- data and label pairs (value) of each returned set (key)
                dict -- size (value) of each returned set (key)
        """
        x = np.arange(len(self.VIDS_OBJS))
        fold_method = group_fetching[method](**kwargs)

        if self.files['aloi'] is not None:
            self.data['aloi'], aloi_size = _loadFile(
                os.path.join(self.dataset_dir, self.files['aloi']),
                self.out_layer, [''])

        for train_vid_idx, test_vid_idx in fold_method.split(x, x, self.VIDS_OBJS):
            test_objs = [self.VIDS_OBJS[vid] for vid in test_vid_idx]
            test_vids = [['video{}'.format(vid+1)] for vid in test_vid_idx]
            train_vids = list(product(self.VDAO_ILLU, set(
                [self.VIDS_OBJS[idx] for idx in train_vid_idx]), self.VDAO_POS))

            if self.mode == 'test':
                self.data['test'], test_size, test_vid_name = _loadFile(
                    os.path.join(self.dataset_dir, self.files['test']),
                    self.out_layer, test_vids)
                yield (self.data['test'], test_vid_name), {'test': test_size}

            else:
                self.data['val'], val_size = None, 0
                if self.val_set is True:
                    train_vids, val_vids = self.split_validation(
                        data=train_vids, random_state=0)

                    self.data['val'], val_size, _  = _loadFile(
                        os.path.join(self.dataset_dir, self.files['train']),
                        self.out_layer, val_vids, test_objs)

                self.data['train'], train_size, _  = _loadFile(
                    os.path.join(self.dataset_dir, self.files['train']),
                    self.out_layer, train_vids, test_objs)

                if self.val_set is True:
                    yield {'train': _merge_datasets(
                                    (self.data['train'], self.data['aloi'])),
                           'val': self.data['val']}, \
                          {'train': train_size, 'val': val_size}
                else:
                    yield {'train': _merge_datasets(
                                    (self.data['train'], self.data['aloi']))}, \
                          {'train': train_size}

    def split_validation(self,
                         ratio=None,
                         data=None,
                         labels=None,
                         random_state=None):
        """Split the training data into two separate disjoint sets: train and
            val sets. The separation is done on the entire videos. However,
            the same obj may appear both in train and val sets.

        Keyword Arguments:
            ratio {int} -- [description] (default: {None})
            data {nd.array} -- [description] (default: {None})
            labels {nd.array} -- [description] (default: {None})
            random_state {int} -- The seed for the pseudorandom generator (default: {None})

        Returns:
            lists -- List containing train-test split of inputs.
        """
        extra_data = None
        if ratio is None:
            ratio = self.val_ratio

        if data is None:
            data = np.asarray(self.train_entries)

        x_train, x_val, _, _ = train_test_split(
            data, np.ones(len(data)), test_size=ratio, random_state=random_state)

        return x_train, x_val


def _merge_datasets(datasets):
    """Merge (concatenates along samples dimension) together datasets
    (data, label).

    Arguments:
        datasets {list} -- Contains (data, label) elements to be merged together

    Returns:
        A single dataset tuple (data, label) consisting of all input
        datasets in the same exact order.
    """
    return tuple(np.concatenate(data) for data in zip(*tuple(filter(None, datasets))))


def _loadFile(basepath, out_layer, vid_name_iter, exceptions=[], verbose=1):
    """Loads data from within the HDF5 file specified by 'basepath' relative
    to the feature maps of the videos 'vid_name_iter' obtained from layer
    'out_layer'. It's possible to consider forbidden substrigs defined by
    'exceptions' such that any data whose path contains such substrings is not
    loaded.

    Arguments:
        basepath {string} -- Name of the HDF5 from which to get the data
        out_layer {string} -- Name of the layer from which to get the output
            feature maps
        vid_name_iter {iterable} -- Name of the videos from which to get the data

    Keyword Arguments:
        exceptions {list} -- Vids containing such substr should be skipped
            (default: {[]})
        verbose {int} -- The level of detail to print while processing
            (default: {1})

    Raises:
        NameError -- If mode (train/test) cannot be inferred from `basepath`
            then name is incorrect
        KeyError -- If requested video is not found (possible only in train mode)

    Returns:
        nd.array -- data
        nd.array -- labels
    """
    if 'train' in basepath:
        mode = 'train'
    elif 'test' in basepath:
        mode = 'test'
    else:
        raise NameError('Could not determine mode (\'train\'/\'test\') from '
                        'HDF5 filename \'{}\''.format(basepath))

    if verbose > 1:
        print('{} SET...'.format(mode.upper()))
        print('exceptions:{}\n'.format(exceptions))

    h5_file = h5py.File(basepath, 'r')

    data = []
    labels = []
    vid_name = []
    for items in vid_name_iter:
        if not isinstance(items, list):
            items = list(items)
        filepath = '_'.join(items + [out_layer])
        filepath = filepath + '_{{}}_{}_SET'.format(mode.upper())
        if any(substr in filepath for substr in exceptions):
            continue
        try:
            vid_name.append(filepath)
            data.append(h5_file[filepath.format('X')].value)
            labels.append(h5_file[filepath.format('y')].value)
        except KeyError as exception:
            if mode is 'test':
                raise exception

    set_size = [vid_labels.shape[0] for vid_labels in labels]
    data = np.concatenate(data)
    labels = np.concatenate(labels).astype(int)

    h5_file.close()

    return (data, labels), set_size, [name.split('_')[0] for name in vid_name]
