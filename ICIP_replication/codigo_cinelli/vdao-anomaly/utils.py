import json
import pickle

import numpy as np

from sklearn.base import TransformerMixin

class Flatten(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.reshape(X.shape[0], -1)

class History(object):
    def __init__(self):
        self.history = {}

    def update(self, name, value, mode=None):
        if isinstance(value, dict):
            self.history[mode][name] = value
            return
        try:
            self.history[mode][name] += [value]
        except KeyError:
            try:
                self.history[mode][name] = [value]
            except KeyError:
                self.history[mode] = {name: [value]}

    def averages(self, weights_key, exclude=[]):
        exclude += [weights_key]
        for mode, history in self.history.items():
            weights = self.history[mode].get(weights_key)
            weights = [len(frames) for frames in weights]
            for name, meter in history.items():
                if name in exclude:
                    continue
                self.history[mode][name] += [
                    np.average(meter, weights=weights, axis=0)]


def print_result(history, title, exclude=[]):
    # print('\n{}'.format(title))
    msg = ['\n{}\n'.format(title)]
    msg += ['{0}: {1}\n'.format(name, meter[-1]) for name, meter in
            history.items() if name not in exclude]

    print(''.join(msg))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, History):
            return obj.history
        return json.JSONEncoder.default(self, obj)


def save_data(data, filename, json_format=True, pickle_format=True):
    if json_format:
        with open(filename + '.json', 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
    if pickle_format:
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def parse_kwparams(kwlst):
    '''
    Parses key-worded parameters.

    @param kwstr key-worded parameters list to be parsed.

    @return dictionary with the key-worded parameters.
    '''

    # Set in dictionary form
    kwparams = {}
    for param in kwlst:
        k, v = param.split('=')
        try:
            kwparams[k] = json.loads(v)
        except json.JSONDecodeError:
            kwparams[k] = v
    return kwparams

