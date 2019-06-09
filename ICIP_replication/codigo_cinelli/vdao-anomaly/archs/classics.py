import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from utils import Flatten
from sklearn.externals import joblib

import pdb

__all__ = ['randomforest']


class RandomForest(object):
    """ Random Forest Classifier
    """
    def __init__(self, model_path=None, layer=None, group_idx=None,
                 nb_trees=None, max_depth=None, bootstrap=True, oob_score=False,
                 save_path=None, nb_jobs=None):
        """
        Keyword Arguments:
            model_path {str} -- The path from which to load the models (default: {None})
            layer {str} -- The layer from which features were extracted (default: {None})
            group_idx {int} -- The data fold being used  (default: {None})
            nb_trees {int} -- The number of trees in the forest (default: {None})
            max_depth {int} -- The maximum depth of the tree. If None go as deep as possible (default: {None})
            bootstrap {bool} -- Whether bootstrap samples are used when building trees (default: {True})
            oob_score {bool} -- Whether to use out-of-bag samples to estimate the generalization accuracy (default: {False})
            save_path {str} -- The target dir where the model should be saved (default: {None})
            nb_jobs {int} -- The number of jobs to run in parallel for both fit and predict (default: {None})
        """

        self.group_idx = group_idx
        self.subdir = os.path.join(save_path, layer)
        if model_path is not None:
            self.load(model_path, layer, group_idx)
        else:
            self.create(nb_trees, max_depth, bootstrap, nb_jobs, oob_score)

    def load(self, model_path, layer, group_idx):
        model_name = os.path.join(
            model_path, layer, 'model.test{:02d}.pkl'.format(group_idx))
        self.model = joblib.load(model_name)

    def create(self, nb_trees, max_depth, bootstrap=True, nb_jobs=4, oob_score=False):
        """ Creates an instance of the sklearn ensemble model Random Forest.

        Arguments:
            nb_trees {int} -- The number of trees in the forest.
            max_depth {int} -- The maximum depth of the tree.

        Keyword Arguments:
            bootstrap {bool} -- Whether bootstrap samples are used when building
                trees. (default: {True})
            nb_jobs {int} -- The number of jobs to run in parallel for both fit
                and predict (default: {4})
            oob_score {bool} -- Whether to use out-of-bag samples to estimate
                the generalization accuracy. (default: {False})
        """
        classifier = RandomForestClassifier(
            n_estimators=nb_trees, max_depth=max_depth, max_features='auto',
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=nb_jobs, random_state=0)

        self.model = Pipeline([('flatten', Flatten()), ('forest', classifier)])

    def fit(self, X, y, val_data=None):
        self.model.fit(X, y)
        self.save_model()

    def predict_proba(self, samples):
        return self.model.predict_proba(samples)[:, 1]

    def save_model(self):
        if not os.path.exists(self.subdir):
            os.makedirs(self.subdir)

        filename = os.path.join(
            self.subdir, 'model.test{:02d}.pkl'.format(self.group_idx))
        joblib.dump(self.model, filename)


def randomforest(load_path=None,  save_path=None, layer=None, group_idx=None, **kwargs):
    """Instantiates a sklearn pipeline containing the desired Random Forest
     ensemble model

    Keyword Arguments:
        load_path {str} -- The path from which to load the models (default: {None})
        save_path {str} -- The target dir where the model should be saved (default: {None})
        layer {str} -- The layer from which features were extracted (default: {None})
        group_idx {int} -- The data fold being used  (default: {None})

    Returns:
        sklearn.pipeline.Pipeline -- Pipeline containing specified Random Forest
            (sklearn.ensemble.RandomForestClassifier)
    """

    nb_trees = kwargs.pop('nb_trees', None)
    max_depth = kwargs.pop('max_depth', None)
    bootstrap = kwargs.pop('bootstrap', True)
    oob_score = kwargs.pop('oob_score', False)
    nb_jobs = kwargs.pop('nb_jobs', 4)

    classifier = RandomForest(model_path=load_path, layer=layer, group_idx=group_idx,
                              nb_trees=nb_trees, max_depth=max_depth, bootstrap=bootstrap,
                              oob_score=oob_score, nb_jobs=nb_jobs, save_path=save_path)
    return classifier
