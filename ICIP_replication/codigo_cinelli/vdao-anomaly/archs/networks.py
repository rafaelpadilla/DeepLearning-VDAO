import os
import glob

import keras.models
from keras.layers import Activation, Dense, Flatten
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam, Adamax
from keras import callbacks

__all__ = ['mlp']

optimizers = {'adam': Adam, 'adamax': Adamax, 'sgd': SGD}


class MLP(object):
    """ Multilayer Perceptron Classifier (Fully-Connected network)
    """

    def __init__(self, load_path=None, layer=None, group_idx=None,
                 input_shape=None, nb_neurons=None, weight_decay=0,
                 save_path=None):
        """
        Keyword Arguments:
            load_path {str} -- The path from which to load the models (default: {None})
            layer {str} -- The layer from which features were extracted (default: {None})
            group_idx {int} -- The data fold being used  (default: {None})
            input_shape {list} -- [description] (default: {None})
            nb_neurons {int} -- [description] (default: {None})
            weight_decay {int} -- [description] (default: {0})
            save_path {str} -- The target dir where the model should be saved (default: {None})
        """
        self.group_idx = group_idx
        self.subdir = os.path.join(save_path, layer)
        if load_path is not None:
            self.load(load_path, layer)
        else:
            self.create(input_shape, nb_neurons, weight_decay=weight_decay)

    def load(self, load_path, layer):

        model_name = os.path.join(
            load_path, layer, 'model.test{:02d}-ep'.format(self.group_idx))
        model_name = glob.glob(model_name + '*')[-1]

        self.model = keras.models.load_model(model_name, compile=False)

    def create(self, input_shape, nb_neurons, weight_decay=0):
        """ Creates an instance of keras Sequential class with the specified
        number of layer, each one consting of a Fully-Connnected and Relu
        activation. The last activation layer is a Sigmoid.

        Arguments:
            input_shape {list} -- Size of the tensor at the input of the model
            nb_neurons {iterable} -- iterable containing the nb of neurons on
                each layer of the network

        Keyword Arguments:
            weight_decay {int} -- The L2 regularization weight for the loss (default: {0})
        """
        classifier = Sequential()
        classifier.add(Flatten(input_shape=input_shape))

        for idx, neurons_in_layer in enumerate(nb_neurons):
            classifier.add(Dense(neurons_in_layer,
                                 name='Dense_feat_{}'.format(idx),
                                 kernel_regularizer=l2(weight_decay)))
            classifier.add(Activation('relu'))

        classifier.add(Dense(1, name='Dense_feat',
                             kernel_regularizer=l2(weight_decay)))
        classifier.add(Activation('sigmoid'))

        self.model = classifier

    def set_callbacks(self, lr_span, lr_factor):
        """Set useful periodic Keras callbacks:
        `LearningRateScheduler` updates the lr at the end of each epoch
        `ModelCheckpoint` saves model at the end of each epoch (if conditions are met)
        `CSVLogger` writes results to a csv file

        Arguments:
            lr_span {int} -- The number of epoch to wait until changing lr
            lr_factor {float} -- By how much to modify the lr
        """
        if not os.path.exists(self.subdir):
            os.makedirs(self.subdir)

        # learning rate schedule
        def schedule(epoch, lr): return lr * lr_factor**(epoch // lr_span)
        lr_scheduler = callbacks.LearningRateScheduler(schedule)

        # Should I monitor here the best val_loss or the metrics of interest?
        # If not all samples are used in an epoch val_loss is noisy
        checkpointer = callbacks.ModelCheckpoint(
            os.path.join(self.subdir,
                         'model.test{:02d}-ep{{epoch:02d}}.pth'.format(
                             self.group_idx)),
            monitor='val_loss', save_best_only=True, mode='min')

        csv_logger = callbacks.CSVLogger(
            os.path.join(self.subdir,
                         'training.test{:02d}.log'.format(self.group_idx)))

        self.callbacks = [lr_scheduler, csv_logger, checkpointer]

    def compile(self, optim, lr, lr_span, lr_factor, metrics,
                loss='binary_crossentropy'):
        optimizer = optimizers[optim.lower()](lr=lr)
        self.set_callbacks(lr_span, lr_factor)
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)

    def fit(self, X, y, val_data=None):
        history = self.model.fit(x=X,
                                 y=y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 initial_epoch=self.init_epoch,
                                 shuffle=True,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 validation_data=val_data
                                 )
        return history

    def predict_proba(self, samples):
        return self.model.predict_proba(samples, batch_size=self.batch_size,
                                        verbose=0)

    def set_batch(self, batch_size):
        """Set the mini-batch size

        Arguments:
            batch_size {int} -- Mini-batch size
        """
        self.batch_size = batch_size

    def set_epochs(self, nb_epochs, start_from=0):
        """Set the number of epochs to be trained on

        Arguments:
            nb_epochs {int} -- The total number of epochs desired

        Keyword Arguments:
            start_from {int} -- From which epoch to start training.
                Useful on retraining (default: {0})
        """
        self.epochs = nb_epochs
        self.init_epoch = start_from


def mlp(load_path=None,  save_path=None, layer=None, group_idx=None, **kwargs):
    """Instantiates a Sequential keras model containing the fully connected net

    Keyword Arguments:
        load_path {str} -- The path from which to load the models (default: {None})
        save_path {str} -- The target dir where the model should be saved (default: {None})
        layer {str} -- The layer from which features were extracted (default: {None})
        group_idx {int} -- The data fold being used  (default: {None})

    Returns:
        keras.model.Sequential -- The specified dense network
    """
    input_shape = kwargs.pop('input_shape', None)
    nb_neurons = kwargs.pop('nb_neurons', None)
    weight_decay = kwargs.pop('weight_decay', None)

    classifier = MLP(load_path=load_path, layer=layer, group_idx=group_idx,
                     input_shape=input_shape, nb_neurons=nb_neurons,
                     weight_decay=weight_decay, save_path=save_path)
    return classifier
