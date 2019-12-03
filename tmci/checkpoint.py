import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
from . import ops


def save_weights(model, backend, config="", 
        include_optimizer=True):
    if not (isinstance(backend, str)):
        raise TypeError("backend should be a string")
    if not (isinstance(config, str)):
        raise TypeError("config should be a string")
    tensors = []
    for l in model.layers:
        for w in l.weights:
            tensors.append(w)
    if include_optimizer:
        for w in model.optimizer.weights:
            tensors.append(w)
    ops.checkpoint(backend=backend,
                   config=config,
                   tensors=tensors)

def load_weights(model, backend, config="",
        include_optimizer=True):
    if not (isinstance(backend, str)):
        raise TypeError("backend should be a string")
    if not (isinstance(config, str)):
        raise TypeError("config should be a string")
    tensors = []
    for l in model.layers:
        for w in l.weights:
            tensors.append(w)
    if include_optimizer:
        for w in model.optimizer.weights:
            tensors.append(w)
    ops.restore(backend=backend,
            config=config,
            tensors=tensors)

class CheckpointCallback(Callback):
    """Generic TMCI checkpoint callback class."""

    def __init__(self, backend, config="",
            frequency={'epoch': 1},
            include_optimizer=True):
        if not (isinstance(backend, str)):
            raise TypeError("backend should be a string")
        if not (isinstance(config, str)):
            raise TypeError("config should be a string")
        self.__backend = backend
        self.__config = config
        self.__frequency = frequency
        self.__include_optimizer = include_optimizer

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if 'epoch' in self.__frequency:
            if epoch % self.__frequency['epoch'] == 0:
                save_weights(self.model,
                        backend=self.__backend,
                        config=self.__config,
                        include_optimizer=self.__include_optimizer)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if 'batch' in self.__frequency:
            if epoch % self.__frequency['batch'] == 0:
                save_weights(self.model,
                        backend=self.__backend,
                        config=self.__config,
                        include_optimizer=self.__include_optimizer)

