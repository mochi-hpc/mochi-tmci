from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from . import ops


class CheckpointCallback(Callback):
    """Generic TMCI checkpoint callback class."""

    def __init__(self, backend, config="", frequency={'epoch': 1}, include_optimizer=True):
        if not (isinstance(backend, str)):
            raise TypeError("backend should be a string")
        if not (isinstance(config, str)):
            raise TypeError("config should be a string")
        self.__backend = backend
        self.__config = config
        self.__frequency = frequency
        self.__save_op = None
        self.__load_op = None

    def on_train_begin(self, logs={}):
        self.__tensors = []
        for l in self.model.layers:
            for w in l.weights:
                self.__tensors.append(w)
        if self.include_optimizer:
            for w in self.model.optimizer.weights:
                self.__tensors.append(w)
        self.__save_op = ops.checkpoint(
                backend=self.__backend,
                config=self.__config,
                tensors=self.__tensors)
    
    def on_train_end(self, logs={}):
        del self.__save_op
        del self.__tensors

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if 'epoch' in self.__frequency:
            if epoch % self.__frequency['epoch'] == 0:
                self.__save_op.run()

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if 'batch' in self.__frequency:
            if epoch % self.__frequency['batch'] == 0:
                self.__save_op.run()

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
    save_op = ops.checkpoint(
                backend=backend,
                config=config,
                tensors=tensors)
    save_op.run()

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
    load_op = ops.restore(
                backend=backend,
                config=config,
                tensors=tensors)
    load_op.run()
