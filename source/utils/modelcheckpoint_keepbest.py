from keras.callbacks import Callback
from keras.models import clone_model
import numpy as np
import copy

class modelcheckpoint_keepbest(Callback):
    """record the model after every epoch.
    """

    def __init__(self, monitor='val_loss', verbose=0,                 
                 mode='auto', period=1):
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.ever_best_model=None
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('modelcheckpoint_keepbest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def get_best_mode(self):
        return self.ever_best_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
      
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                                'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f'
                                ' keep model'
                                % (epoch + 1, self.monitor, self.best,current))
                    self.best = current
                    self.ever_best_model = clone_model(self.model)
                    self.ever_best_model.set_weights(self.model.get_weights())
            


