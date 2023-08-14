from tensorflow.keras import callbacks
import tensorflow as tf
import keras
import numpy as np 

class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_val_loss = np.Inf
        self.best_coverage = 0.95
        self.best_mpiw_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None): 
        #val_mpiw
        val_mpiw_loss = logs.get('val_mpiw')
        # val_loss
        val_loss=logs.get('val_loss')
        # val_coverage
        val_coverage = logs.get('val_coverage')

        # If BOTH the validation loss AND coverage(coverage < 0.95) does not improve for 'patience' epochs, stop training early.
        if np.less_equal(val_mpiw_loss, self.best_mpiw_loss) and np.greater_equal(val_coverage, self.best_coverage):
            #self.best_val_loss = val_loss
            self.best_mpiw_loss = val_mpiw_loss
            self.best_coverage = 0.95
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                
                if self.best_weights is not None:
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    #self.best_weights = self.model.get_weights()
                    self.model.set_weights(self.best_weights)
                else:
                    self.model.stop_training = True
                    print("PICP is not satified, Please check the lamda value.")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


