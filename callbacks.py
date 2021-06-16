import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class LearningRateExponentialDecay(Callback):
    def __init__(self, init_learning_rate, decay_rate, decay_steps):
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = 0

    def on_batch_begin(self, batch, logs=None):
        actual_lr = float(K.get_value(self.model.optimizer.lr))
        decayed_learning_rate = actual_lr * (self.decay_rate**(1.0 / self.decay_steps))
        K.set_value(self.model.optimizer.lr, decayed_learning_rate)
        self.global_step += 1