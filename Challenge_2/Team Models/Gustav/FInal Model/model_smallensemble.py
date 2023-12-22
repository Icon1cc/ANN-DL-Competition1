import os
import numpy as np
import tensorflow as tf

num_categories = 6
seq_length = 128        # predictions based on previous seq_length data entries
forecast_length = 9     # predicting forecast_length time steps into the future (9 for Phase 1, 18 for Phase 2)
sample_length = seq_length + forecast_length          # predictions based on previous seq_length data entries
telescope = 3

    
class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'LSTM_v1'))

    def predict(self, X, features):
        X_ = X[:, -seq_length:]
        X_1 = self.model.predict(X_)  # shape forecast_length
        X_ = np.concatenate((X_[:, telescope:],X_1), axis=1)
        X_2 = self.model.predict(X_)
        X_ = np.concatenate((X_[:, telescope:],X_2), axis=1)
        X_3 = self.model.predict(X_)

        out = np.concatenate((X_1, X_2, X_3), axis=1)
        return out

