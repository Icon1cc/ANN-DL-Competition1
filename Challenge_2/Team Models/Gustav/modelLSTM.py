import os
import numpy as np
import tensorflow as tf

num_categories = 6
seq_length = 128        # predictions based on previous seq_length data entries
forecast_length = 9     # predicting forecast_length time steps into the future (9 for Phase 1, 18 for Phase 2)
sample_length = seq_length + forecast_length

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):

        X = X[:, -seq_length:]

        out = self.model.predict(X)  # shape forecast_length
        return out