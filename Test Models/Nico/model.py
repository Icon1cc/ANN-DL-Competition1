import os
import numpy as np
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
        out = self.model.predict(X)         # Predict => [p('healthy'), p('unhealthy')]
        out = tf.argmax(out, axis=-1)       # Return max probable class: 0 for 'healthy', 1 for 'unhealthy'

        return out