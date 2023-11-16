import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


class model:
    def __init__(self, path = ""):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(preprocess_input(X))
        out = tf.argmax(out, axis=-1)  # Shape [BS]

        return out
    
