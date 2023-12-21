import os
import numpy as np
import tensorflow as tf
import joblib


num_categories = 6
seq_length_LSTM = 128        # predictions based on previous seq_length data entries
forecast_length = 9     # predicting forecast_length time steps into the future (9 for Phase 1, 18 for Phase 2)
seq_length_GRU = 26         # predictions based on previous seq_length data entries
sample_length_GRU = seq_length_GRU + forecast_length
sample_length_LSTM = seq_length_LSTM + forecast_length
telescope = 9
    
class model:
    def __init__(self, path):
        self.model = (tf.keras.models.load_model(os.path.join(path, 'LSTMModel')),
                       tf.keras.models.load_model(os.path.join(path, 'GRUModel')),
                       tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/model')))
        self.rscaler_X = joblib.load(os.path.join(path, 'SubmissionModel/rscaler_X.save'))
        self.rscaler_y = joblib.load(os.path.join(path, 'SubmissionModel/rscaler_y.save'))


    def predict(self, X, features):
        X_LSTM = X[:, -seq_length_LSTM:]
        X_GRU = X[:, -seq_length_GRU:]
        X_1 = self.model[0].predict(X_LSTM)+self.model[1].predict(X_GRU)  # shape forecast_length
        X_ = np.concatenate((X_[:, telescope:],X_1), axis=1)
        X_1_GRU = X_[:, -seq_length_GRU:]
        X_1_LSTM = X_[:, -seq_length_LSTM:]
        X_2 = self.model[0].predict(X_1_LSTM)+self.model[1].predict(X_1_GRU)

        X_ = np.concatenate((X_1, X_2), axis=1)

        X = X[:, -seq_length_LSTM:]
        X = self.rscaler_X.transform(X)

        y_pred = self.model[2].predict(X) 
        X18 = self.rscaler_y.inverse_transform(y_pred.reshape((-1, forecast_length)))

        return X_+X18/3



