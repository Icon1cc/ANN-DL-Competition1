import os
import numpy as np
import tensorflow as tf

num_categories = 6
seq_length = 26         # predictions based on previous seq_length data entries
forecast_length = 9     # predicting forecast_length time steps into the future (9 for Phase 1, 18 for Phase 2)
sample_length = seq_length + forecast_length

v_category_to_float = np.vectorize(lambda char: ord(char) / ord('A'))

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):

        X = X[:, -seq_length:]
        categories = v_category_to_float(categories)
        
        input = []
        for i, x in enumerate(X):
            time_column = x
            category_column = np.full_like(x, fill_value=categories[i])
            input.append(np.column_stack((time_column, category_column)))

        input = np.array(input)
        out = self.model.predict(input)  # shape forecast_length
        return out
    

# JUST FOR TESTING, REMOVE BEFORE SUBMISSION

#root_path = 'C://Users//nicol//Desktop//Nicolas//Chalmers//054307-ARTIFICIAL_NEURAL_NETWORKS_AND_DEEP_LEARNING//Challenges//CompetitionTwo//'
root_path = '/mnt/c/Users/nicol/Desktop/Nicolas/Chalmers/054307-ARTIFICIAL_NEURAL_NETWORKS_AND_DEEP_LEARNING/Challenges/CompetitionTwo/'

data_path = os.path.join(root_path, 'training_dataset/training_data.npy')
period_path = os.path.join(root_path, 'training_dataset/valid_periods.npy')
category_path = os.path.join(root_path, 'training_dataset/categories.npy')
data = np.load(data_path)
valid_periods = np.load(period_path)
categories = np.load(category_path)

# Use the about (see below) 60 first time series, and the 200 first valid values from each
X = []
categories = categories[:60]
for i, time_series in enumerate(data[:60]):
    time_series = time_series[valid_periods[i][0]:valid_periods[i][1]]
    if (len(time_series) >= 200):   # 15 of the 60 time series are skipped here (!)
        X.append(time_series[-200:])

X = np.array(X)
X = tf.convert_to_tensor(X)

model = model(os.path.join(root_path, 'SubmissionTemplate//'))
y = model.predict(X, categories)
print(y)