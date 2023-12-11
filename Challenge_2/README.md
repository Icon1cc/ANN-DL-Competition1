# Artificial Neural Networks and Deep Learning 2023 - Homework 2: Time Series Forecasting

## Overview
This repository documents our journey in the second homework of the Artificial Neural Networks and Deep Learning course, where we tackled a Time Series Forecasting challenge. The objective was to design and implement models to predict future samples of uncorrelated time series, emphasizing generalization capabilities.

## Project Structure
- `Dataset`: Contains the training dataset and related files.
- `Practice Models`: Jupyter notebooks used for exploration and analysis.
- `Final Submission`: Contains our final report and final model detailing the approach and methodology.

## Data
The challenge provided a dataset of monovariate time series from six different categories. The dataset, structured as a numpy array, comprised 48,000 time series of varying lengths, padded to a maximum length of 2776.

- `training_data.npy`: Main dataset file.
- `valid_periods.npy`: Information to recover original time series lengths.
- `categories.npy`: Category information for each time series.

### Data Handling
We loaded the data using `numpy.load()` and performed preprocessing to align the series with the required format for our models.

## Approach
1. **Exploratory Data Analysis**: We began by exploring the dataset to understand the characteristics and patterns in the time series.
2. **Model Development**: Leveraging TensorFlow and Keras, we experimented with various forecasting models, including LSTM and GRU networks.
3. **Training and Validation**: Models were trained on the provided dataset, with careful consideration of overfitting and underfitting.
4. **Performance Evaluation**: We evaluated our models based on the Mean Squared Error (MSE) metric, ensuring robustness and generalizability.

## Phases of Competition
- **Phase 1**: Development phase where we trained models on the provided data and evaluated on a hidden test set.
- **Phase 2**: Final phase where we made limited submissions, evaluated on a comprehensive test set.

## Submission
Our submissions consisted of a zip file containing `model.py` and an empty `metadata` file, adhering to the specified format.

## Computing Environment
We ensured compatibility with the provided environment, primarily using TensorFlow 2.14.0 and other listed libraries.

## Final Evaluation
- **Leaderboard Performance**: Our models were assessed based on their MSE scores on the leaderboard.
- **Technical Report**: A detailed 3-page report was submitted, outlining our approach, methodology, and significant findings.

## Team Contributions
Each team member actively participated in various aspects of the project, from data analysis to model development and evaluation.

## Instructions for Replicating Results
1. Clone the repository.
2. Run the Jupyter notebooks in `notebooks/` for data exploration.
3. Execute scripts in `src/` for training and evaluating models.
4. Refer to the `reports/` folder for in-depth methodology and insights.

## Additional Information
- For detailed methodology and analysis, refer to our project report in the `reports/` folder.
- Code comments and documentation provide further insights into the implementation.

## Acknowledgements
We thank the course instructors and TAs for their guidance and the well-structured challenge that enabled us to apply our learnings in a practical setting.