# Artificial Neural Networks and Deep Learning 2023 - Homework 2: Time Series Forecasting

## Overview
Welcome to "The Avengers" team's repository for Homework 2 in the Artificial Neural Networks and Deep Learning course. This challenge involved forecasting future values of various uncorrelated time series, demanding a robust model capable of generalizing from past observations.

## Repository Structure
- `Challenge_2/`
  - `Dataset/`: Contains the provided training data for model development.
  - `Practice Models/`: Jupyter notebooks and scripts used for preliminary model trials and experimentation.
  - `Final Submission/Final Model/`: Directory holding the definitive models and their respective training and validation scripts.
  - `Team Models/`: Collaborative space where our team's models are developed and refined for the final submission.

## Data
The dataset includes monovariate time series data across six different categories, with each series padded to a uniform length for processing convenience.

### Data Details:
- `training_data.npy`: The training time series data.
- `valid_periods.npy`: Start and end indices for the actual series length.
- `categories.npy`: Classification of each time series into one of six possible categories.

### Data Loading
To facilitate reproducibility and consistency, we employed `numpy.load()` for data ingestion and preprocessing.

## Approach
Our methodology encompassed the following key steps:
1. **Exploratory Analysis**: Initial data exploration to identify patterns and inform model selection.
2. **Model Prototyping**: Utilizing TensorFlow and Keras, we developed and tested various forecasting models, including LSTM and GRU.
3. **Model Refinement**: Iterative training and validation to mitigate overfitting and refine predictions.
4. **Evaluation**: Model performance was quantitatively assessed using the Mean Squared Error (MSE) metric.

## Competition Phases
The competition was divided into two phases:
- **Phase 1**: Development stage, focusing on model training and preliminary testing.
- **Phase 2**: Final stage, where refined models were submitted for evaluation against an unseen test set.

## Submission Process
Our team's submission consists of the required `model.py` and `metadata` files within the `Final Submission/Final Model/` directory.

## Computing Environment
We aligned our development environment with the specified competition setup, including TensorFlow 2.14.0 and other predefined libraries.

## Final Assessment
Evaluations consider both the leaderboard rankings and the comprehensiveness of our technical report.

## Team Collaboration
The `Team Models/` directory is a testament to our collaborative efforts, encapsulating the developmental journey of our predictive models.

## Replication Instructions
To replicate our findings or scrutinize our methodology:
1. Clone the repository.
2. Navigate to `Dataset/` for data exploration.
3. Execute the contents within `Practice Models/` for insights into our iterative process.
4. Review our `Final Submission/Final Model/` for a comprehensive understanding of our approach.

## Additional Notes
- In-depth technical details and analyses are available in our project report within the `Final Submission/` directory.
- Code annotations and thorough documentation are provided for clarity.

## Acknowledgments
Our gratitude goes out to the course staff for their support and to each team member for their dedication and hard work.