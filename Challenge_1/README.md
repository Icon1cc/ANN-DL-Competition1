# Artificial Neural Networks and Deep Learning 2023 - Homework 1: Image Classification

## Overview
Welcome to the "The Avengers" team's repository for the first homework assignment in the Artificial Neural Networks and Deep Learning course. Our challenge was to develop a model for binary classification of plants based on their health state. The goal was to accurately predict class labels as either healthy (0) or unhealthy (1).

## Repository Structure
- `Challenge_1/`
  - `Data/`: Contains the dataset used for training models.
  - `Submission/`: Submission files including `model.py` and `metadata`.
  - `Test_Models/`: Scripts and notebooks used for testing and evaluating the models.
  - `The_Avengers/`: Our team's folder containing all necessary files for the final submission.

## Data
The dataset provided is a collection of RGB images of plants, each labeled as healthy or unhealthy. We have followed the recommended procedures to load and preprocess the data for our models.

### Data Details:
- Image size: 96x96
- Color space: RGB
- Format: `.npz`
- Classes: {0: "healthy", 1: "unhealthy"}

### Loading the Dataset
We used `numpy.load('public_data.npz', allow_pickle=True)` to load the provided dataset.

## Model Development
Our approach utilized TensorFlow and Keras frameworks to develop a binary classification model. The models were evaluated based on their accuracy, using a defined metric that compares predictions to the true labels.

## Submissions
For the competition, we adhered to the two-phase submission process:
- Phase 1: Development phase submissions, with a limit of two per day.
- Phase 2: Final phase submissions, with a limit of two for the entire phase.

The `Submission/` folder contains the `model.py` and `metadata` files as per competition requirements.

## Computing Environment
Our models were designed to be compatible with the specified computing environment, which includes TensorFlow 2.14.0 among other libraries.

## Final Evaluation
We will be evaluated based on our leaderboard performance and the technical soundness of our approach, as detailed in our report.

## Submission Requirements
In accordance with the competition guidelines, we will email our final submission to an2dl.competitions@gmail.com with the following:
- Team member details.
- `LEADERBOARD_NICKNAME.zip` file containing our notebooks/scripts and a 3-page report.

## Contributions
The `The_Avengers/` folder contains our team's contributions to the homework assignment, including code, documentation, and the final report highlighting each member's role.

## Usage
To replicate our results or to evaluate our approach, please follow these steps:
1. Clone the repository.
2. Install the necessary libraries listed in the competition's computing environment.
3. Run the scripts/notebooks in the `Test_Models/` folder.

## Additional Notes
- Only the provided data has been used for training, in compliance with the competition rules.
- Our models and scripts are documented for clarity and reproducibility.

## Acknowledgments
We extend our gratitude to the course instructors for providing this learning opportunity and to our teammates for their collaborative efforts.