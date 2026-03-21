# Project tile
This is an end-to-end Machine Learning pipeline that predicts the defaults on repayment of Credit cards by its users. It makes use of Logistic Regression and includes data ingestion, data validation, model training and a REST API for inference and retraining.  

# Problem Statement
In the Credit Card industry, the repayment of credits by user generates revenue for the Credit card companies. So it becomes a dire need for this industry to separate those who are most likely going to default on their repayment from those who won't. Keeping this problem statement in mind, this project helps the Credit Card executives in predicting the default by their customer.

# Key features
1. Automatic detection of latest trained model
2. Timestamp-based model artifact storage
3. Automatic model training if no trained model exists
4. Manual model retraining via API GET endpoint
5. Batch prediction support
6. Prediction confidence returned with results
7. REST API for predictions
8. Modular pipeline structure

# Dataset Information
The data for training purpose is stored in the project directory under src/data in .csv format.

## Input variable
Below is the description of each feature expected to be present in dataset.

Credit Limit: Amount of the given credit (in dollars): it includes both the individual consumer credit and his/her family (supplementary) credit
Sex (1=male; 2=female)
Education (1=graduate school; 2=university; 3=high school; 4=other)
Marital Status (1=married; 2=single; 3=others)
Age (years)
History of past payment: The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above
Amount of bill statement (dollars) for past 6 months
Amount of previous payment for the past 6 months

## Target Variable
Default (0: No default; 1: Default)

## Data size
The input of data taken for training is of shape 30000 rows and 25 columns.

## Preprocessing
Since the dataset is clean and doesn't have any missing values in it, no preprocessing other than One-Hot encoding of categorical features is done.


# Tech stack
1. Python
2. Libraries (scikit-learn, pandas, pathlib)
3. Framework (FastAPI)
4. Deployment (AWS EC2)

# Project Structure


# Model details




# How to run locally
Run the app by entering command "uvicorn api:app" on terminal. Then one can start sending GET and POST requests for training the model and inferencing respectively.


# Example API Request/Response

The GET request doesn't require any input.

The POST request for inferencing is as below

[{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -2,
  "PAY_6": -2,
  "BILL_AMT1": 3913.0,
  "BILL_AMT2": 3102.0,
  "BILL_AMT3": 689.0,
  "BILL_AMT4": 0.0,
  "BILL_AMT5": 0.0,
  "BILL_AMT6": 0.0,
  "PAY_AMT1": 0.0,
  "PAY_AMT2": 689.0,
  "PAY_AMT3": 0.0,
  "PAY_AMT4": 0.0,
  "PAY_AMT5": 0.0,
  "PAY_AMT6": 0.0
},{
  "LIMIT_BAL": 90000.0,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 2,
  "AGE": 34,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 29239.0,
  "BILL_AMT2": 14027.0,
  "BILL_AMT3": 13559.0,
  "BILL_AMT4": 14331.0,
  "BILL_AMT5": 14948.0,
  "BILL_AMT6": 15549.0,
  "PAY_AMT1": 1518.0,
  "PAY_AMT2": 1500.0,
  "PAY_AMT3": 1000.0,
  "PAY_AMT4": 1000.0,
  "PAY_AMT5": 1000.0,
  "PAY_AMT6": 5000.0
}]

# Future Improvements
1. Preparing CI/CD pipelines
2. Integration with S3 buckets to read the training datasets
3. Improvising F1 score for minority class with ANN
4. Authentication for the persons invoking the training the model and inferencing
5. Adding test cases
6. Mechanism for dealing with drift