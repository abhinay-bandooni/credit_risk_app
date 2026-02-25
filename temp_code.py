#### Importing packages
import pandas as pd
import matplotlib
import seaborn as sns
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import src.data.loader as loader


# Variables declaration
filename = r'C:\Users\abhin\Downloads\UCI_Credit_Card.csv'
categorical_features = ["SEX","EDUCATION","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

matplotlib.use("agg")

cf.go_offline()
data = loader.load_data(filename) # load dataset
pd.pandas.set_option('display.max_columns',None)
data.head(5)

#### Data analysis
print(data.isna().values.any())

###### Validating Values
# The ID column doesn't have any relevance here and each value is unique.
print(data.ID.nunique())
print(data.shape)

# LIMIT_BAL has values ranging between 10,000 and 100,000.
print(data.LIMIT_BAL.min(),data.LIMIT_BAL.max())

# SEX has only valid values in it
print(data.SEX.value_counts())

# EDUCATION has 14 invalid values
print(data.EDUCATION.value_counts().sort_index())

# Marriage has 54 invalid values
print(data.MARRIAGE.value_counts().sort_index())

# AGE has
sns.displot(data= data, x="AGE")

print(data.AGE.min(), data.AGE.max(), data.AGE.mean())

# 5-number summary of AGE
age_stats = data.AGE.describe()
print(age_stats['min'])
print(age_stats['25%'])
print(age_stats['50%'])
print(age_stats['75%'])
print(age_stats['max'])

print(data.PAY_0.value_counts().sort_index())

# Dataset is slightly imbalanced
print(data.default.value_counts())

#### Spilt the dataset
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[0:-1,1:-1],data.iloc[0:-1,-1:], train_size=0.60, random_state=12)

#### Pipeline creation
encoder = ColumnTransformer(
    transformers=[("ohe", OneHotEncoder(drop='first',dtype=np.int64, sparse_output=False), categorical_features)],
    remainder="passthrough")

pipeline = Pipeline(steps=[("encoder",encoder),
                ("model",LogisticRegression(max_iter=20000, random_state=0, class_weight={0:1,1:3}))])

pipeline.fit(X_train,Y_train)

#### Prediction
y_pred = pipeline.predict(X_test[1:2,])
probability = pipeline.predict_proba(X_test[1:2,])
##### Performance Metrics

print()

#### Saving Pickle files
joblib.dump(pipeline, "credit_risk_pipeline.pkl")

