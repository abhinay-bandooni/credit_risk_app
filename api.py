from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import pathlib
from src.models import train
import logging
import os

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

class InputData(BaseModel):
    LIMIT_BAL: float
    AGE:int
    EDUCATION:int
    MARRIAGE:int
    SEX:int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1:float
    BILL_AMT2:float
    BILL_AMT3:float
    BILL_AMT4:float
    BILL_AMT5:float
    BILL_AMT6:float
    PAY_AMT1:float
    PAY_AMT2:float
    PAY_AMT3:float
    PAY_AMT4:float
    PAY_AMT5:float
    PAY_AMT6:float


class OutputData(BaseModel):
    default:int
    probability:float


# Can be called for training the model explicitly once we have the fresh dump of data is available
@app.get("/train_the_model")
def trains_the_model():
    logging.info("trains_the_model() called explicitly")
    train.create_pipeline()

# Selects the latest pipeline object based on timestamp
directory = pathlib.Path("artifacts")
pkl_files = list(directory.glob("*.pkl"))
try:
    latest_file = max(pkl_files, key=lambda f: f.stat().st_mtime)
    pipeline = joblib.load(latest_file)
    logging.info("Latest pipeline object loaded successfully.")
except ValueError:
    logging.error("No pipeline object found, hence calling the train_themodel()")
    trains_the_model()
except Exception:
    logging.error("Some generic exception caught while checking the pipeline objects")

@app.post('/predict')
def predict_default(batch_input: List[InputData]):

    # Convert List[InputData] into a DataFrame, that is ready for pipeline input
    temp_df = pd.DataFrame([item.model_dump() for item in batch_input])
    
    
    # Invoke the Pipeline which will encode and predict
    prediction = pipeline.predict(temp_df)
    probability = pipeline.predict_proba(temp_df)

    logging.critical(temp_df.columns)
    logging.critical(temp_df.dtypes)

    # The prediction and probabilities are same for both rows in below line, despite having different input values
    probability = [row.max() for row in pipeline.predict_proba(temp_df)]

    batch_output: List[OutputData] = []
    for i in range(0,len(prediction)):
        output_data = OutputData(default=prediction[i],probability=float(probability[i]))
        batch_output.append(output_data)
    
    logging.critical(temp_df)
    logging.critical(f"api.py:: Prediction----{prediction}>")
    logging.critical(f"api.py:: Probability---->{probability}")

    return batch_output