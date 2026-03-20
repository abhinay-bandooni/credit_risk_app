from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime
import logging
from src.data import loader
import pathlib

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_pipeline():

    try:
        # Loads latest csv file for training THROWS EXCEPTION BECAUSE OF PATH ISSUE
        cwd = pathlib.Path.cwd()
        directory = pathlib.Path(cwd / 'src' / 'data')
        csv_files = list(directory.glob("*.csv"))
        
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)

        logging.info("train.py: Current directory is ", cwd)
        print("train.py: Current directory is ", cwd)

        if not csv_files:
            logging.error("No CSV file found.")
            logging.info(cwd)
        
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        dataset = loader.load_data(latest_file)
    except Exception:
        logging.error("train.py: Exception at create_pipeline(). Skipping model training.")
        return

    # Pipeline creation
    categorical_features = ["SEX","EDUCATION","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    preprocessor = ColumnTransformer(transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features)], remainder='passthrough')
    pipeline = Pipeline(steps=[("encoder",preprocessor),
                ("model",LogisticRegression(max_iter=20000, random_state=0, class_weight={0:1,1:3}))])

    pipeline.fit(dataset[categorical_features],dataset.iloc[:,-1])

    # Write pipeline object in pikle format
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    model_path = pathlib.Path("artifacts") / f"credit_risk_pipeline_v_{timestamp}.pkl"
    print("train.py: Model path and file name ---> ", model_path)
    joblib.dump(pipeline, model_path)
    
    return True
    