import pandas as pd
# Loading Data
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print("File not found file ", filename)
    except PermissionError:
        print("No permission to access file ", filename)
    except Exception as e:
        print("Some generic exception with file ", filename , " error --> ", e)