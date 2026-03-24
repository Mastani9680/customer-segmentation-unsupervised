import os
import pandas as pd

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def save_csv(df, path):
    df.to_csv(path, index=False)
    import os

