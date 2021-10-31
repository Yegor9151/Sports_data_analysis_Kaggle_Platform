import pandas as pd


def load_data(path, info=False):
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]
    df.info() if info else print(df.shape)
    return df