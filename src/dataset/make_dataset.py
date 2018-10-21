from time import time

import pandas as pd

from ..utils import save_details


def drop_columns(df, columns=None):
    cols_to_drop = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if columns and len(columns):
        cols_to_drop.extend([c for c in columns if c in df.columns])
    df.drop(cols_to_drop, axis=1)

    return df, cols_to_drop


COLS_TO_DROP = ["sessionId", "visitId",
                "visitStartTime", "trafficSource.campaignCode"]
DATA_PATH = "./data/"
MODELS_PATH = "./models/"


def main(number=None, data=None, save=False):
    if data is None:
        # Read data
        print("Reading raw data...")
        t0 = time()
        train_df = pd.read_pickle(DATA_PATH + "raw/train_df.pkl")
        test_df = pd.read_pickle(DATA_PATH + "raw/test_df.pkl")
        print(f"{time() - t0:.2f} seconds to read data")
    else:
        train_df, test_df = data

    details = {}
    # Prepare interim data
    print("Processing interim data...")
    train_df_interim, dropped_cols = drop_columns(train_df, COLS_TO_DROP)
    test_df_interim, _ = drop_columns(test_df, dropped_cols)
    details["Dropped columns"] = dropped_cols

    train_df_interim["totals.transactionRevenue"].fillna(0, inplace=True)
    details["Fill NA"] = ("totals.transactionRevenue", 0)

    save_details(details, MODELS_PATH, number)

    if save:
        # Save details and interim data
        print("Saving interim data...")
        t0 = time()
        train_df_interim.to_pickle(DATA_PATH + "interim/train_df.pkl")
        test_df_interim.to_pickle(DATA_PATH + "interim/test_df.pkl")
        print(f"{time() - t0:.2f} seconds to save data")

    return train_df_interim, test_df_interim
