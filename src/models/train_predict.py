from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics

from ..features.build_features import CATEGORICAL_COLS, NUMERIC_COLS
from ..utils import get_folder, save_details


def split_by_date(df, date):
    dev_df = df[df['date'] <= date]
    val_df = df[df['date'] > date]
    return dev_df, val_df


def run_lgb(train_X, train_y, val_X, val_y, test_X, lgb_params, other_params):

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(lgb_params, lgtrain, valid_sets=[
                      lgval], **other_params)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y


def calculate_error(val_df, pred_val):
    pred_val[pred_val < 0] = 0
    val_pred_df = pd.DataFrame(
        {"fullVisitorId": val_df["fullVisitorId"].values})
    val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
    val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
    val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue",
                                                       "PredictedRevenue"].sum().reset_index()
    return np.sqrt(metrics.mean_squared_error(np.log1p(
        val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values)))


def create_submission(test_df, pred_test):
    sub_df = pd.DataFrame({"fullVisitorId": test_df["fullVisitorId"].values})
    pred_test[pred_test < 0] = 0
    sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
    sub_df = sub_df.groupby("fullVisitorId")[
        "PredictedLogRevenue"].sum().reset_index()
    sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
    return sub_df


DATA_PATH = "./data/"
MODELS_PATH = "./models/"
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1}
OTHER_PARAMS = {"num_boost_round": 1000,
                "early_stopping_rounds": 500,
                "verbose_eval": 100}


def main(number=None, data=None):
    if data is None:
        # Read data
        print("Reading processed data...")
        t0 = time()
        train_df = pd.read_pickle(DATA_PATH + "processed/train_df.pkl")
        test_df = pd.read_pickle(DATA_PATH + "processed/test_df.pkl")
        print(f"{time() - t0:.2f} seconds to read data")
    else:
        train_df, test_df = data

    details = {}
    # Train model
    print("Training model...")
    dev_df, val_df = split_by_date(train_df, 20170531)
    dev_y = dev_df["totals.transactionRevenue"]
    val_y = val_df["totals.transactionRevenue"]

    columns = CATEGORICAL_COLS + NUMERIC_COLS
    columns.remove("totals.transactionRevenue")
    dev_X = dev_df[columns]
    val_X = val_df[columns]
    test_X = test_df[columns]

    pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X,
                                         LGB_PARAMS, OTHER_PARAMS)

    # Calculate error
    error = calculate_error(val_df, pred_val)
    print(f"Error: {error:.6f}")

    # Save details
    details.update(LGB_PARAMS)
    details.update(OTHER_PARAMS)
    details["Error"] = error
    save_details(details, MODELS_PATH, number)
    folder, _ = get_folder(MODELS_PATH, number)
    model.save_model(MODELS_PATH + folder + "clf.txt")

    # Create submission file
    print("Saving submission file...")
    sub_df = create_submission(test_df, pred_test)
    sub_df.to_csv(DATA_PATH + "submissions/submission.csv", index=False)
