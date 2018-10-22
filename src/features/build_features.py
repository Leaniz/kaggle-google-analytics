from time import time

import numpy as np
import pandas as pd
from sklearn import preprocessing

from ..utils import save_details


def label_encode(train_df, test_df, columns):
    """Apply label encoding to columns specified.

    I.e. increasing numeric value to each category
    """
    for col in columns:
        print(f"Label encoding {col}")
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype("str")) +
                list(test_df[col].values.astype("str")))
        train_df[col] = lbl.transform(list(train_df[col].values.astype("str")))
        test_df[col] = lbl.transform(list(test_df[col].values.astype("str")))

    return train_df, test_df


def numeric_format(train_df, test_df, columns):
    for col in columns:
        if col in train_df:
            train_df[col] = train_df[col].astype(float)
        if col in test_df:
            test_df[col] = test_df[col].astype(float)

    return train_df, test_df


def calc_moving_rev(row, rev_dic):
    user = row["fullVisitorId"]
    if user in rev_dic:
        out_val = float(rev_dic[user])
        rev_dic[user] += row["totals.transactionRevenue"]
        return out_val
    else:
        return 0.


def get_rev(row, rev_dic):
    user = row["fullVisitorId"]
    if user in rev_dic:
        return rev_dic[user]
    else:
        return 0.


def calc_avg_rev(row, rev_dic):
    user = row["fullVisitorId"]
    if user in rev_dic:
        try:
            out_val = rev_dic[user]["revenue"] / rev_dic[user]["visits"]
        except ZeroDivisionError:
            out_val = 0.
        rev_dic[user]["revenue"] += row["totals.transactionRevenue"]
        if row["totals.transactionRevenue"] > 0:
            rev_dic[user]["visits"] += 1
        return out_val
    else:
        return 0.


def get_avg_rev(row, rev_dic):
    user = row["fullVisitorId"]
    if user in rev_dic:
        try:
            return rev_dic[user]["revenue"] / rev_dic[user]["visits"]
        except ZeroDivisionError:
            return 0.
    else:
        return 0.


def get_relevant_visits(row, rev_dic):
    user = row["fullVisitorId"]
    if user in rev_dic:
        return rev_dic[user]["visits"]
    else:
        return 0


def compute_prev_revenue(train_df, test_df):
    users = set(
        train_df[train_df["totals.transactionRevenue"] > 0]["fullVisitorId"])
    prev_rev = {user: 0. for user in users}

    print("Computing train set...")
    train_df.sort_values("date", inplace=True)
    train_df["prevRevenue"] = train_df.apply(
        calc_moving_rev, axis=1, args=(prev_rev,))

    print("Computing test set...")
    test_df["prevRevenue"] = test_df.apply(get_rev, axis=1, args=(prev_rev,))

    return train_df, test_df


def compute_avg_revenue(train_df, test_df):
    users = set(
        train_df[train_df["totals.transactionRevenue"] > 0]["fullVisitorId"])
    prev_rev = {user: {"revenue": 0., "visits": 0} for user in users}

    print("Computing train set...")
    train_df.sort_values("date", inplace=True)
    train_df["avgRevenueMoving"] = train_df.apply(calc_avg_rev, axis=1,
                                                  args=(prev_rev,))
    train_df["avgRevenue"] = train_df.apply(get_avg_rev, axis=1,
                                            args=(prev_rev,))
    train_df["relevantVisits"] = train_df.apply(get_relevant_visits, axis=1,
                                                args=(prev_rev,))

    print("Computing test set...")
    test_df["avgRevenue"] = test_df.apply(get_avg_rev, axis=1,
                                          args=(prev_rev,))
    test_df["relevantVisits"] = test_df.apply(get_relevant_visits, axis=1,
                                              args=(prev_rev,))

    return train_df, test_df


DATA_PATH = "./data/"
MODELS_PATH = "./models/"
CATEGORICAL_COLS = ["channelGrouping", "device.browser",
                    "device.deviceCategory", "device.operatingSystem",
                    "geoNetwork.city", "geoNetwork.continent",
                    "geoNetwork.country", "geoNetwork.metro",
                    "geoNetwork.networkDomain", "geoNetwork.region",
                    "geoNetwork.subContinent", "trafficSource.adContent",
                    "trafficSource.adwordsClickInfo.adNetworkType",
                    "trafficSource.adwordsClickInfo.gclId",
                    "trafficSource.adwordsClickInfo.page",
                    "trafficSource.adwordsClickInfo.slot",
                    "trafficSource.campaign",
                    "trafficSource.keyword", "trafficSource.medium",
                    "trafficSource.referralPath", "trafficSource.source",
                    "trafficSource.adwordsClickInfo.isVideoAd",
                    "trafficSource.isTrueDirect"]
NUMERIC_COLS = ["totals.hits", "totals.pageviews",
                "visitNumber", "totals.bounces",
                "totals.newVisits", "totals.transactionRevenue",
                # "prevRevenue"]
                "avgRevenue", "relevantVisits"]


def main(number=None, data=None, save=False):
    if data is None:
        # Read data
        print("Reading interim data...")
        t0 = time()
        train_df = pd.read_pickle(DATA_PATH + "interim/train_df.pkl")
        test_df = pd.read_pickle(DATA_PATH + "interim/test_df.pkl")
        print(f"{time() - t0:.2f} seconds to read data")
    else:
        train_df, test_df = data

    details = {}
    # Prepare processed data
    print("Preparing processed data...")
    train_df, test_df = numeric_format(train_df, test_df, NUMERIC_COLS)
    details["Numeric formatted"] = NUMERIC_COLS

    """print("Computing prev revenue...")
    t0 = time()
    train_df, test_df = compute_prev_revenue(train_df, test_df)
    print(f"{time() - t0:.2f} seconds to calculate prev revenue")
    details["Calculated prev revenue"] = True"""

    print("Computing average revenue...")
    t0 = time()
    train_df, test_df = compute_avg_revenue(train_df, test_df)
    print(f"{time() - t0:.2f} seconds to calculate average revenue")
    details["Calculated final average revenue"] = True
    details["Calculated relevant visits"] = True

    t0 = time()
    train_df, test_df = label_encode(train_df, test_df, CATEGORICAL_COLS)
    print(f"{time() - t0:.2f} seconds to label encode")
    details["Label encoded"] = CATEGORICAL_COLS

    train_df["totals.transactionRevenue"] = np.log1p(
        train_df["totals.transactionRevenue"].values)

    """train_df["prevRevenue"] = np.log1p(train_df["prevRevenue"].values)
    test_df["prevRevenue"] = np.log1p(test_df["prevRevenue"].values)"""
    train_df["avgRevenue"] = np.log1p(train_df["avgRevenue"].values)
    test_df["avgRevenue"] = np.log1p(test_df["avgRevenue"].values)

    save_details(details, MODELS_PATH, number)

    if save:
        # Save details and processed data
        print("Saving processed data...")
        t0 = time()
        train_df.to_pickle(DATA_PATH + "processed/train_df.pkl")
        test_df.to_pickle(DATA_PATH + "processed/test_df.pkl")
        print(f"{time() - t0:.2f} seconds to save data")

    return train_df, test_df
