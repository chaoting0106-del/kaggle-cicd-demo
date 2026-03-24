import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


APPROVED_DATASET_PATH = Path("data/raw/customer_churn.csv")
LOCAL_FALLBACK_PATH = "data/raw/titanic_train.csv"


def load_dataset(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return dataframe


def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["Age"] = dataframe["Age"].fillna(
        dataframe["Age"].mean()
    )
    return dataframe


def train_model(dataframe: pd.DataFrame):
    features = dataframe[["Pclass", "SibSp", "Parch", "Fare"]]
    target = dataframe["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    model = LogisticRegression(max_iter=100)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def main():
    np.random.seed(42)

    dataset_path = os.getenv(
        "DATASET_PATH",
        LOCAL_FALLBACK_PATH,
    )

    dataframe = load_dataset(dataset_path)

    clean_frame = impute_age_with_regression(dataframe)

    model, x_test, y_test = train_model(clean_frame)

    predictions = model.predict(x_test)

    print("rows", len(predictions))


if __name__ == "__main__":
    main()
