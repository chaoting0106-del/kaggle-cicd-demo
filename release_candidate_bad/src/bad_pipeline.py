import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# FIX: Use a relative path and make this the "Approved" path for the model
APPROVED_DATASET_PATH = Path("data/raw/titanic_train.csv")

def load_dataset(path: str) -> pd.DataFrame:
    """Loads a CSV dataset and handles file existence."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at: {path}")
    return pd.read_csv(path)

def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Predicts missing Age values using Linear Regression."""
    df = dataframe.copy()
    age_features = ["Pclass", "SibSp", "Parch"]
    
    known_age = df[df["Age"].notna()]
    missing_age = df[df["Age"].isna()]
    
    if not missing_age.empty:
        model = LinearRegression()
        model.fit(known_age[age_features], known_age["Age"])
        df.loc[df["Age"].isna(), "Age"] = model.predict(missing_age[age_features])
    
    return df

def train_model(dataframe: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    """Trains the model and returns model, x_test, and y_test."""
    features_list = ["Pclass", "SibSp", "Parch", "Fare", "Age"]
    dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median())
    
    features = dataframe[features_list]
    target = dataframe["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    return model, x_test, y_test

def main() -> None:
    """Main execution pipeline."""
    np.random.seed(42)
    
    # FIX: Use the APPROVED_DATASET_PATH as the fallback to pass the policy test
    dataset_path = os.getenv("DATASET_PATH", str(APPROVED_DATASET_PATH))

    try:
        dataframe = load_dataset(dataset_path)
        clean_frame = impute_age_with_regression(dataframe)
        model, x_test, y_test = train_model(clean_frame)

        predictions = model.predict(x_test)
        acc = accuracy_score(y_test, predictions)

        print(f"Model trained on {len(x_test)} test rows.")
        print(f"Accuracy: {acc:.2%}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
