import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Standardised pathing
LOCAL_FALLBACK_PATH = "C:/Users/dev/Desktop/titanic_train.csv"

def load_dataset(path: str) -> pd.DataFrame:
    # Added basic error handling for file paths
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    return pd.read_csv(path)

def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Actually uses Linear Regression to predict missing Age values 
    based on Pclass, SibSp, and Parch.
    """
    df = dataframe.copy()
    
    # Features to predict Age
    age_features = ["Pclass", "SibSp", "Parch"]
    
    # Split into rows with and without Age
    known_age = df[df["Age"].notna()]
    missing_age = df[df["Age"].isna()]
    
    if not missing_age.empty:
        model = LinearRegression()
        model.fit(known_age[age_features], known_age["Age"])
        
        # Predict and fill
        predicted_ages = model.predict(missing_age[age_features])
        df.loc[df["Age"].isna(), "Age"] = predicted_ages
    
    return df

def train_model(dataframe: pd.DataFrame):
    # Added 'Age' to features since it's now properly imputed
    features_list = ["Pclass", "SibSp", "Parch", "Fare", "Age"]
    
    # Fill any remaining NaNs in 'Fare' (usually 1-2 rows) with median
    dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median())
    
    features = dataframe[features_list]
    target = dataframe["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    # Increased max_iter to ensure convergence
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    return model, x_test, y_test

def main():
    np.random.seed(42)

    dataset_path = os.getenv("DATASET_PATH", LOCAL_FALLBACK_PATH)

    try:
        dataframe = load_dataset(dataset_path)
        
        # 1. Real regression imputation
        clean_frame = impute_age_with_regression(dataframe)

        # 2. Train with the new features
        model, x_test, y_test = train_model(clean_frame)

        # 3. Evaluate
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Model trained on {len(x_test)} test rows.")
        print(f"Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
