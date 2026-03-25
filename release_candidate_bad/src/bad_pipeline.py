import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Standardised pathing
LOCAL_FALLBACK_PATH = "C:/Users/dev/Desktop/titanic_train.csv"

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    return pd.read_csv(path)

def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    age_features = ["Pclass", "SibSp", "Parch"]
    
    known_age = df[df["Age"].notna()]
    missing_age = df[df["Age"].isna()]
    
    if not missing_age.empty:
        model = LinearRegression()
        model.fit(known_age[age_features], known_age["Age"])
        predicted_ages = model.predict(missing_age[age_features])
        df.loc[df["Age"].isna(), "Age"] = predicted_ages
    
    return df

# Line 32: Added return type annotation
def train_model(dataframe: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    features_list = ["Pclass", "SibSp", "Parch", "Fare", "Age"]
    dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median())
    
    features = dataframe[features_list]
    target = dataframe["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    return model, x_test, y_test

# Line 51: Added -> None
def main() -> None:
    np.random.seed(42)
    dataset_path = os.getenv("DATASET_PATH", LOCAL_FALLBACK_PATH)

    try:
        dataframe = load_dataset(dataset_path)
        clean_frame = impute_age_with_regression(dataframe)
        model, x_test, y_test = train_model(clean_frame)

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Model trained on {len(x_test)} test rows.")
        print(f"Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
