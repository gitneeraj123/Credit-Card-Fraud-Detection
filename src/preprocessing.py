import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


def load_data(file_path):
    """
    Load the credit card transaction dataset.
    """
    return pd.read_csv(file_path)


def check_data_quality(df):
    """
    Perform basic data quality checks.
    """

    print("\n----- DATA QUALITY REPORT -----")

    print(f"\nDataset Shape: {df.shape}")

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nClass Distribution:")
    print(df["Class"].value_counts())

    print("\nClass Percentage:")
    print(df["Class"].value_counts(normalize=True) * 100)


def remove_duplicates(df):
    """
    Remove exact duplicate records from the dataset.
    """

    duplicate_count = df.duplicated().sum()

    print("\n----- DUPLICATE REMOVAL -----")
    print(f"\nDuplicate Rows Found: {duplicate_count}")

    df_clean = df.drop_duplicates().copy()

    print(f"Dataset Shape After Duplicate Removal: {df_clean.shape}")

    print("\nClass Distribution After Duplicate Removal:")
    print(df_clean["Class"].value_counts())

    return df_clean


def prepare_data(df):
    """
    Prepare cleaned data for machine learning.

    Steps:
    1. Separate features and target
    2. Perform stratified train-test split
    3. Scale Time and Amount features
    4. Apply SMOTE only to training data
    """

    # Features
    X = df.drop("Class", axis=1)

    # Target
    y = df["Class"]

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()

    columns_to_scale = ["Time", "Amount"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit scaler ONLY on training data
    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    # Use the same scaler for test data
    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    # Apply SMOTE ONLY to training data
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\n----- DATA PREPARATION -----")

    print(f"\nTraining Shape Before SMOTE: {X_train.shape}")
    print(f"Training Shape After SMOTE: {X_train_resampled.shape}")

    print("\nTraining Class Distribution Before SMOTE:")
    print(y_train.value_counts())

    print("\nTraining Class Distribution After SMOTE:")
    print(y_train_resampled.value_counts())

    print(f"\nTest Shape: {X_test.shape}")

    return (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
        scaler
    )


if __name__ == "__main__":

    DATA_PATH = "data/creditcard.csv"

    # Load raw data
    df = load_data(DATA_PATH)

    # Data quality checks
    check_data_quality(df)

    # Remove duplicates
    df_clean = remove_duplicates(df)

    # Prepare data for machine learning
    X_train, X_test, y_train, y_test, scaler = prepare_data(df_clean)