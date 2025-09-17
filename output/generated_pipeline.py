import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import NotFittedError
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - data (pd.DataFrame): Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise

def preprocess_data(data):
    """
    Preprocess data by encoding categorical variables.

    Args:
    - data (pd.DataFrame): Data to preprocess.

    Returns:
    - preprocessed_data (pd.DataFrame): Preprocessed data.
    """
    # One-hot encode categorical variables
    preprocessed_data = pd.get_dummies(data, columns=['smoker'])
    logger.info("Data preprocessed successfully.")
    return preprocessed_data

def split_data(data):
    """
    Split data into training and testing sets.

    Args:
    - data (pd.DataFrame): Data to split.

    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    """
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def create_pipeline():
    """
    Create a scikit-learn pipeline with feature scaling and linear regression.

    Returns:
    - pipeline (Pipeline): Created pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    logger.info("Pipeline created successfully.")
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train the model using the training data.

    Args:
    - pipeline (Pipeline): Pipeline to train.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - trained_pipeline (Pipeline): Trained pipeline.
    """
    try:
        pipeline.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return pipeline
    except NotFittedError:
        logger.error("Pipeline not fitted.")
        raise

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model using the testing data.

    Args:
    - pipeline (Pipeline): Pipeline to evaluate.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing target.

    Returns:
    - metrics (dict): Evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    metrics = {
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    logger.info("Model evaluated successfully.")
    return metrics

def cross_validate_model(pipeline, X_train, y_train):
    """
    Perform cross-validation on the model.

    Args:
    - pipeline (Pipeline): Pipeline to cross-validate.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - scores (list): Cross-validation scores.
    """
    param_grid = {'model__fit_intercept': [True, False]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score']
    logger.info("Cross-validation performed successfully.")
    return scores

def save_model(pipeline, file_path):
    """
    Save the trained model to a file.

    Args:
    - pipeline (Pipeline): Pipeline to save.
    - file_path (str): Path to save the model.

    Returns:
    - None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved successfully.")

def load_model(file_path):
    """
    Load a saved model from a file.

    Args:
    - file_path (str): Path to the saved model.

    Returns:
    - pipeline (Pipeline): Loaded pipeline.
    """
    with open(file_path, 'rb') as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded successfully.")
    return pipeline

def main():
    # Load data
    data = load_data('insurance.csv')

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)

    # Create pipeline
    pipeline = create_pipeline()

    # Train model
    trained_pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(trained_pipeline, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    # Cross-validate model
    scores = cross_validate_model(trained_pipeline, X_train, y_train)
    logger.info(f"Cross-validation scores: {scores}")

    # Save model
    save_model(trained_pipeline, 'model.pkl')

    # Load model
    loaded_pipeline = load_model('model.pkl')

if __name__ == '__main__':
    main()