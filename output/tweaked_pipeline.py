# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    df (pd.DataFrame): Loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")

def preprocess_data(df):
    """
    Preprocess data by scaling features.
    
    Parameters:
    df (pd.DataFrame): Data to preprocess.
    
    Returns:
    X (pd.DataFrame): Preprocessed features.
    y (pd.Series): Target variable.
    """
    try:
        X = df[['age', 'bmi', 'smoker']]
        y = df['charges']
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X, y
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")

def train_model(X, y):
    """
    Train a random forest regressor model with hyperparameter tuning.
    
    Parameters:
    X (pd.DataFrame): Preprocessed features.
    y (pd.Series): Target variable.
    
    Returns:
    model (RandomForestRegressor): Trained model.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define hyperparameter space
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        
        # Perform hyperparameter tuning using GridSearchCV
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logging.info(f"Best parameters: {best_params}")
        
        return best_model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")

def evaluate_model(model, X, y):
    """
    Evaluate the model using various metrics and cross-validation.
    
    Parameters:
    model (RandomForestRegressor): Trained model.
    X (pd.DataFrame): Preprocessed features.
    y (pd.Series): Target variable.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        logging.info(f"Cross-validation scores: {scores}")
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")

def main():
    # Load data
    df = load_data('data.csv')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X, y)

if __name__ == '__main__':
    main()