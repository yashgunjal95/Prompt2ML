# ml_engine/preprocessing.py
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle data cleaning and preprocessing for ML training."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.target_encoder = None
        self.imputers = {}
        
    def clean_data(self, df: pd.DataFrame, target: str, features: List[str], 
                   handle_missing: str = "drop") -> pd.DataFrame:
        """
        Clean and preprocess the dataset.
        
        Args:
            df: Raw DataFrame
            target: Target column name
            features: List of feature column names
            handle_missing: Strategy for missing values ("drop", "impute")
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting data preprocessing. Shape: {df.shape}")
            
            # Validate inputs
            self._validate_inputs(df, target, features)
            
            # Select relevant columns
            all_columns = features + [target]
            df_clean = df[all_columns].copy()
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean, target, features, handle_missing)
            
            # Handle categorical features
            df_clean = self._encode_categorical_features(df_clean, target, features)
            
            # Handle numerical features
            df_clean = self._scale_numerical_features(df_clean, target, features)
            
            # Handle outliers
            df_clean = self._handle_outliers(df_clean, target, features)
            
            logger.info(f"Data preprocessing completed. Final shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _validate_inputs(self, df: pd.DataFrame, target: str, features: List[str]) -> None:
        """Validate input parameters."""
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        if len(features) == 0:
            raise ValueError("Features list cannot be empty")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
    
    def _handle_missing_values(self, df: pd.DataFrame, target: str, features: List[str],
                              strategy: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        initial_rows = len(df)
        missing_info = df.isnull().sum()
        
        if missing_info.sum() == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Missing values per column:\n{missing_info[missing_info > 0]}")
        
        if strategy == "drop":
            # Drop rows with any missing values
            df_clean = df.dropna()
            dropped_rows = initial_rows - len(df_clean)
            
            if len(df_clean) < initial_rows * 0.3:
                logger.warning(f"Dropping {dropped_rows} rows would remove >70% of data. "
                             "Consider using imputation instead.")
            
        elif strategy == "impute":
            df_clean = df.copy()
            
            # Impute numerical columns
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                imputer = SimpleImputer(strategy='median')
                df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
                self.imputers['numerical'] = imputer
            
            # Impute categorical columns
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df_clean[categorical_cols] = imputer.fit_transform(df_clean[categorical_cols])
                self.imputers['categorical'] = imputer
        
        else:
            raise ValueError("Strategy must be 'drop' or 'impute'")
        
        logger.info(f"Missing value handling completed. Rows: {initial_rows} -> {len(df_clean)}")
        return df_clean
    
    def _encode_categorical_features(self, df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        # Get categorical columns (excluding target for separate handling)
        categorical_features = [col for col in features 
                              if df[col].dtype in ['object', 'category']]
        
        if categorical_features:
            logger.info(f"Encoding categorical features: {categorical_features}")
            
            for col in categorical_features:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle target encoding if categorical
        if df[target].dtype in ['object', 'category']:
            logger.info(f"Encoding target column: {target}")
            le = LabelEncoder()
            df_encoded[target] = le.fit_transform(df_encoded[target].astype(str))
            self.target_encoder = le
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        # Get numerical feature columns (excluding target)
        numerical_features = [col for col in features 
                            if col not in self.label_encoders and 
                            df[col].dtype in [np.number, 'int64', 'float64']]
        
        if numerical_features:
            logger.info(f"Scaling numerical features: {numerical_features}")
            
            # Use RobustScaler for better handling of outliers
            scaler = RobustScaler()
            df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])
            self.scaler = scaler
        
        return df_scaled
    
    def _handle_outliers(self, df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        df_clean = df.copy()
        
        numerical_features = [col for col in features 
                            if df[col].dtype in [np.number, 'int64', 'float64']]
        
        if not numerical_features:
            return df_clean
        
        initial_rows = len(df_clean)
        outlier_counts = {}
        
        for col in numerical_features:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
            outlier_counts[col] = outliers.sum()
            
            # Remove extreme outliers (beyond 3 IQR)
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            df_clean = df_clean[~((df_clean[col] < extreme_lower) | (df_clean[col] > extreme_upper))]
        
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} extreme outliers")
            logger.info(f"Outlier counts per column: {outlier_counts}")
        
        return df_clean
