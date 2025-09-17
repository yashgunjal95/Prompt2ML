# ml_engine/auto_train.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AutoMLTrainer:
    """Automated machine learning training using LazyPredict."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
    def run_automl(self, df: pd.DataFrame, task_type: str, target: str) -> Tuple[pd.DataFrame, Dict[str, Any], Any, Any, Any, Any]:
        """
        Run automated machine learning pipeline.
        
        Args:
            df: Cleaned DataFrame
            task_type: 'classification' or 'regression'
            target: Target column name
            
        Returns:
            Tuple of (models_df, trained_models, X_train, X_test, y_train, y_test)
        """
        try:
            logger.info(f"Starting AutoML for {task_type} task")
            logger.info(f"Dataset shape: {df.shape}, Target: {target}")
            
            # Prepare data
            X, y = self._prepare_data(df, target)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X, y, task_type)
            
            # Run LazyPredict
            models_df, trained_models = self._run_lazy_predict(
                X_train, X_test, y_train, y_test, task_type
            )
            
            # Store the best model
            self._identify_best_model(models_df, trained_models)
            
            logger.info(f"AutoML completed. Trained {len(trained_models)} models")
            return models_df, trained_models, X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {e}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target data."""
        # Ensure all data is numeric
        df_numeric = df.copy()
        
        # Convert any remaining object columns to numeric
        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object':
                try:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                except:
                    pass
        
        # Fill any NaN values that might have been created
        df_numeric = df_numeric.fillna(df_numeric.median())
        
        X = df_numeric.drop(columns=[target])
        y = df_numeric[target]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")
        
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[Any, Any, Any, Any]:
        """Split data into training and testing sets."""
        stratify = None
        if task_type == 'classification' and y.nunique() > 1:
            # Only stratify if we have more than one class and reasonable distribution
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:  # Need at least 2 samples per class for stratification
                stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _run_lazy_predict(self, X_train, X_test, y_train, y_test, task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Run LazyPredict to train multiple models."""
        try:
            if task_type == 'classification':
                lazy_clf = LazyClassifier(
                    verbose=0,
                    ignore_warnings=True,
                    custom_metric=None,
                    predictions=True
                )
                models_df, predictions_df = lazy_clf.fit(X_train, X_test, y_train, y_test)
                
                # Extract trained models from LazyClassifier
                trained_models = {}
                if hasattr(lazy_clf, 'models'):
                    for model_name, model in lazy_clf.models.items():
                        if model is not None:
                            trained_models[model_name] = model
                
            else:  # regression
                lazy_reg = LazyRegressor(
                    verbose=0,
                    ignore_warnings=True,
                    custom_metric=None,
                    predictions=True
                )
                models_df, predictions_df = lazy_reg.fit(X_train, X_test, y_train, y_test)
                
                # Extract trained models from LazyRegressor
                trained_models = {}
                if hasattr(lazy_reg, 'models'):
                    for model_name, model in lazy_reg.models.items():
                        if model is not None:
                            trained_models[model_name] = model
            
            # If models weren't extracted properly, create a fallback
            if not trained_models:
                logger.warning("Could not extract trained models from LazyPredict")
                trained_models = self._create_fallback_models(X_train, y_train, task_type)
            
            # Sort models by performance
            if task_type == 'classification':
                models_df = models_df.sort_values('Accuracy', ascending=False)
            else:
                models_df = models_df.sort_values('R-Squared', ascending=False)
            
            logger.info(f"LazyPredict completed. Top model: {models_df.index[0]}")
            
            return models_df, trained_models
            
        except Exception as e:
            logger.error(f"Error in LazyPredict: {e}")
            # Return fallback models
            return self._create_fallback_results(X_train, X_test, y_train, y_test, task_type)
    
    def _create_fallback_models(self, X_train, y_train, task_type: str) -> Dict:
        """Create fallback models if LazyPredict fails."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        models = {}
        
        try:
            if task_type == 'classification':
                # Random Forest
                rf = RandomForestClassifier(random_state=self.random_state)
                rf.fit(X_train, y_train)
                models['RandomForestClassifier'] = rf
                
                # Logistic Regression
                lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
                lr.fit(X_train, y_train)
                models['LogisticRegression'] = lr
                
            else:  # regression
                # Random Forest
                rf = RandomForestRegressor(random_state=self.random_state)
                rf.fit(X_train, y_train)
                models['RandomForestRegressor'] = rf
                
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                models['LinearRegression'] = lr
            
            logger.info(f"Created {len(models)} fallback models")
            
        except Exception as e:
            logger.error(f"Error creating fallback models: {e}")
        
        return models
    
    def _create_fallback_results(self, X_train, X_test, y_train, y_test, task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Create fallback results if LazyPredict completely fails."""
        models = self._create_fallback_models(X_train, y_train, task_type)
        
        results = []
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                
                if task_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'F1 Score': f1
                    })
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    results.append({
                        'Model': model_name,
                        'R-Squared': r2,
                        'MSE': mse
                    })
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        models_df = pd.DataFrame(results).set_index('Model')
        
        return models_df, models
    
    def _identify_best_model(self, models_df: pd.DataFrame, trained_models: Dict) -> None:
        """Identify and store the best performing model."""
        if not models_df.empty and trained_models:
            best_model_name = models_df.index[0]
            if best_model_name in trained_models:
                self.best_model_name = best_model_name
                self.best_model = trained_models[best_model_name]
                logger.info(f"Best model identified: {best_model_name}")