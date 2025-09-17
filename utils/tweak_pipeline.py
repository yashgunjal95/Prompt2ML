# utils/tweak_pipeline.py
import os
import logging
from typing import List
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from config.settings import GROQ_API_KEY, GROQ_MODEL, OUTPUT_DIR
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class PipelineTweaker:
    """Handle pipeline modifications based on user requests."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
        
        self.template = PromptTemplate.from_template("""
        You are an expert Python ML engineer. The user wants to modify an existing ML pipeline.

        Current Configuration:
        - Task: {task_type}
        - Target: {target}
        - Features: {features}

        User's Modification Request:
        {tweak_prompt}

        Generate updated, production-ready scikit-learn code that incorporates the user's requested changes.

        Guidelines:
        1. Maintain the original task structure but implement requested modifications
        2. If the user wants to change algorithms, suggest and implement better alternatives
        3. If they want better preprocessing, add advanced techniques
        4. If they want better evaluation, add comprehensive metrics and visualization
        5. If they want hyperparameter tuning, add GridSearchCV or RandomizedSearchCV
        6. Include proper error handling and logging
        7. Make the code modular and well-documented
        8. Follow Python best practices

        Common modification patterns:
        - "Make it better" -> Add hyperparameter tuning, cross-validation, advanced preprocessing
        - "Use different algorithm" -> Implement ensemble methods or specific requested algorithms
        - "Add feature engineering" -> Implement feature selection, polynomial features, etc.
        - "Improve accuracy" -> Add advanced techniques like stacking, boosting
        - "Make it faster" -> Optimize preprocessing, use efficient algorithms

        Return ONLY the complete Python code without markdown formatting.
        """)
    
    def tweak_pipeline(self, tweak_prompt: str, task_type: str, target: str, features: List[str]) -> str:
        """
        Generate modified pipeline code based on user's tweaking request.
        
        Args:
            tweak_prompt: User's modification request
            task_type: 'classification' or 'regression'
            target: Target column name
            features: List of feature column names
            
        Returns:
            Modified Python code as string
        """
        try:
            logger.info(f"Tweaking pipeline based on request: {tweak_prompt[:100]}...")
            
            prompt = self.template.format(
                task_type=task_type,
                target=target,
                features=', '.join(features),
                tweak_prompt=tweak_prompt
            )
            
            response = self.model.invoke([HumanMessage(content=prompt)])
            tweaked_code = response.content
            
            # Clean up the generated code
            tweaked_code = self._clean_generated_code(tweaked_code)
            
            # Save to file
            self._save_code_to_file(tweaked_code, "tweaked_pipeline.py")
            
            logger.info("Pipeline tweaking completed successfully")
            return tweaked_code
            
        except Exception as e:
            logger.error(f"Error tweaking pipeline: {e}")
            return self._get_fallback_tweaked_code(tweak_prompt, task_type, target, features)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing markdown formatting."""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def _save_code_to_file(self, code: str, filename: str) -> None:
        """Save tweaked code to file."""
        try:
            filepath = OUTPUT_DIR / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Tweaked code saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving tweaked code: {e}")
    
    def _get_fallback_tweaked_code(self, tweak_prompt: str, task_type: str, target: str, features: List[str]) -> str:
        """Generate fallback tweaked code if LLM fails."""
        features_str = ', '.join([f'"{f}"' for f in features])
        
        # Analyze the tweak prompt for common patterns
        tweak_lower = tweak_prompt.lower()
        
        if any(keyword in tweak_lower for keyword in ['better', 'improve', 'optimize']):
            return self._generate_improved_pipeline(task_type, target, features_str)
        elif any(keyword in tweak_lower for keyword in ['hyperparameter', 'tuning', 'grid search']):
            return self._generate_tuned_pipeline(task_type, target, features_str)
        elif any(keyword in tweak_lower for keyword in ['ensemble', 'voting', 'stacking']):
            return self._generate_ensemble_pipeline(task_type, target, features_str)
        else:
            return self._generate_enhanced_pipeline(task_type, target, features_str)
    
    def _generate_improved_pipeline(self, task_type: str, target: str, features_str: str) -> str:
        """Generate an improved version of the pipeline."""
        if task_type == 'classification':
            algorithm = "RandomForestClassifier"
            metrics_import = "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
            evaluation = """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print('\\nClassification Report:')
    print(report)
    print('\\nConfusion Matrix:')
    print(cm)"""
        else:
            algorithm = "RandomForestRegressor"
            metrics_import = "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error"
            evaluation = """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')"""

        return f'''
"""
Enhanced ML Pipeline with Advanced Features
Task: {task_type.capitalize()}
Target: {target}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import {algorithm}
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
{metrics_import}
from sklearn.pipeline import Pipeline
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with advanced preprocessing and model selection."""
    
    def __init__(self, task_type="{task_type}"):
        self.task_type = task_type
        self.pipeline = None
        self.best_params = None
        self.cv_scores = None
    
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset with advanced techniques."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded: {{df.shape}}")
            
            features = [{features_str}]
            target = "{target}"
            
            X = df[features].copy()
            y = df[target].copy()
            
            # Advanced preprocessing
            # Handle missing values with strategy based on data type
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
            
            # Encode categorical features
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Feature scaling with RobustScaler (better for outliers)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
            
            logger.info("Advanced preprocessing completed")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {{e}}")
            raise
    
    def create_pipeline(self):
        """Create an advanced ML pipeline."""
        try:
            # Feature selection
            if self.task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k='all')
                estimator = {algorithm}(random_state=42)
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
                estimator = {algorithm}(random_state=42)
            
            # Create pipeline
            self.pipeline = Pipeline([
                ('selector', selector),
                ('estimator', estimator)
            ])
            
            logger.info("Advanced pipeline created")
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {{e}}")
            raise
    
    def train_with_hyperparameter_tuning(self, X, y):
        """Train model with hyperparameter tuning."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Define hyperparameters for tuning
            param_grid = {{
                'selector__k': [5, 10, 'all'],
                'estimator__n_estimators': [100, 200, 300],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10]
            }}
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=5,
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Starting hyperparameter tuning...")
            grid_search.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            self.pipeline = grid_search.best_estimator_
            
            # Make predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Evaluate model{evaluation}
            
            # Cross-validation
            self.cv_scores = cross_val_score(
                self.pipeline, X, y, cv=5,
                scoring='accuracy' if self.task_type == 'classification' else 'r2'
            )
            
            print(f"\\nCross-validation scores: {{self.cv_scores}}")
            print(f"Average CV score: {{self.cv_scores.mean():.4f}} (+/- {{self.cv_scores.std() * 2:.4f}})")
            print(f"\\nBest parameters: {{self.best_params}}")
            
            return X_test, y_test, y_pred
            
        except Exception as e:
            logger.error(f"Error in training: {{e}}")
            raise
    
    def save_model(self, filepath="enhanced_model.pkl"):
        """Save the trained model."""
        try:
            model_data = {{
                'pipeline': self.pipeline,
                'best_params': self.best_params,
                'cv_scores': self.cv_scores,
                'task_type': self.task_type
            }}
            joblib.dump(model_data, filepath)
            logger.info(f"Enhanced model saved to {{filepath}}")
        except Exception as e:
            logger.error(f"Error saving model: {{e}}")

def main():
    """Main execution function."""
    try:
        # Initialize pipeline
        ml_pipeline = EnhancedMLPipeline()
        
        # Load and preprocess data
        X, y = ml_pipeline.load_and_preprocess_data("your_dataset.csv")
        
        # Create and train pipeline
        ml_pipeline.create_pipeline()
        X_test, y_test, y_pred = ml_pipeline.train_with_hyperparameter_tuning(X, y)
        
        # Save model
        ml_pipeline.save_model()
        
        logger.info("Enhanced pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {{e}}")

if __name__ == "__main__":
    main()
'''

    def _generate_tuned_pipeline(self, task_type: str, target: str, features_str: str) -> str:
        """Generate a pipeline with extensive hyperparameter tuning."""
        # This would be a more complex implementation focusing on hyperparameter tuning
        return self._generate_improved_pipeline(task_type, target, features_str)
    
    def _generate_ensemble_pipeline(self, task_type: str, target: str, features_str: str) -> str:
        """Generate an ensemble pipeline."""
        # This would implement ensemble methods
        return self._generate_improved_pipeline(task_type, target, features_str)
    
    def _generate_enhanced_pipeline(self, task_type: str, target: str, features_str: str) -> str:
        """Generate a generally enhanced pipeline."""
        return self._generate_improved_pipeline(task_type, target, features_str)

# Legacy function for backward compatibility
def tweak_pipeline(tweak_prompt: str, task_type: str, target: str, features: List[str]) -> str:
    """Legacy function for pipeline tweaking."""
    tweaker = PipelineTweaker()
    return tweaker.tweak_pipeline(tweak_prompt, task_type, target, features)