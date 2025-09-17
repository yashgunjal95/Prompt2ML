# agents/code_gen.py
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

class CodeGenerator:
    """Generate Python ML pipeline code using LLM."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
        
        self.template = PromptTemplate.from_template("""
        You are an expert Python ML engineer. Generate a complete, production-ready scikit-learn pipeline.

        Task: {task_type}
        Target: {target}
        Features: {features}

        Requirements:
        1. Import all necessary libraries
        2. Data loading and preprocessing
        3. Feature engineering (if applicable)
        4. Model training with best practices
        5. Model evaluation with appropriate metrics
        6. Cross-validation
        7. Model persistence (save/load)
        8. Error handling
        9. Logging
        10. Documentation and comments

        Generate clean, well-structured Python code that follows PEP 8 standards.
        Include explanatory comments and docstrings.
        Make the code modular with functions for different steps.
        
        Return ONLY the Python code without any markdown formatting or explanation.
        """)
    
    def generate_pipeline_code(self, task_type: str, target: str, features: List[str]) -> str:
        """
        Generate ML pipeline code.
        
        Args:
            task_type: 'classification' or 'regression'
            target: Target column name
            features: List of feature column names
            
        Returns:
            Generated Python code as string
        """
        try:
            logger.info(f"Generating pipeline code for {task_type} task")
            
            prompt = self.template.format(
                task_type=task_type,
                target=target,
                features=', '.join(features)
            )
            
            response = self.model.invoke([HumanMessage(content=prompt)])
            generated_code = response.content
            
            # Clean up the generated code
            generated_code = self._clean_generated_code(generated_code)
            
            # Save to file
            self._save_code_to_file(generated_code, "generated_pipeline.py")
            
            logger.info("Pipeline code generated successfully")
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating pipeline code: {e}")
            return self._get_fallback_code(task_type, target, features)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing markdown formatting."""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _save_code_to_file(self, code: str, filename: str) -> None:
        """Save generated code to file."""
        try:
            filepath = OUTPUT_DIR / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Code saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving code to file: {e}")
    
    def _get_fallback_code(self, task_type: str, target: str, features: List[str]) -> str:
        """Generate fallback code template if LLM generation fails."""
        features_str = ', '.join([f'"{f}"' for f in features])
        
        if task_type == 'classification':
            model_import = "from sklearn.ensemble import RandomForestClassifier"
            model_init = "model = RandomForestClassifier(random_state=42)"
            metrics = "from sklearn.metrics import accuracy_score, classification_report"
            evaluation = """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)"""
        else:
            model_import = "from sklearn.ensemble import RandomForestRegressor"
            model_init = "model = RandomForestRegressor(random_state=42)"
            metrics = "from sklearn.metrics import mean_squared_error, r2_score"
            evaluation = """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'R² Score: {r2:.4f}')"""

        fallback_code = f'''
"""
Generated ML Pipeline for {task_type.capitalize()} Task
Target: {target}
Features: {features}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
{model_import}
{metrics}
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    try:
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded: {{df.shape}}")
        
        # Select features and target
        features = [{features_str}]
        target = "{target}"
        
        X = df[features].copy()
        y = df[target].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X, y, scaler
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {{e}}")
        raise

def train_model(X, y):
    """Train the machine learning model."""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        {model_init}
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model{evaluation}
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation scores: {{cv_scores}}")
        print(f"Average CV score: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")
        
        return model, X_test, y_test, y_pred
        
    except Exception as e:
        logger.error(f"Error in model training: {{e}}")
        raise

def save_model(model, scaler, filepath="model.pkl"):
    """Save the trained model and scaler."""
    try:
        joblib.dump({{'model': model, 'scaler': scaler}}, filepath)
        logger.info(f"Model saved to {{filepath}}")
    except Exception as e:
        logger.error(f"Error saving model: {{e}}")

def load_model(filepath="model.pkl"):
    """Load the trained model and scaler."""
    try:
        loaded = joblib.load(filepath)
        return loaded['model'], loaded['scaler']
    except Exception as e:
        logger.error(f"Error loading model: {{e}}")
        return None, None

def main():
    """Main pipeline execution."""
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_data("your_dataset.csv")
        
        # Train model
        model, X_test, y_test, y_pred = train_model(X, y)
        
        # Save model
        save_model(model, scaler)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        return fallback_code

# Legacy function for backward compatibility
def generate_pipeline_code(task_type: str, target: str, features: List[str]) -> str:
    """Legacy function for generating pipeline code."""
    generator = CodeGenerator()
    return generator.generate_pipeline_code(task_type, target, features)

