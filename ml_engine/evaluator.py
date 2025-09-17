# ml_engine/evaluator.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, task_type: str, y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            task_type: 'classification' or 'regression'
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name} for {task_type} task")
            
            if task_type == 'classification':
                metrics = self._evaluate_classification(y_true, y_pred)
            else:
                metrics = self._evaluate_regression(y_true, y_pred)
            
            metrics['model_name'] = model_name
            metrics['task_type'] = task_type
            
            self.evaluation_results[model_name] = metrics
            logger.info(f"Evaluation completed for {model_name}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
    
    def _evaluate_classification(self, y_true, y_pred) -> Dict[str, Any]:
        """Evaluate classification model."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = class_report
            
            # Confusion matrix data
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            logger.info(f"Classification metrics - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
        
        return metrics
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, Any]:
        """Evaluate regression model."""
        metrics = {}
        
        try:
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # MAPE (handling division by zero)
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            except:
                metrics['mape'] = None
            
            logger.info(f"Regression metrics - R²: {metrics['r2_score']:.4f}, "
                       f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, task_type: str, 
                             model_name: str = "Model") -> Optional[plt.Figure]:
        """Generate confusion matrix plot for classification tasks."""
        if task_type != 'classification':
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted Labels', fontsize=12)
            ax.set_ylabel('True Labels', fontsize=12)
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            logger.info(f"Confusion matrix plot created for {model_name}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")
            return None
    
    def plot_regression_results(self, y_true, y_pred, model_name: str = "Model") -> Optional[plt.Figure]:
        """Generate regression evaluation plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot for residuals normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Regression Analysis - {model_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            logger.info(f"Regression plots created for {model_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regression plots: {e}")
            return None

# Legacy functions for backward compatibility
def plot_conf_matrix(y_true, y_pred, task_type: str) -> Optional[plt.Figure]:
    """Legacy function for confusion matrix plotting."""
    evaluator = ModelEvaluator()
    return evaluator.plot_confusion_matrix(y_true, y_pred, task_type)

def get_metrics(task_type: str, y_true, y_pred) -> Dict[str, Any]:
    """Legacy function for getting metrics."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(task_type, y_true, y_pred)