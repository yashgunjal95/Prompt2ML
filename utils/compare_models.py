# utils/compare_models.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare machine learning models side by side."""
    
    def __init__(self):
        self.comparison_history = []
    
    def compare_models(self, models_df: pd.DataFrame, model1: str, model2: str) -> pd.DataFrame:
        """
        Compare two models side by side with detailed analysis.
        
        Args:
            models_df: DataFrame containing model performance metrics
            model1: Name of the first model
            model2: Name of the second model
            
        Returns:
            DataFrame with detailed comparison
        """
        try:
            logger.info(f"Comparing models: {model1} vs {model2}")
            
            # Validate inputs
            if model1 not in models_df.index or model2 not in models_df.index:
                missing_models = [m for m in [model1, model2] if m not in models_df.index]
                raise ValueError(f"Model(s) not found in DataFrame: {missing_models}")
            
            if model1 == model2:
                raise ValueError("Cannot compare a model with itself")
            
            # Extract metrics for both models
            model1_metrics = models_df.loc[model1].copy()
            model2_metrics = models_df.loc[model2].copy()
            
            # Create comparison DataFrame
            comparison_data = {
                f"{model1}": model1_metrics,
                f"{model2}": model2_metrics
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Add difference and percentage difference columns
            comparison_df = self._add_difference_columns(comparison_df, model1, model2)
            
            # Add winner column
            comparison_df = self._add_winner_column(comparison_df, model1, model2)
            
            # Round numerical values for better display
            comparison_df = self._round_numerical_values(comparison_df)
            
            # Store comparison in history
            self._store_comparison(model1, model2, comparison_df)
            
            logger.info(f"Model comparison completed: {model1} vs {model2}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    def _add_difference_columns(self, df: pd.DataFrame, model1: str, model2: str) -> pd.DataFrame:
        """Add difference and percentage difference columns."""
        differences = []
        pct_differences = []
        
        for metric in df.index:
            try:
                val1 = pd.to_numeric(df.loc[metric, model1], errors='coerce')
                val2 = pd.to_numeric(df.loc[metric, model2], errors='coerce')
                
                if pd.isna(val1) or pd.isna(val2):
                    differences.append("N/A")
                    pct_differences.append("N/A")
                else:
                    diff = val2 - val1
                    differences.append(diff)
                    
                    if val1 != 0:
                        pct_diff = (diff / abs(val1)) * 100
                        pct_differences.append(pct_diff)
                    else:
                        pct_differences.append("N/A")
                        
            except (ValueError, TypeError):
                differences.append("N/A")
                pct_differences.append("N/A")
        
        df['Difference (B-A)'] = differences
        df['% Change'] = pct_differences
        
        return df
    
    def _add_winner_column(self, df: pd.DataFrame, model1: str, model2: str) -> pd.DataFrame:
        """Add a column indicating which model performs better for each metric."""
        winners = []
        
        # Define metrics where higher is better
        higher_is_better = [
            'accuracy', 'f1_score', 'precision', 'recall', 'r2_score', 'roc_auc',
            'Accuracy', 'F1 Score', 'Precision', 'Recall', 'R-Squared', 'ROC AUC'
        ]
        
        # Define metrics where lower is better
        lower_is_better = [
            'mae', 'mse', 'rmse', 'mape', 'time_taken',
            'MAE', 'MSE', 'RMSE', 'MAPE', 'Time Taken'
        ]
        
        for metric in df.index:
            try:
                val1 = pd.to_numeric(df.loc[metric, model1], errors='coerce')
                val2 = pd.to_numeric(df.loc[metric, model2], errors='coerce')
                
                if pd.isna(val1) or pd.isna(val2):
                    winners.append("N/A")
                elif any(keyword in metric for keyword in higher_is_better):
                    if val2 > val1:
                        winners.append(model2)
                    elif val1 > val2:
                        winners.append(model1)
                    else:
                        winners.append("Tie")
                elif any(keyword in metric for keyword in lower_is_better):
                    if val2 < val1:
                        winners.append(model2)
                    elif val1 < val2:
                        winners.append(model1)
                    else:
                        winners.append("Tie")
                else:
                    # For unknown metrics, assume higher is better
                    if val2 > val1:
                        winners.append(f"{model2} (assumed)")
                    elif val1 > val2:
                        winners.append(f"{model1} (assumed)")
                    else:
                        winners.append("Tie")
                        
            except (ValueError, TypeError):
                winners.append("N/A")
        
        df['Better Model'] = winners
        
        return df
    
    def _round_numerical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round numerical values for better display."""
        for col in df.columns:
            for idx in df.index:
                try:
                    val = df.loc[idx, col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        if abs(val) >= 1:
                            df.loc[idx, col] = round(val, 4)
                        else:
                            df.loc[idx, col] = round(val, 6)
                except:
                    pass
        
        return df
    
    def _store_comparison(self, model1: str, model2: str, comparison_df: pd.DataFrame) -> None:
        """Store comparison in history for future reference."""
        comparison_record = {
            'timestamp': pd.Timestamp.now(),
            'model1': model1,
            'model2': model2,
            'comparison': comparison_df.copy()
        }
        
        self.comparison_history.append(comparison_record)
        
        # Keep only last 10 comparisons
        if len(self.comparison_history) > 10:
            self.comparison_history = self.comparison_history[-10:]
    
    def get_comparison_summary(self, models_df: pd.DataFrame, model1: str, model2: str) -> Dict[str, Any]:
        """
        Get a summary of the comparison between two models.
        
        Returns:
            Dictionary with comparison summary
        """
        try:
            comparison_df = self.compare_models(models_df, model1, model2)
            
            # Count wins for each model
            winner_counts = comparison_df['Better Model'].value_counts()
            
            summary = {
                'model1': model1,
                'model2': model2,
                'total_metrics': len(comparison_df),
                'model1_wins': winner_counts.get(model1, 0),
                'model2_wins': winner_counts.get(model2, 0),
                'ties': winner_counts.get('Tie', 0),
                'na_values': winner_counts.get('N/A', 0)
            }
            
            # Determine overall winner
            if summary['model1_wins'] > summary['model2_wins']:
                summary['overall_winner'] = model1
            elif summary['model2_wins'] > summary['model1_wins']:
                summary['overall_winner'] = model2
            else:
                summary['overall_winner'] = 'Tie'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating comparison summary: {e}")
            return {}

# Legacy function for backward compatibility
def compare_models(models_df: pd.DataFrame, model1: str, model2: str) -> pd.DataFrame:
    """Legacy function for model comparison."""
    comparator = ModelComparator()
    return comparator.compare_models(models_df, model1, model2)