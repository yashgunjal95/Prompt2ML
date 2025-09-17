# agents/prompt_parser.py
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import GROQ_API_KEY, GROQ_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

class PromptParser:
    """Parse natural language prompts to extract ML task information."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
        self.parser = JsonOutputParser()
        
        self.template = PromptTemplate.from_template("""
        You are an expert data scientist. Analyze the user's request and dataset columns to extract:
        1. Task type (classification or regression)
        2. Target column name
        3. Feature columns to use
        
        Dataset columns: {columns}
        
        User request: {prompt}
        
        Rules:
        - For classification: target should have discrete/categorical values
        - For regression: target should have continuous numerical values
        - Features should exclude the target column
        - All column names must exactly match the dataset columns
        - If unclear, make reasonable assumptions based on column names and user intent
        
        Return ONLY valid JSON in this exact format:
        {{
            "task_type": "classification" or "regression",
            "target": "exact_column_name",
            "features": ["col1", "col2", "col3"]
        }}
        """)
    
    def parse(self, prompt: str, dataframe: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Parse user prompt to extract ML task configuration.
        
        Args:
            prompt: User's natural language description
            dataframe: The dataset DataFrame
            
        Returns:
            Dictionary with task_type, target, and features, or None if parsing fails
        """
        try:
            # Prepare column information
            columns_info = self._get_column_info(dataframe)
            
            # Format the prompt
            formatted_prompt = self.template.format(
                columns=columns_info,
                prompt=prompt
            )
            
            # Get response from LLM
            response = self.model.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse JSON response
            result = self.parser.parse(response.content)
            
            # Validate the result
            validated_result = self._validate_result(result, dataframe)
            
            if validated_result:
                logger.info(f"Successfully parsed prompt: {validated_result}")
                return validated_result
            else:
                logger.warning("Prompt parsing validation failed")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            return None
    
    def _get_column_info(self, df: pd.DataFrame) -> str:
        """Get formatted column information for the prompt."""
        column_details = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().head(3).tolist()
            
            column_details.append(
                f"{col} ({dtype}, {unique_count} unique values, samples: {sample_values})"
            )
        
        return ", ".join(column_details)
    
    def _validate_result(self, result: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Validate the parsed result against the actual dataset."""
        try:
            # Check required keys
            required_keys = ['task_type', 'target', 'features']
            if not all(key in result for key in required_keys):
                logger.error(f"Missing required keys. Got: {list(result.keys())}")
                return None
            
            # Validate task type
            if result['task_type'] not in ['classification', 'regression']:
                logger.error(f"Invalid task type: {result['task_type']}")
                return None
            
            # Validate target column exists
            if result['target'] not in df.columns:
                logger.error(f"Target column '{result['target']}' not found in dataset")
                return None
            
            # Validate features exist and don't include target
            features = result['features']
            if not isinstance(features, list) or len(features) == 0:
                logger.error("Features must be a non-empty list")
                return None
            
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Features not found in dataset: {missing_features}")
                return None
            
            if result['target'] in features:
                logger.warning("Target column found in features, removing it")
                features = [f for f in features if f != result['target']]
                result['features'] = features
            
            # Additional validation based on task type
            target_col = df[result['target']]
            
            if result['task_type'] == 'classification':
                # For classification, target should have reasonable number of unique values
                unique_count = target_col.nunique()
                if unique_count > 50:
                    logger.warning(f"Target has {unique_count} unique values for classification task")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return None