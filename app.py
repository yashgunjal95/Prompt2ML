# app.py - Main Streamlit Application (Rewritten)
import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional
import json
from pathlib import Path

# Import all components
from config.settings import PROJECT_ROOT
from utils.logger import setup_logger
from db.session_store import SessionStore
from agents.prompt_parser import PromptParser
from ml_engine.preprocessing import DataPreprocessor
from retriever.vector_store import VectorStore
from agents.chat_memory import ChatMemory
from ml_engine.auto_train import AutoMLTrainer
from ml_engine.evaluator import ModelEvaluator
from agents.code_gen import CodeGenerator
from utils.compare_models import ModelComparator
from utils.tweak_pipeline import PipelineTweaker
import os
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logger = setup_logger()

class Prompt2MLApp:
    """Main Prompt2ML Streamlit Application."""
    
    def __init__(self):
        """Initialize the application with all components."""
        try:
            # Initialize all components
            self.session_store = SessionStore()
            self.prompt_parser = PromptParser()
            self.preprocessor = DataPreprocessor()
            self.vector_store = VectorStore()
            self.chat_memory = ChatMemory()
            self.automl_trainer = AutoMLTrainer()
            self.evaluator = ModelEvaluator()
            self.code_generator = CodeGenerator()
            self.model_comparator = ModelComparator()
            self.pipeline_tweaker = PipelineTweaker()
            
            # Initialize session state variables
            self._initialize_session_state()
            
            logger.info("Prompt2ML application initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            st.error(f"Failed to initialize application: {e}")
            st.stop()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        session_vars = {
            'df': None,
            'cleaned_df': None,
            'result': None,
            'models_df': None,
            'trained_models': None,
            'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
            'current_session_id': None
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def run(self) -> None:
        """Run the main application."""
        try:
            # Configure Streamlit page
            st.set_page_config(
                page_title="Prompt2ML - AutoML Builder",
                page_icon="🧠",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Main title
            st.title("🧠 Prompt2ML - AutoML Builder")
            st.markdown("---")
            
            # Sidebar for navigation and session management
            self._render_sidebar()
            
            # Main content area
            self._render_main_content()
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            st.error(f"Application error: {e}")
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with navigation and session management."""
        st.sidebar.title("🔧 Navigation")
        
        # Session management
        st.sidebar.subheader("📂 Session Management")
        
        if st.sidebar.button("🗑️ Clear Current Session"):
            self._clear_current_session()
            st.sidebar.success("Session cleared!")
        
        # Load previous sessions
        self._render_session_loader()
        
        # Application info
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "Prompt2ML is an intelligent AutoML builder that converts "
            "natural language descriptions into complete ML pipelines."
        )
        
        # Current session info
        if st.session_state.result:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Current Task")
            st.sidebar.write(f"**Task**: {st.session_state.result['task_type'].title()}")
            st.sidebar.write(f"**Target**: {st.session_state.result['target']}")
            st.sidebar.write(f"**Features**: {len(st.session_state.result['features'])}")
    
    def _render_session_loader(self) -> None:
        """Render the session loading interface."""
        if st.sidebar.checkbox("📋 View Previous Sessions"):
            sessions = self.session_store.fetch_sessions(limit=20)
            
            if sessions:
                st.sidebar.markdown("**Recent Sessions:**")
                
                for session in sessions:
                    session_id, prompt, summary, task_type, target, features, created_at = session
                    
                    # Create a short description
                    short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    session_label = f"Session {session_id}: {target} ({task_type})"
                    
                    if st.sidebar.button(session_label, key=f"load_{session_id}"):
                        self._load_session(session)
                        st.experimental_rerun()
            else:
                st.sidebar.info("No previous sessions found.")
    
    def _render_main_content(self) -> None:
        """Render the main content area."""
        # Phase 1: Data Upload and Prompt
        st.header("📁 Phase 1: Data Upload & Task Description")
        df, prompt = self._handle_data_upload_and_prompt()
        
        if df is not None and prompt:
            st.session_state.df = df
            
            # Phase 2: Prompt Analysis
            st.header("🔍 Phase 2: Task Analysis")
            result = self._handle_prompt_analysis(prompt, df)
            
            if result:
                st.session_state.result = result
                
                # Phase 3: Data Preprocessing
                st.header("🧹 Phase 3: Data Preprocessing")
                cleaned_df = self._handle_data_preprocessing(df, result)
                
                if cleaned_df is not None:
                    st.session_state.cleaned_df = cleaned_df
                    
                    # Phase 4: Interactive Chat
                    st.header("💬 Phase 4: Ask Questions About Your Data")
                    self._handle_interactive_chat(result)
                    
                    # Phase 5: AutoML Training
                    st.header("🤖 Phase 5: AutoML Training")
                    automl_results = self._handle_automl_training(cleaned_df, result)
                    
                    if automl_results[0] is not None:
                        models_df, trained_models, X_train, X_test, y_train, y_test = automl_results
                        
                        # Store in session state
                        st.session_state.models_df = models_df
                        st.session_state.trained_models = trained_models
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        # Phase 6: Model Evaluation
                        st.header("📊 Phase 6: Model Evaluation")
                        self._handle_model_evaluation(models_df, trained_models, X_test, y_test, result)
                        
                        # Phase 7: Code Generation
                        st.header("🧾 Phase 7: Code Generation")
                        self._handle_code_generation(result)
                        
                        # Phase 8: Model Comparison
                        st.header("⚖️ Phase 8: Model Comparison")
                        self._handle_model_comparison(models_df)
                        
                        # Phase 9: Pipeline Tweaking
                        st.header("🔧 Phase 9: Pipeline Tweaking")
                        self._handle_pipeline_tweaking(result)
    
    def _handle_data_upload_and_prompt(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Handle file upload and prompt input."""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📂 Upload your CSV dataset",
                type=["csv"],
                help="Upload a CSV file containing your dataset"
            )
        
        with col2:
            prompt = st.text_area(
                "📝 Describe what you want the model to do",
                height=100,
                placeholder="E.g., 'Predict house prices based on location and size' or 'Classify emails as spam or not spam'"
            )
        
        df = None
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Display dataset info
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("📋 Dataset Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.subheader("📊 Dataset Info")
                    st.write(f"**Rows**: {df.shape[0]:,}")
                    st.write(f"**Columns**: {df.shape[1]}")
                    st.write(f"**Missing Values**: {df.isnull().sum().sum()}")
                    st.write(f"**Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    # Column types
                    st.subheader("🏷️ Column Types")
                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f"**{dtype}**: {count} columns")
                
            except Exception as e:
                st.error(f"❌ Error loading CSV file: {e}")
                logger.error(f"Error loading CSV: {e}")
                return None, None
        
        return df, prompt
    
    def _handle_prompt_analysis(self, prompt: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Handle prompt analysis and task extraction."""
        with st.spinner("🔍 Analyzing your prompt..."):
            try:
                result = self.prompt_parser.parse(prompt, df)
                
                if result:
                    st.success("✅ Prompt analyzed successfully!")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("🎯 Extracted Task Information")
                        st.write(f"**Task Type**: `{result['task_type'].title()}`")
                        st.write(f"**Target Column**: `{result['target']}`")
                        st.write(f"**Number of Features**: {len(result['features'])}")
                    
                    with col2:
                        st.subheader("📋 Selected Features")
                        for i, feature in enumerate(result['features'], 1):
                            st.write(f"{i}. `{feature}`")
                    
                    # Validation info
                    target_info = df[result['target']].describe() if result['target'] in df.columns else None
                    if target_info is not None:
                        st.subheader("🎯 Target Variable Analysis")
                        
                        if result['task_type'] == 'classification':
                            value_counts = df[result['target']].value_counts()
                            st.write(f"**Unique Classes**: {len(value_counts)}")
                            
                            if len(value_counts) <= 10:
                                st.bar_chart(value_counts)
                        else:
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.write(target_info)
                            with col2:
                                st.line_chart(df[result['target']].head(100))
                    
                    return result
                else:
                    st.error("❌ Failed to analyze prompt. Please try rephrasing your request.")
                    st.info("💡 Try being more specific about what you want to predict and which columns to use.")
                    return None
                    
            except Exception as e:
                st.error(f"❌ Error analyzing prompt: {e}")
                logger.error(f"Prompt analysis error: {e}")
                return None
    
    def _handle_data_preprocessing(self, df: pd.DataFrame, result: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Handle data preprocessing."""
        with st.spinner("🧹 Cleaning and preprocessing data..."):
            try:
                # Preprocessing options
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    missing_strategy = st.selectbox(
                        "🔧 Missing Value Strategy",
                        ["drop", "impute"],
                        index=1,
                        help="Choose how to handle missing values"
                    )
                
                with col2:
                    st.write(f"**Missing values per column:**")
                    missing_counts = df[result['features'] + [result['target']]].isnull().sum()
                    missing_counts = missing_counts[missing_counts > 0]
                    if len(missing_counts) > 0:
                        st.write(missing_counts)
                    else:
                        st.write("No missing values found! ✅")
                
                # Perform preprocessing
                cleaned_df = self.preprocessor.clean_data(
                    df, result['target'], result['features'], missing_strategy
                )
                
                st.success("✅ Data preprocessing completed!")
                
                # Show preprocessing results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Before vs After")
                    comparison_df = pd.DataFrame({
                        'Before': [df.shape[0], df.shape[1], df.isnull().sum().sum()],
                        'After': [cleaned_df.shape[0], cleaned_df.shape[1], cleaned_df.isnull().sum().sum()]
                    }, index=['Rows', 'Columns', 'Missing Values'])
                    st.dataframe(comparison_df)
                
                with col2:
                    st.subheader("🔍 Cleaned Data Preview")
                    st.dataframe(cleaned_df.head(), use_container_width=True)
                
                # Store context for retrieval
                summary = f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns. Target: {result['target']}. Features: {result['features']}"
                self.vector_store.store_context(result.get('original_prompt', ''), summary)
                
                return cleaned_df
                
            except Exception as e:
                st.error(f"❌ Error preprocessing data: {e}")
                logger.error(f"Data preprocessing error: {e}")
                return None
    
    def _handle_interactive_chat(self, result: Dict[str, Any]) -> None:
        """Handle interactive chat about the dataset."""
        st.subheader("🤖 AI Assistant")
        st.write("Ask questions about your dataset, task, or machine learning approach!")
        
        # Chat input
        user_question = st.text_input(
            "💭 Your question:",
            placeholder="e.g., 'What does this target variable represent?' or 'What algorithms would work best?'"
        )
        
        if user_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = self.chat_memory.ask_question(user_question)
                    
                    st.success("🤖 **AI Response:**")
                    st.write(response['answer'])
                    
                    # Save interaction to session store
                    summary = f"Target: {result['target']}, Features: {', '.join(result['features'])}"
                    session_id = self.session_store.save_session(
                        user_question, summary, result['task_type'], result['target'], result['features']
                    )
                    st.session_state.current_session_id = session_id
                    
                except Exception as e:
                    st.error(f"❌ Error processing your question: {e}")
                    logger.error(f"Chat error: {e}")
    
    def _handle_automl_training(self, cleaned_df: pd.DataFrame, result: Dict[str, Any]) -> Tuple:
        """Handle AutoML training process."""
        with st.spinner("🤖 Training multiple ML models..."):
            try:
                models_df, trained_models, X_train, X_test, y_train, y_test = self.automl_trainer.run_automl(
                    cleaned_df, result['task_type'], result['target']
                )
                
                st.success("✅ AutoML training completed!")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🏆 Model Performance Leaderboard")
                    st.dataframe(
                        models_df.round(4),
                        use_container_width=True,
                        height=400
                    )
                
                with col2:
                    st.subheader("📊 Training Summary")
                    st.metric("Models Trained", len(models_df))
                    st.metric("Training Set Size", len(X_train))
                    st.metric("Test Set Size", len(X_test))
                    
                    if result['task_type'] == 'classification':
                        best_score = models_df.iloc[0]['Accuracy'] if 'Accuracy' in models_df.columns else 'N/A'
                        st.metric("Best Accuracy", f"{best_score:.4f}" if best_score != 'N/A' else 'N/A')
                    else:
                        best_score = models_df.iloc[0]['R-Squared'] if 'R-Squared' in models_df.columns else 'N/A'
                        st.metric("Best R² Score", f"{best_score:.4f}" if best_score != 'N/A' else 'N/A')
                
                return models_df, trained_models, X_train, X_test, y_train, y_test
                
            except Exception as e:
                st.error(f"❌ Error in AutoML training: {e}")
                logger.error(f"AutoML training error: {e}")
                return None, None, None, None, None, None
    
    def _handle_model_evaluation(self, models_df: pd.DataFrame, trained_models: Dict, 
                                X_test, y_test, result: Dict[str, Any]) -> None:
        """Handle model evaluation and visualization."""
        if models_df.empty or not trained_models:
            st.warning("⚠️ No models available for evaluation.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_model = st.selectbox(
                "🎯 Select Model for Detailed Evaluation",
                options=list(trained_models.keys()),
                help="Choose a model to see detailed evaluation metrics and visualizations"
            )
        
        with col2:
            st.write("**Available Models:**")
            st.write(f"✅ {len(trained_models)} models trained")
            st.write(f"🏆 Best model: {models_df.index[0]}")
        
        if selected_model:
            try:
                # Get model and make predictions
                model = trained_models[selected_model]
                y_pred = model.predict(X_test)
                
                # Evaluate model
                metrics = self.evaluator.evaluate_model(
                    result['task_type'], y_test, y_pred, selected_model
                )
                
                # Display metrics
                st.subheader(f"📊 Detailed Metrics - {selected_model}")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if result['task_type'] == 'classification':
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                    else:
                        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                
                with col2:
                    # Model info from leaderboard
                    model_info = models_df.loc[selected_model] if selected_model in models_df.index else None
                    if model_info is not None:
                        st.write("**Leaderboard Stats:**")
                        for metric, value in model_info.items():
                            if isinstance(value, (int, float)):
                                st.write(f"**{metric}**: {value:.4f}")
                            else:
                                st.write(f"**{metric}**: {value}")
                
                # Visualizations
                st.subheader("📈 Model Visualizations")
                
                if result['task_type'] == 'classification':
                    # Confusion matrix
                    fig = self.evaluator.plot_confusion_matrix(y_test, y_pred, result['task_type'], selected_model)
                    if fig:
                        st.pyplot(fig)
                else:
                    # Regression plots
                    fig = self.evaluator.plot_regression_results(y_test, y_pred, selected_model)
                    if fig:
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error evaluating model {selected_model}: {e}")
                logger.error(f"Model evaluation error: {e}")
    
    def _handle_code_generation(self, result: Dict[str, Any]) -> None:
        """Handle code generation for the ML pipeline."""
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🚀 Generate Pipeline Code", type="primary"):
                with st.spinner("🔧 Generating production-ready code..."):
                    try:
                        code = self.code_generator.generate_pipeline_code(
                            result['task_type'], result['target'], result['features']
                        )
                        
                        st.session_state['generated_code'] = code
                        st.success("✅ Code generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating code: {e}")
                        logger.error(f"Code generation error: {e}")
        
        with col1:
            st.write("Generate a complete, production-ready Python script for your ML pipeline!")
        
        # Display generated code
        if 'generated_code' in st.session_state and st.session_state['generated_code']:
            st.subheader("🧾 Generated Pipeline Code")
            
            # Code display with syntax highlighting
            st.code(st.session_state['generated_code'], language='python', line_numbers=True)
            
            # Download button
            st.download_button(
                label="📥 Download Python Code",
                data=st.session_state['generated_code'],
                file_name=f"{result['task_type']}_pipeline_{result['target']}.py",
                mime="text/x-python"
            )
    
    def _handle_model_comparison(self, models_df: pd.DataFrame) -> None:
        """Handle model comparison functionality."""
        if len(models_df) < 2:
            st.info("ℹ️ Need at least 2 models for comparison.")
            return
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            model_a = st.selectbox(
                "🥊 Model A",
                options=models_df.index.tolist(),
                key="model_comparison_a"
            )
        
        with col2:
            model_b = st.selectbox(
                "🥊 Model B",
                options=models_df.index.tolist(),
                index=1 if len(models_df) > 1 else 0,
                key="model_comparison_b"
            )
        
        with col3:
            if st.button("⚖️ Compare Models", type="primary"):
                if model_a != model_b:
                    try:
                        comparison_df = self.model_comparator.compare_models(models_df, model_a, model_b)
                        summary = self.model_comparator.get_comparison_summary(models_df, model_a, model_b)
                        
                        st.session_state['comparison_result'] = comparison_df
                        st.session_state['comparison_summary'] = summary
                        
                    except Exception as e:
                        st.error(f"❌ Error comparing models: {e}")
                        logger.error(f"Model comparison error: {e}")
                else:
                    st.warning("⚠️ Please select different models for comparison.")
        
        # Display comparison results
        if 'comparison_result' in st.session_state and st.session_state['comparison_result'] is not None:
            st.subheader("📊 Model Comparison Results")
            
            # Summary
            if 'comparison_summary' in st.session_state:
                summary = st.session_state['comparison_summary']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Metrics", summary.get('total_metrics', 0))
                with col2:
                    st.metric(f"{model_a} Wins", summary.get('model1_wins', 0))
                with col3:
                    st.metric(f"{model_b} Wins", summary.get('model2_wins', 0))
                with col4:
                    st.metric("Overall Winner", summary.get('overall_winner', 'N/A'))
            
            # Detailed comparison
            st.subheader("🔍 Detailed Comparison")
            st.dataframe(st.session_state['comparison_result'], use_container_width=True)
    
    def _handle_pipeline_tweaking(self, result: Dict[str, Any]) -> None:
        """Handle pipeline tweaking functionality."""
        st.subheader("🔧 Customize Your Pipeline")
        st.write("Describe how you'd like to modify or improve your ML pipeline:")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            tweak_prompt = st.text_area(
                "✨ Modification Request",
                height=100,
                placeholder="e.g., 'Use ensemble methods for better accuracy', 'Add hyperparameter tuning', 'Include feature engineering', etc."
            )
        
        with col2:
            st.write("**Common requests:**")
            st.write("• Better accuracy")
            st.write("• Hyperparameter tuning")
            st.write("• Ensemble methods")
            st.write("• Feature engineering")
            st.write("• Cross-validation")
            st.write("• Different algorithms")
        
        if st.button("♻️ Generate Enhanced Pipeline", type="primary") and tweak_prompt:
            with st.spinner("🔄 Creating enhanced pipeline..."):
                try:
                    enhanced_code = self.pipeline_tweaker.tweak_pipeline(
                        tweak_prompt, result['task_type'], result['target'], result['features']
                    )
                    
                    st.session_state['enhanced_code'] = enhanced_code
                    st.success("✅ Enhanced pipeline generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating enhanced pipeline: {e}")
                    logger.error(f"Pipeline tweaking error: {e}")
        
        # Display enhanced code
        if 'enhanced_code' in st.session_state and st.session_state['enhanced_code']:
            st.subheader("🚀 Enhanced Pipeline Code")
            
            # Show the tweaked code
            st.code(st.session_state['enhanced_code'], language='python', line_numbers=True)
            
            # Download button for enhanced code
            st.download_button(
                label="📥 Download Enhanced Code",
                data=st.session_state['enhanced_code'],
                file_name=f"enhanced_{result['task_type']}_pipeline.py",
                mime="text/x-python"
            )
    
    def _load_session(self, session_data: Tuple) -> None:
        """Load a previous session."""
        try:
            session_id, prompt, summary, task_type, target, features_json, created_at = session_data
            
            # Parse features
            features = json.loads(features_json) if isinstance(features_json, str) else features_json
            
            # Reconstruct session state
            st.session_state.result = {
                'task_type': task_type,
                'target': target,
                'features': features,
                'original_prompt': prompt
            }
            
            st.session_state.current_session_id = session_id
            
            st.success(f"✅ Session {session_id} loaded successfully!")
            st.info(f"**Task**: {task_type.title()}, **Target**: {target}")
            
        except Exception as e:
            st.error(f"❌ Error loading session: {e}")
            logger.error(f"Session loading error: {e}")
    
    def _clear_current_session(self) -> None:
        """Clear the current session state."""
        keys_to_clear = [
            'df', 'cleaned_df', 'result', 'models_df', 'trained_models',
            'X_train', 'X_test', 'y_train', 'y_test', 'current_session_id',
            'generated_code', 'enhanced_code', 'comparison_result', 'comparison_summary'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear chat memory
        try:
            self.chat_memory.clear_memory()
        except:
            pass

def main():
    """Main application entry point."""
    try:
        app = Prompt2MLApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start Prompt2ML application: {e}")
        logger.error(f"Application startup error: {e}")

if __name__ == "__main__":
    main()
