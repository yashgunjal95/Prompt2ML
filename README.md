# 🧠 Prompt2ML - AutoML Builder

An intelligent AutoML platform that converts natural language descriptions into complete machine learning pipelines.

## ✨ Features

- **Natural Language Interface**: Describe your ML task in plain English
- **Automated Model Training**: Train multiple models automatically using LazyPredict
- **Interactive Data Exploration**: Ask questions about your dataset using AI
- **Code Generation**: Generate production-ready Python code
- **Model Comparison**: Compare different models side-by-side
- **Pipeline Tweaking**: Modify and enhance your ML pipeline with natural language
- **Session Management**: Save and reload previous work sessions

## 🚀 Quick Start

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.8+
- GROQ API Key (for LLM functionality)
- Ollama (for local embeddings) or HuggingFace account

## 🔧 Usage

1. **Upload Dataset**: Upload your CSV file
2. **Describe Task**: Tell the system what you want to predict
3. **Automatic Processing**: The system will:
   - Clean and preprocess your data
   - Train multiple ML models
   - Evaluate and compare models
   - Generate production code
4. **Interact**: Ask questions and modify your pipeline as needed

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with scikit-learn and LazyPredict
- **LLM Integration**: Groq API for natural language processing
- **Vector Store**: Chroma for context retrieval
- **Database**: SQLite for session management

## 📁 Project Structure

```
prompt2ml/
├── app.py                 # Main Streamlit application
├── config/
│   └── settings.py        # Configuration settings
├── agents/
│   ├── prompt_parser.py   # Natural language prompt parsing
│   ├── chat_memory.py     # Interactive chat functionality
│   └── code_gen.py        # Code generation
├── ml_engine/
│   ├── preprocessing.py   # Data preprocessing
│   ├── auto_train.py      # AutoML training
│   └── evaluator.py       # Model evaluation
├── utils/
│   ├── compare_models.py  # Model comparison utilities
│   ├── tweak_pipeline.py  # Pipeline modification
│   └── logger.py          # Logging utilities
├── retriever/
│   └── vector_store.py    # Vector database operations
├── db/
│   └── session_store.py   # Session management
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
"""