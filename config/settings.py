import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_DIR = PROJECT_ROOT / "db"
CHROMA_DIR = PROJECT_ROOT / "chroma_data"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, DB_DIR, CHROMA_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # More reliable model

# Database configuration
DB_PATH = DB_DIR / "sessions.db"

# Model configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
