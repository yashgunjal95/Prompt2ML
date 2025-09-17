import logging
import sys
from pathlib import Path

def setup_logger(name: str = "prompt2ml", level: int = logging.INFO) -> logging.Logger:
    """Setup application logger with proper formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "prompt2ml.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger