import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration"""
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Simple models (no API, no heavy dependencies)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'not-needed')
    EMBEDDING_MODEL = "simple-hash"
    LLM_MODEL = "template"
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '300'))
    
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.1'))
    TOP_K_CHUNKS = 3
    MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '5'))
    
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    DOCS_PATH = 'docs.json'
    
    @staticmethod
    def validate():
        """Validate configuration"""
        print("✅ Using lightweight local models - no dependencies!")