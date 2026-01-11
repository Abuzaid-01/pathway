

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BOOKS_DIR = DATA_DIR / "books"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_CSV = BASE_DIR / "results.csv"

# API Keys - Support for MASSIVE rotation (10 Groq + 10 Gemini = 20 keys!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 10 Groq API Keys (Never run out!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")
GROQ_API_KEY_4 = os.getenv("GROQ_API_KEY_4", "")
GROQ_API_KEY_5 = os.getenv("GROQ_API_KEY_5", "")
GROQ_API_KEY_6 = os.getenv("GROQ_API_KEY_6", "")
GROQ_API_KEY_7 = os.getenv("GROQ_API_KEY_7", "")
GROQ_API_KEY_8 = os.getenv("GROQ_API_KEY_8", "")
GROQ_API_KEY_9 = os.getenv("GROQ_API_KEY_9", "")
GROQ_API_KEY_10 = os.getenv("GROQ_API_KEY_10", "")

# 10 Gemini API Keys (Massive backup!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3", "")
GEMINI_API_KEY_4 = os.getenv("GEMINI_API_KEY_4", "")
GEMINI_API_KEY_5 = os.getenv("GEMINI_API_KEY_5", "")
GEMINI_API_KEY_6 = os.getenv("GEMINI_API_KEY_6", "")
GEMINI_API_KEY_7 = os.getenv("GEMINI_API_KEY_7", "")
GEMINI_API_KEY_8 = os.getenv("GEMINI_API_KEY_8", "")
GEMINI_API_KEY_9 = os.getenv("GEMINI_API_KEY_9", "")
GEMINI_API_KEY_10 = os.getenv("GEMINI_API_KEY_10", "")

# Collect all available API keys for rotation (up to 20 keys!)
API_KEYS = {
    'groq': [k for k in [
        GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_API_KEY_4, GROQ_API_KEY_5,
        GROQ_API_KEY_6, GROQ_API_KEY_7, GROQ_API_KEY_8, GROQ_API_KEY_9, GROQ_API_KEY_10
    ] if k],
    'gemini': [k for k in [
        GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5,
        GEMINI_API_KEY_6, GEMINI_API_KEY_7, GEMINI_API_KEY_8, GEMINI_API_KEY_9, GEMINI_API_KEY_10
    ] if k],
    'openai': [OPENAI_API_KEY] if OPENAI_API_KEY else [],
}

# Model Configuration - ADVANCED FOR MAXIMUM ACCURACY
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"  
# NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"  
# NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  
USE_NLI_MODEL = True  # Enable for better accuracy

# LLM Configuration - ENHANCED with API calling and rotation
USE_LLM_API = True  # for deep reasoning
USE_API_ROTATION = True  # Rotate between multiple API keys
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_PROVIDER = "groq"  # Primary: "groq", Fallback: "gemini", "openai"
LLM_TEMPERATURE = 0.1  # Low temperature for consistent reasoning
LLM_MAX_TOKENS = 2000  # For comprehensive analysis
GEMINI_MODEL = "gemini-2.5-flash-exp"  # Updated to latest Gemini 2.5 Flash model

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 1500

TOP_K_RETRIEVAL = 35  # Increased from 20 for more context
RETRIEVAL_THRESHOLD = 0.20  # Lowered from 0.25 to catch more relevant chunks
USE_PATHWAY_VECTOR_STORE = True
PATHWAY_CACHE_BACKEND = ".cache/pathway_store"
RERANK_TOP_K = 15  # Increased from 10 for better re-ranking

# Reasoning Configuration
USE_ADVERSARIAL = True
USE_TEMPORAL_REASONING = True
USE_CAUSAL_CHAINS = True

# Scoring Weights - REBALANCED to fix 100% consistent issue
WEIGHT_CONTRADICTION = 0.30  # Increased - contradictions matter more
WEIGHT_CAUSAL = 0.15         # Causal reasoning
WEIGHT_CHARACTER = 0.12      # Character consistency
WEIGHT_TEMPORAL = 0.10       # Timeline coherence
WEIGHT_NARRATIVE = 0.08      # Overall fit
WEIGHT_LLM_JUDGMENT = 0.25   # LLM deep analysis

CONSISTENCY_THRESHOLD = 0.45  # Lowered from 0.50 to allow more contradict predictions
# If score >= 0.45 → Consistent, if < 0.45 → Contradict

# Performance
BATCH_SIZE = 4
MAX_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "pipeline.log"

# Cache
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


GENERATE_RATIONALE = True 
RATIONALE_MAX_LENGTH = 2000  
# EVALUATION METRICS - COMPREHENSIVE
EVALUATE_WITH_MULTIPLE_METRICS = True  # Accuracy, Precision, Recall, F1
SAVE_DETAILED_METRICS = True  # Save per-example analysis