# # Configuration for Narrative Consistency System

# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Base Paths
# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR / "data"
# BOOKS_DIR = DATA_DIR / "books"
# TRAIN_CSV = DATA_DIR / "train.csv"
# TEST_CSV = DATA_DIR / "test.csv"
# RESULTS_CSV = BASE_DIR / "results.csv"

# # API Keys (load from environment or .env file)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# # Model Configuration
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # Smaller NLI model (~140MB) for memory-constrained systems
# USE_NLI_MODEL = True  # Enabled with smaller model - better accuracy with reasonable memory usage
# LLM_MODEL = "llama-3.3-70b-versatile"  # Groq model
# LLM_PROVIDER = "groq"  # Options: "openai", "anthropic", "groq"
# # Alternative models:
# # LLM_MODEL = "gpt-4-turbo-preview"  # OpenAI
# # LLM_MODEL = "claude-3-sonnet-20240229"  # Anthropic

# # Chunking Configuration
# CHUNK_SIZE = 1000  # tokens
# CHUNK_OVERLAP = 200  # tokens
# MIN_CHUNK_SIZE = 500
# MAX_CHUNK_SIZE = 1500

# # Retrieval Configuration
# TOP_K_RETRIEVAL = 10  # Reduced from 20 to save memory
# RETRIEVAL_THRESHOLD = 0.5  # Minimum similarity score
# USE_PATHWAY_VECTOR_STORE = True  # Use Pathway's native vector store instead of FAISS
# PATHWAY_CACHE_BACKEND = ".cache/pathway_store"  # Pathway cache for incremental updates

# # Reasoning Configuration
# USE_ADVERSARIAL = True  # Use prosecutor-defense-judge framework
# USE_TEMPORAL_REASONING = True  # Track timelines
# USE_CAUSAL_CHAINS = True  # Build causal graphs

# # Scoring Weights (should sum to 1.0)
# WEIGHT_CONTRADICTION = 0.3
# WEIGHT_CAUSAL = 0.25
# WEIGHT_CHARACTER = 0.2
# WEIGHT_TEMPORAL = 0.15
# WEIGHT_NARRATIVE = 0.1

# # Decision Thresholds
# CONSISTENCY_THRESHOLD = 0.5  # >= 0.5 → consistent (1), < 0.5 → contradict (0)

# # Performance
# BATCH_SIZE = 4
# MAX_WORKERS = 4  # For parallel processing

# # Logging
# LOG_LEVEL = "INFO"
# LOG_FILE = BASE_DIR / "pipeline.log"

# # Cache
# CACHE_DIR = BASE_DIR / ".cache"
# CACHE_DIR.mkdir(exist_ok=True)

# Configuration for Narrative Consistency System
# UPDATED with better defaults based on 38.8% accuracy issue

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

# API Keys - Support for multiple keys with rotation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")  # Second Groq key for rotation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Google Gemini as alternative

# Collect all available API keys for rotation
API_KEYS = {
    'groq': [k for k in [GROQ_API_KEY, GROQ_API_KEY_2] if k],
    'gemini': [GEMINI_API_KEY] if GEMINI_API_KEY else [],
    'openai': [OPENAI_API_KEY] if OPENAI_API_KEY else [],
}

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# UPGRADED NLI MODEL - Larger model for better accuracy with GPU
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"  # CHANGED: large model (~660MB) - much better accuracy
# Alternative options:
# NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"  # Even larger (1.5GB) - best accuracy
# NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # Medium (~440MB)
USE_NLI_MODEL = True  # Enable for better accuracy

# LLM Configuration - ENHANCED with API calling and rotation
USE_LLM_API = True  # NEW: Enable LLM API for deep reasoning
USE_API_ROTATION = True  # Rotate between multiple API keys to avoid rate limits
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq (fast and free)
LLM_PROVIDER = "groq"  # Primary: "groq", Fallback: "gemini", "openai"
LLM_TEMPERATURE = 0.1  # Low temperature for consistent reasoning
LLM_MAX_TOKENS = 2000  # For comprehensive analysis
GEMINI_MODEL = "gemini-1.5-flash"  # Fast Gemini model as fallback

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 1500

# Retrieval Configuration - OPTIMIZED
TOP_K_RETRIEVAL = 20  # CHANGED: Increased for more evidence
RETRIEVAL_THRESHOLD = 0.25  # CHANGED: Lower threshold to get more candidates
USE_PATHWAY_VECTOR_STORE = True
PATHWAY_CACHE_BACKEND = ".cache/pathway_store"
RERANK_TOP_K = 10  # Re-rank top 10 from 20 retrieved

# Reasoning Configuration
USE_ADVERSARIAL = True
USE_TEMPORAL_REASONING = True
USE_CAUSAL_CHAINS = True

# Scoring Weights - ENHANCED WITH LLM
WEIGHT_CONTRADICTION = 0.20  # Direct contradictions
WEIGHT_CAUSAL = 0.20         # Causal reasoning
WEIGHT_CHARACTER = 0.20      # Character consistency
WEIGHT_TEMPORAL = 0.15       # Timeline coherence
WEIGHT_NARRATIVE = 0.10      # Overall fit
WEIGHT_LLM_JUDGMENT = 0.15   # NEW: LLM deep reasoning

# Decision Thresholds - CALIBRATED
CONSISTENCY_THRESHOLD = 0.50  # Balanced threshold
# Using better models + LLM should improve accuracy while maintaining balance

# Performance
BATCH_SIZE = 4
MAX_WORKERS = 4

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "pipeline.log"

# Cache
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# RATIONALE GENERATION (NEW)
GENERATE_RATIONALE = True  # Enable for Track B compliance
RATIONALE_MAX_LENGTH = 500  # Characters per rationale

# EVALUATION METRICS - COMPREHENSIVE
EVALUATE_WITH_MULTIPLE_METRICS = True  # Accuracy, Precision, Recall, F1
SAVE_DETAILED_METRICS = True  # Save per-example analysis