# Configuration for Narrative Consistency System

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BOOKS_DIR = DATA_DIR / "books"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_CSV = BASE_DIR / "results.csv"

# API Keys (load from environment or .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # Smaller NLI model (~140MB) for memory-constrained systems
USE_NLI_MODEL = True  # Enabled with smaller model - better accuracy with reasonable memory usage
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq model
LLM_PROVIDER = "groq"  # Options: "openai", "anthropic", "groq"
# Alternative models:
# LLM_MODEL = "gpt-4-turbo-preview"  # OpenAI
# LLM_MODEL = "claude-3-sonnet-20240229"  # Anthropic

# Chunking Configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 1500

# Retrieval Configuration
TOP_K_RETRIEVAL = 10  # Reduced from 20 to save memory
RETRIEVAL_THRESHOLD = 0.5  # Minimum similarity score

# Reasoning Configuration
USE_ADVERSARIAL = True  # Use prosecutor-defense-judge framework
USE_TEMPORAL_REASONING = True  # Track timelines
USE_CAUSAL_CHAINS = True  # Build causal graphs

# Scoring Weights (should sum to 1.0)
WEIGHT_CONTRADICTION = 0.3
WEIGHT_CAUSAL = 0.25
WEIGHT_CHARACTER = 0.2
WEIGHT_TEMPORAL = 0.15
WEIGHT_NARRATIVE = 0.1

# Decision Thresholds
CONSISTENCY_THRESHOLD = 0.5  # >= 0.5 → consistent (1), < 0.5 → contradict (0)

# Performance
BATCH_SIZE = 4
MAX_WORKERS = 4  # For parallel processing

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "pipeline.log"

# Cache
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
