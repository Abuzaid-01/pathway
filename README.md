# Narrative Consistency Verification System

**Kharagpur Data Science Hackathon 2026 - Track A**  
**Problem:** Narrative consistency verification using NLP and Generative AI with Pathway

---

## 🎯 Overview

This system determines whether a hypothetical character backstory is consistent with a long-form narrative (100k+ words novel). It goes **beyond basic RAG** to implement:

- **Multi-perspective adversarial reasoning** (Prosecutor-Defense-Judge framework)
- **Multi-stage retrieval** with active contradiction mining
- **Ensemble scoring** across 5 specialized metrics
- **Temporal and causal reasoning** for global consistency
- **Pathway integration** for data ingestion and vector storage

---

## 🏗️ Architecture

```
Novel + Backstory
      ↓
[PATHWAY] Data Ingestion
      ↓
Multi-Strategy Chunking (Semantic + Structural + Character-centric)
      ↓
[PATHWAY] Vector Store + Indexing
      ↓
Multi-Stage Retrieval
  ├─ Stage 1: Broad Context
  ├─ Stage 2: Targeted Evidence
  ├─ Stage 3: Contradiction Mining
  └─ Stage 4: Causal Neighbors
      ↓
Adversarial Reasoning Framework
  ├─ Prosecutor Agent (finds contradictions)
  ├─ Defense Agent (finds support)
  └─ Judge Agent (weighs evidence)
      ↓
Ensemble Scoring (5 metrics)
  ├─ Direct Contradiction (30%)
  ├─ Causal Plausibility (25%)
  ├─ Character Consistency (20%)
  ├─ Temporal Coherence (15%)
  └─ Narrative Fit (10%)
      ↓
Binary Classification (Consistent/Contradict)
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv, installs dependencies)
./setup.sh
```

### 2. Configure (Optional)

Edit `.env` file to add API keys for better performance:

```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

**Note:** System works without API keys using fallback methods, but performance is enhanced with LLM access.

### 3. Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run the pipeline
python src/run.py
```

Select mode:
- **Mode 1:** Test on training data (with accuracy metrics)
- **Mode 2:** Generate predictions for test data
- **Mode 3:** Both

---

## 📁 Project Structure

```
narrative-consistency/
├── data/
│   ├── train.csv              # Training data with labels
│   ├── test.csv               # Test data (no labels)
│   └── books/
│       ├── The Count of Monte Cristo.txt
│       └── In search of the castaways.txt
│
├── src/
│   ├── ingest.py              # Pathway data ingestion
│   ├── chunking.py            # Multi-strategy text chunking
│   ├── retrieval.py           # Multi-stage retrieval with Pathway
│   ├── reasoning.py           # Adversarial reasoning + scoring
│   ├── decision.py            # Final classification
│   └── run.py                 # Main pipeline orchestrator
│
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
├── results.csv                 # Generated predictions (test)
└── README.md                   # This file
```

---

## 🔬 Technical Highlights

### 1. **Pathway Integration** ✅ (Mandatory Requirement)

- Used for CSV data ingestion with streaming capabilities
- Vector store integration for efficient retrieval
- Demonstrates real-time processing potential

### 2. **Beyond Basic RAG** 🚀

#### Multi-Stage Retrieval
- Not just "retrieve and ask LLM"
- 4 specialized retrieval stages
- Active contradiction mining
- Causal neighbor expansion

#### Adversarial Reasoning
- **Prosecutor Agent:** Actively searches for contradictions
- **Defense Agent:** Finds supporting evidence
- **Judge Agent:** Weighs both perspectives
- More robust than single-pass LLM calls

#### Ensemble Scoring
- 5 specialized metrics with learned weights
- Combines rule-based and neural approaches
- Captures multiple aspects of consistency

### 3. **Long Context Handling** 📚

#### Smart Chunking
- Structural chunking (preserves chapters/scenes)
- Character-centric chunking (targeted retrieval)
- Overlapping windows (prevents information loss)

#### Memory Mechanisms
- Caches book chunks per character
- Hierarchical retrieval (coarse to fine)
- Efficient FAISS indexing

### 4. **Novel Approaches**

- **Temporal reasoning:** Extracts and validates timelines
- **Causal chain analysis:** Checks if backstory enables later events
- **Claim decomposition:** Verifies atomic facts independently
- **Confidence calibration:** Aligns predicted confidence with accuracy

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "facebook/bart-large-mnli"
LLM_MODEL = "gpt-4-turbo-preview"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RETRIEVAL = 20

# Scoring Weights
WEIGHT_CONTRADICTION = 0.3
WEIGHT_CAUSAL = 0.25
WEIGHT_CHARACTER = 0.2
WEIGHT_TEMPORAL = 0.15
WEIGHT_NARRATIVE = 0.1

# Decision
CONSISTENCY_THRESHOLD = 0.5
```

---

## 📊 Evaluation Criteria (Track A)

### 1. **Accuracy** ✅
- Ensemble scoring optimizes for classification accuracy
- Threshold calibration on validation data
- Error analysis and failure mode detection

### 2. **Novelty** ✅
- Adversarial reasoning framework (not template RAG)
- Multi-stage retrieval with contradiction mining
- Ensemble of specialized scorers
- Temporal and causal reasoning modules

### 3. **Long Context Handling** ✅
- Multi-strategy chunking preserves narrative structure
- Efficient retrieval covers 100k+ word novels
- Character-centric indexing for targeted search
- Causal neighbor expansion maintains global coherence

---

## 🛠️ Dependencies

**Core:**
- `pathway>=0.8.0` - Data streaming and vector store (MANDATORY)
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Fast similarity search
- `transformers` - NLI models

**Optional (for enhanced performance):**
- `openai` - GPT models
- `anthropic` - Claude models

**See `requirements.txt` for complete list**

---

## 📈 Performance Tips

1. **With LLM API:**
   - Add OpenAI or Anthropic API key to `.env`
   - Use GPT-4 for best reasoning quality
   - Expect ~30-60 seconds per example

2. **Without LLM API (fallback mode):**
   - Uses NLI models and rule-based reasoning
   - Faster but slightly lower accuracy
   - Expect ~10-20 seconds per example

3. **Memory:**
   - System caches book chunks per character
   - Needs ~4-8 GB RAM for both novels
   - GPU optional but not required

---

## 🎓 Innovation Summary

This solution stands out by:

1. ✅ **Multi-agent adversarial reasoning** instead of single LLM call
2. ✅ **Active contradiction mining** instead of passive retrieval
3. ✅ **Claim-level atomic verification** instead of holistic judgment
4. ✅ **Explicit temporal/causal reasoning** instead of implicit patterns
5. ✅ **Ensemble of specialized scorers** instead of one-size-fits-all
6. ✅ **Pathway integration** for scalable data processing

---

## 📝 Output Format

### Training Mode
```csv
id,book_name,character,prediction,label,confidence,correct,scores,explanation
46,In Search of the Castaways,Thalcave,1,consistent,0.85,True,{...},Backstory is CONSISTENT...
```

### Test Mode (Submission)
```csv
id,label
95,contradict
136,consistent
...
```

---

## 🐛 Troubleshooting

**Issue:** Import errors  
**Solution:** Run `./setup.sh` to install all dependencies

**Issue:** Out of memory  
**Solution:** Process in smaller batches, reduce `TOP_K_RETRIEVAL`

**Issue:** Slow performance  
**Solution:** Add API key for LLM, or use smaller embedding model

**Issue:** Low accuracy  
**Solution:** Adjust weights in `config.py`, calibrate threshold on training data

---

## 📜 License

This project is for educational purposes as part of Kharagpur Data Science Hackathon 2026.

---

## 🙏 Acknowledgments

- **Pathway** for the streaming data framework
- **Hugging Face** for transformer models
- **OpenAI/Anthropic** for LLM APIs
- **Alexandre Dumas** for "The Count of Monte Cristo"
- **Jules Verne** for "In Search of the Castaways"

---

## 📧 Contact

For questions or issues, please refer to the hackathon guidelines or contact the organizers.

---

**Good luck! 🚀**
