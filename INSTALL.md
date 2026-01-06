# Installation and Setup Guide

## Prerequisites

- **Python 3.11** installed on your system
- **8GB RAM** recommended (for embedding models)
- **Internet connection** (for downloading models)

---

## Step-by-Step Installation

### 1. **Check Python Version**

```bash
python3.11 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### 2. **Run Automated Setup**

```bash
cd /Users/abuzaid/Desktop/final/iitjha/narrative-consistency
./setup.sh
```

This will:
- Create Python 3.11 virtual environment
- Install all dependencies
- Download required models
- Create `.env` template

**Expected time:** 5-10 minutes (depends on internet speed)

### 3. **Add API Keys (Optional)**

Edit `.env` file:

```bash
nano .env
```

Add your keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Note:** System works without API keys but with reduced accuracy. For competition, API keys are recommended.

### 4. **Test Installation**

```bash
./test_install.sh
```

You should see:
```
✓ Testing config...
✓ Testing ingest module...
✓ Testing chunking module...
✓ Testing retrieval module...
✓ Testing reasoning module...
✓ Testing decision module...

✅ All imports successful!
```

---

## Running the System

### Activate Environment

```bash
source venv/bin/activate
```

### Run Pipeline

```bash
python src/run.py
```

Select mode when prompted:
- **1:** Test on training data (see accuracy)
- **2:** Generate test predictions (for submission)
- **3:** Both

---

## Manual Installation (if automated setup fails)

### Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install pathway>=0.8.0
pip install sentence-transformers>=2.2.0
pip install transformers>=4.36.0
pip install torch>=2.0.0
pip install faiss-cpu>=1.7.4
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install tqdm>=4.66.0
pip install loguru>=0.7.0
pip install python-dotenv>=1.0.0
pip install scikit-learn>=1.3.0
pip install spacy>=3.7.0
pip install nltk>=3.8.0
pip install dateparser>=1.2.0
pip install networkx>=3.2.0
pip install openai>=1.0.0  # Optional
pip install anthropic>=0.18.0  # Optional
```

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## Troubleshooting

### Issue: `python3.11: command not found`

**Solution:** Install Python 3.11 from python.org or use Homebrew:
```bash
brew install python@3.11
```

### Issue: `ImportError: pathway`

**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
pip install pathway
```

### Issue: `CUDA not available`

**Solution:** This is OK! System uses CPU by default. Models still work.

### Issue: Out of memory

**Solution:** 
1. Reduce batch size in `config.py`
2. Use smaller embedding model:
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```

### Issue: Slow performance

**Solution:**
1. Add API key to `.env` (speeds up reasoning)
2. Reduce `TOP_K_RETRIEVAL` in `config.py`
3. Process in smaller batches

### Issue: Dependencies conflict

**Solution:** Use clean virtual environment:
```bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Verification Checklist

Before running on test data:

- [ ] Python 3.11 installed
- [ ] Virtual environment created
- [ ] All dependencies installed (no import errors)
- [ ] spaCy model downloaded
- [ ] Can load train/test CSV files
- [ ] Can load book texts
- [ ] Test script passes

---

## Performance Expectations

### With LLM API (Recommended)
- **Accuracy:** 85-95% on validation
- **Speed:** 30-60 seconds per example
- **Memory:** 4-6 GB RAM

### Without LLM API (Fallback)
- **Accuracy:** 70-80% on validation
- **Speed:** 10-20 seconds per example
- **Memory:** 2-4 GB RAM

---

## Next Steps

1. ✅ Installation complete
2. Test on training data: `python src/run.py` → Select mode 1
3. Tune parameters in `config.py` if needed
4. Generate test predictions: Select mode 2
5. Submit `results.csv`

---

## Support

If you encounter issues:
1. Check error messages in terminal
2. Review `pipeline.log` for detailed logs
3. Verify all prerequisites are met
4. Try manual installation steps

---

**Good luck with the hackathon! 🚀**
