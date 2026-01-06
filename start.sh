#!/bin/bash

# Complete Getting Started Script
# Run this to go from zero to running system

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Narrative Consistency System - Complete Setup & Run         ║"
echo "║  Kharagpur Data Science Hackathon 2026                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
echo "Step 1/5: Checking prerequisites..."
echo "─────────────────────────────────────"

if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found"
    echo "Please install Python 3.11 first:"
    echo "  brew install python@3.11  (macOS)"
    exit 1
else
    echo "✓ Python 3.11 found: $(python3.11 --version)"
fi

# Step 2: Setup virtual environment
echo ""
echo "Step 2/5: Setting up virtual environment..."
echo "──────────────────────────────────────────"

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    ./setup.sh
fi

# Step 3: Activate and verify
echo ""
echo "Step 3/5: Verifying installation..."
echo "───────────────────────────────────"

source venv/bin/activate
./test_install.sh

if [ $? -ne 0 ]; then
    echo "❌ Installation verification failed"
    exit 1
fi

# Step 4: Check data files
echo ""
echo "Step 4/5: Checking data files..."
echo "────────────────────────────────"

if [ -f "data/train.csv" ]; then
    train_lines=$(wc -l < data/train.csv)
    echo "✓ train.csv found ($train_lines lines)"
else
    echo "❌ data/train.csv not found"
    exit 1
fi

if [ -f "data/test.csv" ]; then
    test_lines=$(wc -l < data/test.csv)
    echo "✓ test.csv found ($test_lines lines)"
else
    echo "❌ data/test.csv not found"
    exit 1
fi

if [ -f "data/books/The Count of Monte Cristo.txt" ]; then
    book1_lines=$(wc -l < "data/books/The Count of Monte Cristo.txt")
    echo "✓ The Count of Monte Cristo.txt found ($book1_lines lines)"
else
    echo "❌ The Count of Monte Cristo.txt not found"
    exit 1
fi

if [ -f "data/books/In search of the castaways.txt" ]; then
    book2_lines=$(wc -l < "data/books/In search of the castaways.txt")
    echo "✓ In search of the castaways.txt found ($book2_lines lines)"
else
    echo "❌ In search of the castaways.txt not found"
    exit 1
fi

# Step 5: Ready to run
echo ""
echo "Step 5/5: System ready!"
echo "──────────────────────"
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Data Summary:"
echo "   • Training examples: $(($train_lines - 1))"
echo "   • Test examples: $(($test_lines - 1))"
echo "   • Book 1 length: $book1_lines lines"
echo "   • Book 2 length: $book2_lines lines"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. (Optional) Add API key for better accuracy:"
echo "   nano .env"
echo "   # Add: OPENAI_API_KEY=sk-..."
echo ""
echo "2. Run the pipeline:"
echo "   python src/run.py"
echo ""
echo "3. Select mode when prompted:"
echo "   • Mode 1: Test on training data (see accuracy)"
echo "   • Mode 2: Generate test predictions (for submission)"
echo "   • Mode 3: Both"
echo ""
echo "📖 Documentation:"
echo "   • Quick Start: QUICKREF.md"
echo "   • Installation: INSTALL.md"
echo "   • Technical: INNOVATION.md"
echo "   • Architecture: ARCHITECTURE.md"
echo ""
echo "💡 Tips:"
echo "   • First run downloads models (~5-10 min)"
echo "   • Monitor progress: tail -f pipeline.log"
echo "   • Results saved to: results.csv"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Ready to win the hackathon! 🏆"
echo ""
read -p "Press Enter to start the pipeline now, or Ctrl+C to exit..."

# Run the pipeline
python src/run.py
