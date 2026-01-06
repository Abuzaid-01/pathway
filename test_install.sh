#!/bin/bash

# Quick test script to verify installation

echo "Testing Narrative Consistency System..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Test imports
echo "Testing imports..."
python << EOF
import sys
try:
    print("✓ Testing config...")
    import config
    
    print("✓ Testing ingest module...")
    from src.ingest import NarrativeDataIngester
    
    print("✓ Testing chunking module...")
    from src.chunking import MultiStrategyChunker
    
    print("✓ Testing retrieval module...")
    from src.retrieval import PathwayVectorStore
    
    print("✓ Testing reasoning module...")
    from src.reasoning import ConsistencyScoringEngine
    
    print("✓ Testing decision module...")
    from src.decision import DecisionAggregator
    
    print("")
    print("✅ All imports successful!")
    print("")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "System is ready! 🚀"
    echo "========================================"
    echo ""
    echo "To run the full pipeline:"
    echo "  python src/run.py"
    echo ""
    echo "To see detailed logs:"
    echo "  tail -f pipeline.log"
    echo ""
else
    echo "❌ Tests failed. Please check the error messages above."
    exit 1
fi
