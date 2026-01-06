#!/bin/bash

# Setup script for Narrative Consistency System
# Kharagpur Data Science Hackathon 2026

echo "========================================="
echo "Setting up Narrative Consistency System"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3.11 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3.11 not found. Please install Python 3.11 first."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create .env file template
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env << EOF
# API Keys (optional but recommended for better performance)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Note: You can leave these empty to use fallback methods
# But for best results, add at least one LLM API key
EOF
    echo ".env template created. Please add your API keys if available."
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python src/run.py"
echo ""
echo "Optional: Add API keys to .env file for better performance"
echo ""
