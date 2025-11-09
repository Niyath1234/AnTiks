#!/bin/bash

# Text-to-SQL Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "=========================================="
echo "Text-to-SQL Environment Setup"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment 'text_to_sql_env'..."
python3 -m venv text_to_sql_env

# Activate virtual environment
echo "Activating virtual environment..."
source text_to_sql_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements_text_to_sql.txt..."
if [ -f "requirements_text_to_sql.txt" ]; then
    pip install -r requirements_text_to_sql.txt
    echo ""
    echo "✅ All dependencies installed successfully!"
else
    echo "⚠️  Warning: requirements_text_to_sql.txt not found. Installing basic dependencies..."
    pip install torch transformers sentence-transformers chromadb pandas numpy tqdm accelerate
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; import transformers; import sentence_transformers; import chromadb; print('✅ All core packages imported successfully!')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source text_to_sql_env/bin/activate"
    echo ""
    echo "To deactivate, run:"
    echo "  deactivate"
    echo ""
    echo "To run the text-to-SQL system:"
    echo "  python text_to_sql_architecture.py"
    echo ""
else
    echo ""
    echo "⚠️  Some packages may not have installed correctly."
    echo "Please check the error messages above."
fi



