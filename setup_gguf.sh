#!/bin/bash
# Setup script for GGUF optimization on M4 Mac

echo "=========================================="
echo "Setting up GGUF Optimization for M4 Mac"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

echo "📦 Installing llama-cpp-python with Metal support..."
echo ""

# Install llama-cpp-python with Metal backend
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade

echo ""
echo "✅ Installation complete!"
echo ""
echo "💡 To verify installation, run:"
echo "   python -c 'from llama_cpp import Llama; print(\"✅ llama-cpp-python installed\")'"
echo ""
echo "🚀 Now you can use the optimized version:"
echo "   python test_interactive_optimized.py"


