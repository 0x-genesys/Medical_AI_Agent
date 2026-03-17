#!/bin/bash

# Multimodal Medical Assistant - UI Launcher Script
# Automatically sets up environment and launches UI

set -e  # Exit on error

echo "🏥 Multimodal Medical Assistant - UI Launcher"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CODEBASE_DIR="$PROJECT_ROOT/Codebase"
VENV_DIR="$PROJECT_ROOT/venv"

# Check Python version
echo "📋 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Virtual environment not found. Creating..."
    cd "$PROJECT_ROOT"
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Check if requirements are installed
echo ""
echo "📚 Checking dependencies..."
cd "$CODEBASE_DIR"

# Check for key packages
if ! python -c "import gradio" 2>/dev/null; then
    echo "⚠️  Gradio not found. Installing UI dependencies..."
    pip install gradio==4.16.0
    echo "✓ Gradio installed"
else
    echo "✓ Gradio already installed"
fi

if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "⚠️  Core dependencies not found. Installing all requirements..."
    pip install -r requirements.txt
    echo "✓ All dependencies installed"
else
    echo "✓ Core dependencies already installed"
fi

# Check Ollama
echo ""
echo "🤖 Checking Ollama LLM..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found!"
    echo "Please install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
    echo "Then run: ollama pull llama3"
    exit 1
fi

# Check if llama3 model is available
if ! ollama list | grep -q "llama3"; then
    echo "⚠️  Llama 3 model not found!"
    echo "Pulling Llama 3 model (this may take a few minutes)..."
    ollama pull llama3
    echo "✓ Llama 3 model downloaded"
else
    echo "✓ Llama 3 model available"
fi

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🚀 Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
    echo "✓ Ollama service started"
else
    echo "✓ Ollama service running"
fi

# Launch UI
echo ""
echo "🚀 Launching Medical Assistant UI..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 UI will be available at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Stay in Codebase directory to ensure models cache to same location as main.py
cd "$CODEBASE_DIR"
python "$SCRIPT_DIR/medical_assistant_ui.py"

# Deactivate on exit
deactivate
