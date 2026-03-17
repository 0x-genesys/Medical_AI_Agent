@echo off
REM Multimodal Medical Assistant - UI Launcher Script (Windows)
REM Automatically sets up environment and launches UI

echo 🏥 Multimodal Medical Assistant - UI Launcher
echo ==============================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CODEBASE_DIR=%PROJECT_ROOT%\Codebase
set VENV_DIR=%PROJECT_ROOT%\venv

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python 3 is not installed
    echo Please install Python 3.11 or higher
    exit /b 1
)
echo ✓ Python found

REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo.
    echo 📦 Virtual environment not found. Creating...
    cd /d "%PROJECT_ROOT%"
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment exists
)

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo ✓ Virtual environment activated

REM Check if requirements are installed
echo.
echo 📚 Checking dependencies...
cd /d "%CODEBASE_DIR%"

python -c "import gradio" 2>nul
if errorlevel 1 (
    echo ⚠️  Gradio not found. Installing UI dependencies...
    pip install gradio==4.16.0
    echo ✓ Gradio installed
) else (
    echo ✓ Gradio already installed
)

python -c "import sentence_transformers" 2>nul
if errorlevel 1 (
    echo ⚠️  Core dependencies not found. Installing all requirements...
    pip install -r requirements.txt
    echo ✓ All dependencies installed
) else (
    echo ✓ Core dependencies already installed
)

REM Check Ollama
echo.
echo 🤖 Checking Ollama LLM...
where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama not found!
    echo Please install Ollama from: https://ollama.com/download
    echo Then run: ollama pull llama3
    exit /b 1
)

REM Check if llama3 model is available
ollama list | findstr "llama3" >nul
if errorlevel 1 (
    echo ⚠️  Llama 3 model not found!
    echo Pulling Llama 3 model (this may take a few minutes)...
    ollama pull llama3
    echo ✓ Llama 3 model downloaded
) else (
    echo ✓ Llama 3 model available
)

REM Launch UI
echo.
echo 🚀 Launching Medical Assistant UI...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 🌐 UI will be available at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

REM Stay in Codebase directory to ensure models cache to same location as main.py
cd /d "%CODEBASE_DIR%"
python "%SCRIPT_DIR%\medical_assistant_ui.py"

REM Deactivate on exit
call deactivate
