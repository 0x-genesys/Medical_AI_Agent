#!/usr/bin/env python3
"""
Multimodal Medical Assistant - Universal Setup & Launcher
Automatically sets up environment and provides CLI interface

Usage:
    python run_main.py              # Interactive menu
    python run_main.py --ui         # Launch UI dashboard
    python run_main.py --cli        # Launch CLI interface
    python run_main.py --test       # Run tests
    python run_main.py --setup      # Setup only (no launch)
"""

# CRITICAL: Set OpenMP environment BEFORE any imports
# Must be first to prevent multiple OpenMP library crashes
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import subprocess
import platform
import argparse
import time
from pathlib import Path
import shutil

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print("=" * len(text))

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

class MedicalAssistantSetup:
    """
    Automated setup and launcher for Medical Assistant.
    
    Handles complete environment setup including Python version checking,
    virtual environment creation, dependency installation, Ollama setup,
    and application launching. Supports multiple platforms (Windows, macOS,
    Linux, Google Colab).
    
    Attributes:
        script_dir (Path): Directory containing this script (Codebase/)
        project_root (Path): Root project directory
        venv_dir (Path): Virtual environment directory
        ui_dir (Path): UI dashboard directory
        is_windows (bool): Whether running on Windows
        is_colab (bool): Whether running in Google Colab
        python_cmd (str): Python command ('python' or 'python3')
    """
    
    def __init__(self):
        """
        Initialize setup manager with platform detection.
        
        Detects platform, sets up directory paths, and determines
        appropriate Python command for the environment.
        
        Returns:
            None
        """
        self.script_dir = Path(__file__).parent.resolve()
        self.project_root = self.script_dir.parent
        self.venv_dir = self.project_root / "venv"
        self.ui_dir = self.script_dir / "ui-dashboard"
        self.is_windows = platform.system() == "Windows"
        self.python_cmd = "python" if self.is_windows else "python3"
        
        # Detect Google Colab environment
        self.is_colab = self._detect_colab()
    
    def _detect_colab(self):
        """
        Detect if running in Google Colab environment.
        
        Attempts to import google.colab module to determine if
        code is running in Colab notebook environment.
        
        Returns:
            bool: True if running in Colab, False otherwise
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
        
    def check_python_version(self):
        """
        Check if Python version meets minimum requirements.
        
        Verifies that Python 3.9 or higher is installed. Prints
        version information and installation guidance if version
        is too old.
        
        Returns:
            bool: True if Python version is compatible (3.9+), False otherwise
        """
        print_header("📋 Checking Python Version")
        
        try:
            version_info = sys.version_info
            version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            
            if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
                print_error(f"Python {version_str} is too old")
                print_info("This application requires Python 3.9 or higher")
                print_info("Please install Python 3.11+ from https://www.python.org/downloads/")
                return False
            
            print_success(f"Python {version_str} detected")
            return True
            
        except Exception as e:
            print_error(f"Could not determine Python version: {e}")
            return False
    
    def setup_virtual_environment(self):
        """Create and activate virtual environment"""
        print_header("📦 Setting Up Virtual Environment")
        
        if self.is_colab:
            print_info("Running in Google Colab - using system Python")
            print_success("✓ Virtual environment not needed in Colab")
            return True
        
        if self.venv_dir.exists():
            print_success("Virtual environment already exists")
            return True
        
        try:
            print_info("Creating virtual environment...")
            subprocess.run([self.python_cmd, "-m", "venv", str(self.venv_dir)], check=True)
            print_success("Virtual environment created")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to create virtual environment: {e}")
            print_info("Try running: python -m ensurepip --upgrade")
            return False
        except Exception as e:
            print_error(f"Unexpected error creating venv: {e}")
            return False
    
    def get_venv_python(self):
        """Get path to Python executable in venv"""
        if self.is_colab:
            return sys.executable  # Use system Python in Colab
        if self.is_windows:
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:
            return str(self.venv_dir / "bin" / "python")
    
    def get_venv_pip(self):
        """Get path to pip in venv"""
        if self.is_colab:
            return "pip"  # Use system pip in Colab
        if self.is_windows:
            return str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            return str(self.venv_dir / "bin" / "pip")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print_header("📚 Installing Dependencies")
        
        pip_cmd = self.get_venv_pip()
        requirements_file = self.script_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print_error("requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            print_info("Upgrading pip...")
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True, capture_output=True)
            
            # Check if key packages are already installed
            print_info("Checking installed packages...")
            result = subprocess.run([pip_cmd, "list"], capture_output=True, text=True)
            installed_packages = result.stdout.lower()
            
            if "gradio" in installed_packages and "transformers" in installed_packages:
                print_success("Core dependencies already installed")
                return True
            
            # Install all requirements
            print_info("Installing requirements (this may take several minutes)...")
            print_info("Installing PyTorch, transformers, and medical models...")
            
            process = subprocess.Popen(
                [pip_cmd, "install", "-r", str(requirements_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.script_dir)
            )
            
            # Show progress
            for line in process.stdout:
                if "Collecting" in line or "Installing" in line or "Successfully" in line:
                    print(f"  {line.strip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print_success("All dependencies installed successfully")
                
                # Consolidate OpenMP libraries to prevent crashes
                if not self.consolidate_openmp():
                    print_warning("OpenMP consolidation failed, but continuing...")
                
                return True
            else:
                print_error("Failed to install dependencies")
                return False
                
        except Exception as e:
            print_error(f"Error installing dependencies: {e}")
            print_info("Try manually: pip install -r requirements.txt")
            return False
    
    def consolidate_openmp(self):
        """
        Configure OpenMP environment variables to prevent crashes
        
        SAFER APPROACH: Uses environment variables instead of file manipulation
        This avoids risky deletion/symlinking of library files
        """
        print_header("🔧 Configuring OpenMP Environment")
        
        try:
            # Find site-packages in venv
            venv_python = self.get_venv_python()
            result = subprocess.run(
                [venv_python, "-c", "import sys; print([p for p in sys.path if 'site-packages' in p][0])"],
                capture_output=True,
                text=True,
                check=True
            )
            site_packages = Path(result.stdout.strip())
            
            # Check which libraries have OpenMP
            torch_omp = site_packages / "torch" / "lib" / "libomp.dylib"
            sklearn_omp = site_packages / "sklearn" / ".dylibs" / "libomp.dylib"
            faiss_omp = site_packages / "faiss" / ".dylibs" / "libomp.dylib"
            
            libs_found = []
            if torch_omp.exists():
                libs_found.append("PyTorch")
            if sklearn_omp.exists():
                libs_found.append("scikit-learn")
            if faiss_omp.exists():
                libs_found.append("FAISS")
            
            if len(libs_found) == 0:
                print_warning("No OpenMP libraries found (unusual)")
                return True
            
            if len(libs_found) == 1:
                print_success(f"Single OpenMP library: {libs_found[0]}")
                return True
            
            print_info(f"Multiple OpenMP libraries detected: {', '.join(libs_found)}")
            print_info("Using environment variables to handle conflicts:")
            print_info("  • KMP_DUPLICATE_LIB_OK=TRUE (allows multiple OpenMP)")
            print_info("  • OMP_NUM_THREADS=4 (limits threads)")
            print("")
            print_success("✓ Environment-based approach (safe, no file modification)")
            print_info("Note: Environment variables set at top of Python modules")
            
            return True
            
        except Exception as e:
            print_warning(f"Could not check OpenMP libraries: {e}")
            print_info("Continuing anyway - environment variables should handle conflicts")
            return True
    
    def check_ollama(self):
        """Check if Ollama is installed and running"""
        print_header("🤖 Checking Ollama LLM")
        
        # Check if Ollama is installed
        ollama_cmd = shutil.which("ollama")
        if not ollama_cmd:
            print_warning("Ollama not found!")
            print("")
            print_info("Ollama Installation Instructions:")
            print("")
            
            if self.is_colab:
                print("  Google Colab:")
                print("  Run in a notebook cell:")
                print("    !curl -fsSL https://ollama.com/install.sh | sh")
                print("    !nohup ollama serve > /dev/null 2>&1 &")
                print("    !sleep 5 && ollama pull llama3.2")
            elif platform.system() == "Darwin":  # macOS
                print("  macOS:   curl -fsSL https://ollama.com/install.sh | sh")
                print("  Or:      brew install ollama")
            elif platform.system() == "Linux":
                print("  Linux:   curl -fsSL https://ollama.com/install.sh | sh")
            elif platform.system() == "Windows":
                print("  Windows: Download from https://ollama.com/download")
            
            print("")
            print_info("After installing, run: ollama pull llama3")
            return False
        
        print_success("Ollama is installed")
        
        # Check if Ollama service is running
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print_warning("Ollama service not responding")
                self.start_ollama_service()
            else:
                print_success("Ollama service is running")
                
        except subprocess.TimeoutExpired:
            print_warning("Ollama service not responding")
            self.start_ollama_service()
        except Exception as e:
            print_warning(f"Could not check Ollama status: {e}")
        
        return True
    
    def start_ollama_service(self):
        """Start Ollama service in background"""
        print_info("Starting Ollama service...")
        
        try:
            if self.is_windows:
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            time.sleep(3)  # Wait for service to start
            print_success("Ollama service started")
            
        except Exception as e:
            print_warning(f"Could not start Ollama service: {e}")
            print_info("Please start manually: ollama serve")
    
    def check_llama_model(self):
        """Check if llama3 model is available and auto-detect version"""
        print_header("🦙 Checking Llama 3 Model")
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Auto-detect any llama3 variant (llama3, llama3.2, llama3.1, etc.)
            if "llama3" in result.stdout.lower():
                # Extract the actual model name
                for line in result.stdout.split('\n'):
                    if 'llama3' in line.lower():
                        model_name = line.split()[0].split(':')[0]  # Get model name without :latest
                        print_success(f"Llama 3 model is available: {model_name}")
                        
                        # Update config to use detected model
                        self._update_config_model(model_name)
                        return True
            
            print_warning("Llama 3 model not found")
            return self.pull_llama_model()
                
        except Exception as e:
            print_warning(f"Could not check models: {e}")
            return False
    
    def _update_config_model(self, model_name: str):
        """Update config and .env with detected model name"""
        env_file = self.script_dir / ".env"
        
        if env_file.exists():
            # Update .env file
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            with open(env_file, 'w') as f:
                for line in lines:
                    if line.startswith('LLM_MODEL='):
                        f.write(f'LLM_MODEL={model_name}\n')
                    else:
                        f.write(line)
        
        # Update environment variable for current session
        os.environ['LLM_MODEL'] = model_name
        print_info(f"Updated config to use: {model_name}")
    
    def pull_llama_model(self):
        """Pull llama3 model - tries llama3.2 first (newer/smaller), falls back to llama3"""
        print_info("Pulling Llama 3 model (this may take several minutes)...")
        
        # Try llama3.2 first (2GB, newer)
        models_to_try = ["llama3.2", "llama3"]
        
        for model in models_to_try:
            try:
                size = "~2GB" if model == "llama3.2" else "~4.7GB"
                print_info(f"Trying {model} (size: {size})...")
                
                process = subprocess.Popen(
                    ["ollama", "pull", model],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                for line in process.stdout:
                    print(f"  {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    print_success(f"{model} downloaded successfully")
                    self._update_config_model(model)
                    return True
                else:
                    print_warning(f"Failed to pull {model}, trying next...")
                    
            except Exception as e:
                print_warning(f"Error downloading {model}: {e}")
                continue
        
        print_error("Failed to download any Llama 3 variant")
        return False
    
    def create_env_file(self):
        """Create .env file if it doesn't exist"""
        env_file = self.script_dir / ".env"
        env_example = self.script_dir / ".env.example"
        
        if env_file.exists():
            return True
        
        if env_example.exists():
            print_info("Creating .env file from template...")
            shutil.copy(env_example, env_file)
            print_success(".env file created")
        
        return True
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}🏥  Multimodal Medical Assistant - Setup & Launcher{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print("")
        
        steps = [
            ("Python Version", self.check_python_version),
            ("Virtual Environment", self.setup_virtual_environment),
            ("Dependencies", self.install_dependencies),
            ("Environment Config", self.create_env_file),
            ("Ollama Installation", self.check_ollama),
            ("Llama 3 Model", self.check_llama_model),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print("")
                print_error(f"Setup failed at: {step_name}")
                print_info("Please resolve the issue and run again")
                return False
        
        print("")
        print_success("🎉 Setup completed successfully!")
        return True
    
    def launch_ui(self):
        """Launch Gradio UI"""
        print_header("🚀 Launching Web UI")
        
        # Quick verification before launch
        print_info("Verifying Ollama service and model...")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print_error("Ollama service not responding")
                print_info("Start Ollama: ollama serve")
                return False
            
            if "llama3" not in result.stdout.lower():
                print_error("Llama 3 model not found")
                print_info("Pull model: ollama pull llama3.2")
                print_info("Or: ollama pull llama3")
                return False
            
            print_success("✓ Ollama and model ready")
            
        except subprocess.TimeoutExpired:
            print_error("Ollama service not responding")
            print_info("Start Ollama: ollama serve")
            return False
        except FileNotFoundError:
            print_error("Ollama not installed")
            return False
        except Exception as e:
            print_warning(f"Could not verify Ollama: {e}")
            print_info("Proceeding anyway...")
        
        ui_script = self.ui_dir / "medical_assistant_ui.py"
        if not ui_script.exists():
            print_error(f"UI script not found: {ui_script}")
            return False
        
        # Show progress animation
        import threading
        
        stop_animation = threading.Event()
        
        def progress_animation():
            """Animated progress indicator"""
            steps = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            idx = 0
            messages = [
                "Initializing web interface",
                "Loading medical AI models (BioBERT, MedCLIP, Llama3)",
                "Preparing Gradio dashboard",
                "Starting web server"
            ]
            msg_idx = 0
            counter = 0
            
            while not stop_animation.is_set():
                sys.stdout.write(f"\r{Colors.CYAN}{steps[idx]} {messages[msg_idx]}...{Colors.END}")
                sys.stdout.flush()
                idx = (idx + 1) % len(steps)
                counter += 1
                
                # Cycle through messages every 10 iterations
                if counter % 10 == 0:
                    msg_idx = (msg_idx + 1) % len(messages)
                
                time.sleep(0.1)
            
            # Clear the progress line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        
        # Start progress animation in background
        animation_thread = threading.Thread(target=progress_animation, daemon=True)
        animation_thread.start()
        
        try:
            venv_python = self.get_venv_python()
            
            # Give animation time to show initial message
            time.sleep(0.5)
            
            # Stop animation
            stop_animation.set()
            animation_thread.join(timeout=0.5)
            
            print_success("✓ Environment ready - launching UI")
            print("")
            print_info("URL: http://localhost:7860")
            print_info("Press Ctrl+C to stop")
            print("")
            
            # Run from Codebase directory to ensure model caching consistency
            subprocess.run(
                [venv_python, str(ui_script)],
                cwd=str(self.script_dir)
            )
        except KeyboardInterrupt:
            print("")
            print_info("UI stopped by user")
        except Exception as e:
            print_error(f"Error launching UI: {e}")
            return False
        
        return True
    
    def launch_cli(self):
        """Launch CLI interface"""
        print_header("🚀 Launching CLI Interface")
        
        # Quick verification before launch
        print_info("Verifying Ollama service and model...")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print_error("Ollama service not responding")
                print_info("Start Ollama: ollama serve")
                return False
            
            if "llama3" not in result.stdout.lower():
                print_error("Llama 3 model not found")
                print_info("Pull model: ollama pull llama3.2")
                print_info("Or: ollama pull llama3")
                return False
            
            print_success("✓ Ollama and model ready")
            
        except subprocess.TimeoutExpired:
            print_error("Ollama service not responding")
            print_info("Start Ollama: ollama serve")
            return False
        except FileNotFoundError:
            print_error("Ollama not installed")
            return False
        except Exception as e:
            print_warning(f"Could not verify Ollama: {e}")
            print_info("Proceeding anyway...")
        
        cli_script = self.script_dir / "cli_main.py"
        if not cli_script.exists():
            print_error(f"CLI script not found: {cli_script}")
            return False
        
        # Show progress animation
        import threading
        import time
        
        stop_animation = threading.Event()
        
        def progress_animation():
            """Animated progress indicator"""
            steps = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            idx = 0
            messages = [
                "Initializing CLI environment",
                "Loading medical AI models (BioBERT, MedCLIP)",
                "Preparing FAISS knowledge base",
                "Starting interactive session"
            ]
            msg_idx = 0
            counter = 0
            
            while not stop_animation.is_set():
                sys.stdout.write(f"\r{Colors.CYAN}{steps[idx]} {messages[msg_idx]}...{Colors.END}")
                sys.stdout.flush()
                idx = (idx + 1) % len(steps)
                counter += 1
                
                # Cycle through messages every 10 iterations
                if counter % 10 == 0:
                    msg_idx = (msg_idx + 1) % len(messages)
                
                time.sleep(0.1)
            
            # Clear the progress line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        
        # Start progress animation in background
        animation_thread = threading.Thread(target=progress_animation, daemon=True)
        animation_thread.start()
        
        try:
            venv_python = self.get_venv_python()
            
            # Give animation time to show initial message
            time.sleep(0.5)
            
            # Stop animation and launch CLI
            stop_animation.set()
            animation_thread.join(timeout=0.5)
            
            print_success("✓ Environment ready - launching CLI")
            print("")
            
            subprocess.run(
                [venv_python, str(cli_script)],
                cwd=str(self.script_dir)
            )
        except KeyboardInterrupt:
            stop_animation.set()
            print("")
            print_info("CLI stopped by user")
        except Exception as e:
            stop_animation.set()
            print_error(f"Error launching CLI: {e}")
            return False
        
        return True
    
    def run_tests(self):
        """Run test suite"""
        print_header("🧪 Running Tests")
        
        tests_dir = self.script_dir / "tests"
        if not tests_dir.exists():
            print_warning("Tests directory not found")
            return False
        
        try:
            venv_python = self.get_venv_python()
            result = subprocess.run(
                [venv_python, "-m", "pytest", str(tests_dir), "-v"],
                cwd=str(self.script_dir)
            )
            return result.returncode == 0
        except Exception as e:
            print_error(f"Error running tests: {e}")
            return False
    
    def download_models(self):
        """Check model availability - BiomedCLIP downloads automatically from HuggingFace"""
        print_header("📥 Model Check")
        
        print_info("This application uses Microsoft BiomedCLIP")
        print_info("Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        print("")
        print_info("BiomedCLIP automatically downloads from HuggingFace on first use")
        print_info("Cached in: ~/.cache/huggingface/hub/")
        print_info("Download size: ~500MB (one-time download)")
        print("")
        
        # Check if model is already cached
        import os
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        biomedclip_cached = any("BiomedCLIP" in str(p) for p in hf_cache.glob("*") if hf_cache.exists())
        
        if biomedclip_cached:
            print_success("✓ BiomedCLIP already cached")
        else:
            print_info("BiomedCLIP will download on first application launch")
        
        print("")
        print_success("No manual download required!")
        return True
    
    def show_menu(self):
        """Show interactive menu"""
        print("")
        print(f"{Colors.BOLD}Select an option:{Colors.END}")
        print("")
        print("  1. Launch Web UI (Gradio Dashboard)")
        print("  2. Launch CLI Interface")
        print("  3. Run Tests")
        print("  4. Re-run Setup")
        print("  5. Exit")
        print("")
        
        try:
            choice = input(f"{Colors.CYAN}Enter choice (1-5): {Colors.END}").strip()
            return choice
        except KeyboardInterrupt:
            print("")
            return "5"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multimodal Medical Assistant - Setup & Launcher"
    )
    parser.add_argument("--ui", action="store_true", help="Launch Web UI")
    parser.add_argument("--cli", action="store_true", help="Launch CLI interface")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--setup", action="store_true", help="Run setup only")
    parser.add_argument("--download-models", action="store_true", help="Download pretrained model weights")
    
    args = parser.parse_args()
    
    setup = MedicalAssistantSetup()
    
    # Run setup first
    if not setup.run_full_setup():
        sys.exit(1)
    
    # If specific mode requested, launch it
    if args.download_models:
        sys.exit(0 if setup.download_models() else 1)
    elif args.ui:
        setup.launch_ui()
    elif args.cli:
        setup.launch_cli()
    elif args.test:
        sys.exit(0 if setup.run_tests() else 1)
    elif args.setup:
        print("")
        print_success("Setup complete. Use --ui or --cli to launch")
    else:
        # No arguments - prompt user for UI or CLI
        print("")
        print(f"{Colors.BOLD}How would you like to start?{Colors.END}")
        print("")
        print("  1. 🌐 Web UI (Gradio Dashboard)")
        print("  2. 💻 CLI Interface")
        print("")
        
        try:
            choice = input(f"{Colors.CYAN}Enter choice (1 or 2): {Colors.END}").strip()
            
            if choice == "1":
                setup.launch_ui()
            elif choice == "2":
                setup.launch_cli()
            else:
                print_warning("Invalid choice. Use: python main.py --ui  OR  python main.py --cli")
        except KeyboardInterrupt:
            print("")
            print_info("Goodbye!")

if __name__ == "__main__":
    main()
