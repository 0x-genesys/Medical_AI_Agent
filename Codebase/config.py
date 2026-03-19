"""
Configuration module for CLI-based Medical Assistant
Centralized configuration management for healthcare CLI application
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration for AI models used in the medical assistant.
    
    This dataclass stores all model-related configuration parameters including
    the LLM model name, temperature for response generation, maximum token limits,
    and timeout settings for API calls.
    
    Attributes:
        llm_model (str): Name of the Ollama LLM model to use (default: llama3)
        temperature (float): Controls randomness in generation (0.0-1.0, lower is more deterministic)
        max_tokens (int): Maximum number of tokens in model response
        timeout (int): Maximum time in seconds to wait for model response
    """
    llm_model: str = "llama3"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 120


@dataclass
class StorageConfig:
    """
    Configuration for data storage paths and file management.
    
    Manages temporary and output directory paths for the medical assistant.
    Automatically creates directories on initialization if they don't exist.
    
    Attributes:
        temp_dir (Path): Directory for temporary files (default: temp/)
        output_dir (Path): Directory for generated outputs (default: output/)
    """
    temp_dir: Path = Path("temp")
    output_dir: Path = Path("output")
    
    def __post_init__(self):
        """
        Post-initialization hook to create storage directories.
        
        Automatically called after dataclass initialization. Creates both
        temp_dir and output_dir if they don't already exist, including
        any necessary parent directories.
        
        Returns:
            None
        """
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """
    Security and compliance configuration for medical data handling.
    
    Manages HIPAA compliance settings and security features including
    PHI de-identification, audit logging, and retry policies.
    
    Attributes:
        enable_deidentification (bool): Enable PHI removal from stored data
        hipaa_compliant (bool): Enable HIPAA compliance mode
        max_retries (int): Maximum retry attempts for failed operations
    """
    enable_deidentification: bool = True
    hipaa_compliant: bool = True
    max_retries: int = 3


class Config:
    """
    Main configuration class for the medical assistant application.
    
    Aggregates all configuration sections (model, storage, security) into
    a single configuration object. Supports loading configuration from
    environment variables to override defaults.
    
    Attributes:
        model (ModelConfig): AI model configuration
        storage (StorageConfig): Storage paths configuration
        security (SecurityConfig): Security and compliance configuration
    """
    
    def __init__(self):
        """
        Initialize configuration with default values and load from environment.
        
        Creates instances of ModelConfig, StorageConfig, and SecurityConfig,
        then loads any overrides from environment variables.
        
        Returns:
            None
        """
        self.model = ModelConfig()
        self.storage = StorageConfig()
        self.security = SecurityConfig()
        
        self._load_from_env()
    
    def _load_from_env(self):
        """
        Load configuration overrides from environment variables.
        
        Checks for specific environment variables (e.g., LLM_MODEL) and
        updates the corresponding configuration values if found. This allows
        runtime configuration without modifying code.
        
        Environment Variables:
            LLM_MODEL: Override the default LLM model name
        
        Returns:
            None
        """
        if os.getenv("LLM_MODEL"):
            self.model.llm_model = os.getenv("LLM_MODEL")


# Global configuration instance
config = Config()
