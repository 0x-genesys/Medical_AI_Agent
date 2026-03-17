"""
Configuration module for CLI-based Medical Assistant
Centralized configuration management for healthcare CLI application
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    llm_model: str = "llama3"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 120


@dataclass
class StorageConfig:
    """Configuration for data storage"""
    temp_dir: Path = Path("temp")
    output_dir: Path = Path("output")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """Security and compliance configuration for CLI"""
    enable_deidentification: bool = True
    hipaa_compliant: bool = True
    max_retries: int = 3


class Config:
    """Main configuration class for CLI healthcare application"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.storage = StorageConfig()
        self.security = SecurityConfig()
        
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        if os.getenv("LLM_MODEL"):
            self.model.llm_model = os.getenv("LLM_MODEL")


# Global configuration instance
config = Config()
