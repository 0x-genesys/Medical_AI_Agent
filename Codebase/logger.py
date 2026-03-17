"""
Logging configuration for Multimodal Medical Assistant
Provides structured logging with audit trail capabilities
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class MedicalLogger:
    """Custom logger with medical compliance features"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if logger already configured
        if self.logger.handlers:
            return
        
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"medical_assistant_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def audit_log(self, action: str, details: dict):
        """Special audit logging for compliance"""
        audit_message = f"AUDIT: {action} | Details: {details}"
        self.logger.info(audit_message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


def get_logger(name: str) -> MedicalLogger:
    """Factory function to get logger instance"""
    return MedicalLogger(name, log_dir=Path("logs"))
