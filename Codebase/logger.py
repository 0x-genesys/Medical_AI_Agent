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
    """
    Custom logger with medical compliance and audit trail features.
    
    Provides structured logging for the medical assistant with support for
    both file and console logging. Includes special audit logging methods
    for HIPAA compliance tracking.
    
    Attributes:
        logger (logging.Logger): Underlying Python logger instance
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize a medical logger with file and console handlers.
        
        Creates a logger with formatted output to both file (if log_dir provided)
        and console. Prevents duplicate handlers if logger already configured.
        
        Args:
            name (str): Name for this logger (typically module name)
            log_dir (Optional[Path]): Directory for log files (None = console only)
        
        Returns:
            None
        """
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
        """
        Log an audit trail entry for compliance tracking.
        
        Creates special audit log entries prefixed with 'AUDIT:' for
        tracking medical data access and operations. Used for HIPAA
        compliance and security monitoring.
        
        Args:
            action (str): Description of the action being audited
            details (dict): Dictionary of relevant details for the action
        
        Returns:
            None
        """
        audit_message = f"AUDIT: {action} | Details: {details}"
        self.logger.info(audit_message)
    
    def info(self, message: str):
        """
        Log an informational message.
        
        Args:
            message (str): The information message to log
        
        Returns:
            None
        """
        self.logger.info(message)
    
    def error(self, message: str, exc_info: bool = False):
        """
        Log an error message with optional exception traceback.
        
        Args:
            message (str): The error message to log
            exc_info (bool): If True, include full exception traceback
        
        Returns:
            None
        """
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str):
        """
        Log a warning message.
        
        Args:
            message (str): The warning message to log
        
        Returns:
            None
        """
        self.logger.warning(message)
    
    def debug(self, message: str):
        """
        Log a debug message.
        
        Args:
            message (str): The debug message to log
        
        Returns:
            None
        """
        self.logger.debug(message)


def get_logger(name: str) -> MedicalLogger:
    """
    Factory function to create or retrieve a MedicalLogger instance.
    
    Creates a new MedicalLogger with the specified name and default
    log directory (logs/). This is the preferred way to get loggers
    throughout the application.
    
    Args:
        name (str): Name for the logger (typically __name__ of calling module)
    
    Returns:
        MedicalLogger: Configured logger instance
    """
    return MedicalLogger(name, log_dir=Path("logs"))
