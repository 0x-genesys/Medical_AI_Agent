"""
Security and Compliance Module
Handles data de-identification and HIPAA compliance for CLI
"""
import re

from logger import get_logger
from config import config


class SecurityManager:
    """
    Manages security and compliance requirements for CLI usage
    Implements PHI de-identification
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("SecurityManager initialized for CLI")
        
        self.phi_patterns = {
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'mrn': r'\bMRN[:\s]*\d+\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        }
    
    def deidentify_text(self, text: str) -> str:
        """
        Remove or mask Protected Health Information (PHI)
        
        Args:
            text: Input text containing potential PHI
            
        Returns:
            De-identified text
        """
        if not config.security.enable_deidentification:
            return text
        
        self.logger.info("De-identifying text")
        
        deidentified = text
        
        deidentified = re.sub(self.phi_patterns['email'], '[EMAIL]', deidentified)
        deidentified = re.sub(self.phi_patterns['phone'], '[PHONE]', deidentified)
        deidentified = re.sub(self.phi_patterns['ssn'], '[SSN]', deidentified)
        deidentified = re.sub(self.phi_patterns['mrn'], '[MRN]', deidentified)
        
        return deidentified
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            text: User input text
            
        Returns:
            Sanitized text
        """
        sanitized = text.strip()
        
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
