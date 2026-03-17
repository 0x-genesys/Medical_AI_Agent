"""
Simple HIPAA compliance utility
Removes basic PHI from conversation history before storage
"""
import re


def sanitize_phi(text: str) -> str:
    """
    Simple PHI sanitization for conversation history
    Removes: dates, phone numbers, SSN, emails
    Preserves: clinical content (symptoms, diagnoses, medications)
    
    Args:
        text: Original text that may contain PHI
        
    Returns:
        Sanitized text with PHI removed
    """
    # Remove dates (MM/DD/YYYY, MM-DD-YYYY)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Remove SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove patient names (format: "Patient Name: John Doe")
    text = re.sub(r'(?:Patient Name|Name):\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', 
                  'Patient Name: [NAME]', text, flags=re.IGNORECASE)
    
    # Remove MRN/Patient ID
    text = re.sub(r'\b(?:MRN|Patient ID)[:\s]*[A-Z0-9-]+\b', 
                  'Patient ID: [REDACTED]', text, flags=re.IGNORECASE)
    
    # Aggregate ages >89 per HIPAA Safe Harbor
    def replace_age(match):
        age = int(match.group(1))
        return f"Age: >89 years" if age > 89 else match.group(0)
    
    text = re.sub(r'Age:\s*(\d+)\s*years?', replace_age, text, flags=re.IGNORECASE)
    
    return text
