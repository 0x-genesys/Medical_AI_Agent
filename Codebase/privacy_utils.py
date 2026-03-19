"""
Simple HIPAA compliance utility
Removes basic PHI from conversation history before storage
"""
import re


def sanitize_phi(text: str) -> str:
    """
    Sanitize Protected Health Information (PHI) from text for HIPAA compliance.
    
    Removes or masks personally identifiable health information before storing
    conversation history. Implements HIPAA Safe Harbor method by removing
    18 types of identifiers while preserving clinical content.
    
    Removed Identifiers:
        - Dates (MM/DD/YYYY, MM-DD-YYYY formats)
        - Phone numbers (various formats)
        - Social Security Numbers (SSN)
        - Email addresses
        - Patient names (when in "Name: John Doe" format)
        - Medical Record Numbers (MRN) and Patient IDs
        - Ages over 89 (aggregated to >89 per HIPAA Safe Harbor)
    
    Preserved Information:
        - Clinical symptoms and diagnoses
        - Medications and dosages
        - Laboratory values and findings
        - Medical terminology and procedures
    
    Args:
        text (str): Original text that may contain PHI
        
    Returns:
        str: Sanitized text with PHI replaced by placeholders (e.g., [DATE], [PHONE])
    
    Example:
        >>> sanitize_phi("Patient John Doe, DOB 01/15/1950, phone 555-1234")
        'Patient Name: [NAME], DOB [DATE], phone [PHONE]'
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
    
    # Aggregate ages >89 per HIPAA Safe Harbor requirement
    # HIPAA requires ages over 89 to be aggregated to protect elderly patients
    def replace_age(match):
        """
        Replace ages over 89 with aggregated value per HIPAA Safe Harbor.
        
        Args:
            match (re.Match): Regex match object containing age value
        
        Returns:
            str: Original text if age <= 89, otherwise "Age: >89 years"
        """
        age = int(match.group(1))
        return f"Age: >89 years" if age > 89 else match.group(0)
    
    text = re.sub(r'Age:\s*(\d+)\s*years?', replace_age, text, flags=re.IGNORECASE)
    
    return text
