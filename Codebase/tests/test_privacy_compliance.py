"""
Test cases for privacy compliance and de-identification
Ensures HIPAA compliance and PHI removal before storage
"""
import pytest
from privacy_utils import DeIdentifier, SessionManager


class TestDeIdentification:
    """Test suite for de-identification utilities"""
    
    @pytest.fixture
    def deidentifier(self):
        """Initialize de-identifier"""
        return DeIdentifier()
    
    def test_date_anonymization(self, deidentifier):
        """Test that dates are properly anonymized"""
        text = "Patient seen on 03/15/2024 for follow-up"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[DATE]" in anonymized
        assert "03/15/2024" not in anonymized
        assert "dates" in metadata['phi_removed']
    
    def test_phone_anonymization(self, deidentifier):
        """Test that phone numbers are removed"""
        text = "Contact: 555-123-4567 or 555.987.6543"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[PHONE]" in anonymized
        assert "555-123-4567" not in anonymized
        assert "phone" in metadata['phi_removed']
    
    def test_ssn_anonymization(self, deidentifier):
        """Test that SSN is removed"""
        text = "SSN: 123-45-6789"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[SSN]" in anonymized
        assert "123-45-6789" not in anonymized
        assert "ssn" in metadata['phi_removed']
    
    def test_mrn_anonymization(self, deidentifier):
        """Test that medical record numbers are anonymized"""
        text = "Patient ID: MRN-12345-ABC"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[ANONYMIZED]" in anonymized
        assert "MRN-12345-ABC" not in anonymized
        assert "mrn" in metadata['phi_removed']
    
    def test_email_anonymization(self, deidentifier):
        """Test that email addresses are removed"""
        text = "Contact: patient@email.com"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[EMAIL]" in anonymized
        assert "patient@email.com" not in anonymized
        assert "email" in metadata['phi_removed']
    
    def test_address_anonymization(self, deidentifier):
        """Test that street addresses are removed"""
        text = "Lives at 123 Main Street, Apartment 4B"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "[ADDRESS]" in anonymized or "123 Main Street" not in anonymized
        assert "address" in metadata['phi_removed']
    
    def test_zip_code_anonymization(self, deidentifier):
        """Test that ZIP codes are partially anonymized (Safe Harbor)"""
        text = "ZIP: 12345-6789"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        # Should keep first 3 digits per Safe Harbor
        assert "123" in anonymized
        assert "12345" not in anonymized  # Full ZIP removed
        assert "zip" in metadata['phi_removed']
    
    def test_age_over_89_redaction(self, deidentifier):
        """Test that ages over 89 are aggregated per HIPAA Safe Harbor"""
        text = "Age: 92 years"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert ">89" in anonymized or "[REDACTED]" in anonymized
        assert "92" not in anonymized
        assert "age_over_89" in metadata['phi_removed']
    
    def test_age_under_90_preserved(self, deidentifier):
        """Test that ages under 90 are preserved"""
        text = "Age: 65 years"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        assert "65" in anonymized  # Should be preserved
        assert "age_over_89" not in metadata.get('phi_removed', [])
    
    def test_clinical_content_preserved(self, deidentifier):
        """Test that clinical content is preserved during anonymization"""
        text = "Patient has polyuria, polydipsia, and elevated glucose. HbA1c: 8.9%"
        anonymized, metadata = deidentifier.anonymize_text(text)
        
        # Clinical terms should remain
        assert "polyuria" in anonymized
        assert "polydipsia" in anonymized
        assert "glucose" in anonymized
        assert "HbA1c: 8.9%" in anonymized
    
    def test_validate_anonymization_clean_text(self, deidentifier):
        """Test validation of properly anonymized text"""
        text = "Patient has chest pain and dyspnea. Troponin elevated."
        
        # Should pass validation (no PHI)
        assert deidentifier.validate_anonymization(text) == True
    
    def test_validate_anonymization_with_phi(self, deidentifier):
        """Test validation catches remaining PHI"""
        text = "Patient SSN: 123-45-6789 has chest pain"
        
        # Should fail validation (PHI present)
        assert deidentifier.validate_anonymization(text) == False
    
    def test_extract_clinical_features(self, deidentifier):
        """Test extraction of clinical features without PHI"""
        text = "Patient has fever, cough, and chest pain. Taking aspirin and metformin."
        
        features = deidentifier.extract_clinical_features(text)
        
        assert "fever" in features['symptoms_mentioned']
        assert "cough" in features['symptoms_mentioned']
        assert "chest pain" in features['symptoms_mentioned']
        assert "aspirin" in features['medications_mentioned']
        assert "metformin" in features['medications_mentioned']
        assert 'content_hash' in features


class TestSessionManagement:
    """Test suite for privacy-compliant session management"""
    
    @pytest.fixture
    def session_manager(self):
        """Initialize session manager"""
        return SessionManager()
    
    def test_create_session(self, session_manager):
        """Test session creation"""
        session = session_manager.create_session("test_session_001")
        
        assert session['session_id'] == "test_session_001"
        assert 'created_at' in session
        assert session['interactions'] == []
    
    def test_add_interaction_with_anonymization(self, session_manager):
        """Test adding interaction with automatic anonymization"""
        session_id = "test_session_002"
        query = "My phone is 555-123-4567 and I have chest pain"
        response = "You should seek immediate medical attention"
        
        session_manager.add_interaction(session_id, query, response, anonymize=True)
        
        session = session_manager.sessions[session_id]
        interaction = session['interactions'][0]
        
        # PHI should be removed
        assert "555-123-4567" not in interaction['query']
        assert "[PHONE]" in interaction['query']
        
        # Clinical content should remain
        assert "chest pain" in interaction['query']
    
    def test_add_interaction_blocks_phi(self, session_manager):
        """Test that interaction with PHI is blocked if anonymization fails"""
        session_id = "test_session_003"
        
        # Create text with SSN that should be caught
        query = "Patient SSN 123-45-6789"
        response = "Processed"
        
        # Should raise error if PHI detected after anonymization
        try:
            session_manager.add_interaction(session_id, query, response, anonymize=True)
            # If no error, check that SSN was removed
            interaction = session_manager.sessions[session_id]['interactions'][0]
            assert "123-45-6789" not in interaction['query']
        except ValueError:
            # Expected - PHI blocked
            pass
    
    def test_get_session_context(self, session_manager):
        """Test retrieving session context for continuity"""
        session_id = "test_session_004"
        
        # Add multiple interactions
        interactions = [
            ("I have diabetes", "Diabetes requires management"),
            ("What medications?", "Metformin is first-line"),
            ("Any side effects?", "GI upset, lactic acidosis rare")
        ]
        
        for query, response in interactions:
            session_manager.add_interaction(session_id, query, response, anonymize=False)
        
        # Get context
        context = session_manager.get_session_context(session_id, last_n=2)
        
        # Should include recent queries
        assert "Previous Query" in context
        assert "medications" in context.lower() or "side effects" in context.lower()
    
    def test_clinical_features_in_history(self, session_manager):
        """Test that clinical features are extracted and stored"""
        session_id = "test_session_005"
        query = "Patient has fever, cough, and is taking aspirin"
        response = "These are common cold symptoms"
        
        session_manager.add_interaction(session_id, query, response, anonymize=True)
        
        interaction = session_manager.sessions[session_id]['interactions'][0]
        features = interaction['clinical_features']
        
        assert 'fever' in features['symptoms_mentioned']
        assert 'cough' in features['symptoms_mentioned']
        assert 'aspirin' in features['medications_mentioned']


class TestPrivacyCompliance:
    """Test suite for overall privacy compliance"""
    
    def test_no_phi_in_stored_data(self):
        """Test that no PHI exists in stored session data"""
        session_mgr = SessionManager()
        session_id = "compliance_test_001"
        
        # Create interaction with multiple PHI types
        query_with_phi = """
        Patient Name: John Doe
        DOB: 01/15/1980
        Phone: 555-123-4567
        SSN: 123-45-6789
        Has chest pain and elevated troponin
        """
        response = "STEMI protocol activated"
        
        session_mgr.add_interaction(session_id, query_with_phi, response, anonymize=True)
        
        # Retrieve stored interaction
        interaction = session_mgr.sessions[session_id]['interactions'][0]
        stored_query = interaction['query']
        
        # Verify no PHI in storage
        assert "John Doe" not in stored_query or "[NAME]" in stored_query
        assert "01/15/1980" not in stored_query
        assert "555-123-4567" not in stored_query
        assert "123-45-6789" not in stored_query
        
        # Clinical content should be preserved
        assert "chest pain" in stored_query
        assert "troponin" in stored_query
    
    def test_hash_for_deduplication(self):
        """Test that hashing allows deduplication without storing PHI"""
        deidentifier = DeIdentifier()
        
        text1 = "Patient John Doe has diabetes"
        text2 = "Patient John Doe has diabetes"
        text3 = "Patient Jane Smith has diabetes"
        
        hash1 = deidentifier._hash_text(text1)
        hash2 = deidentifier._hash_text(text2)
        hash3 = deidentifier._hash_text(text3)
        
        # Same text should produce same hash
        assert hash1 == hash2
        
        # Different text should produce different hash
        assert hash1 != hash3
        
        # Hash should be irreversible (SHA-256, 64 characters)
        assert len(hash1) == 64
        assert "John Doe" not in hash1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
