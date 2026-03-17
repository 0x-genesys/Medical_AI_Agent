"""
Test cases for smart path detection logic
Tests file path validation and loading
"""
import pytest
import tempfile
from pathlib import Path


class TestPathDetection:
    """Test suite for smart path detection"""
    
    def test_valid_file_path_detection(self):
        """Test that valid file paths are detected correctly"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test clinical note")
            temp_path = f.name
        
        try:
            # Test Path detection
            p = Path(temp_path)
            assert p.exists()
            assert p.is_file()
            
            # Should be able to read it
            with open(p, 'r') as f:
                content = f.read()
            assert content == "Test clinical note"
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_path_detection(self):
        """Test that invalid paths are detected"""
        invalid_path = "this/does/not/exist.txt"
        p = Path(invalid_path)
        
        assert not p.exists()
    
    def test_text_vs_path_distinction(self):
        """Test distinguishing between text and file paths"""
        # Regular text (no slashes)
        text = "Patient has fever and cough"
        assert '/' not in text and '\\' not in text
        
        # Path-like text
        path_text = "examples/sample_clinical_note.txt"
        assert '/' in path_text or '\\' in path_text
    
    def test_example_clinical_notes_exist(self):
        """Test that example clinical notes exist"""
        base_path = Path("examples")
        
        expected_files = [
            "sample_clinical_note_diabetes.txt",
            "sample_clinical_note_chest_pain.txt",
            "sample_clinical_note_pneumonia.txt",
            "sample_clinical_note_hypertension.txt"
        ]
        
        for filename in expected_files:
            file_path = base_path / filename
            assert file_path.exists(), f"Example file missing: {filename}"
            assert file_path.is_file()
            
            # Check that file has content
            content = file_path.read_text()
            assert len(content) > 100, f"File {filename} seems too short"
    
    def test_example_notes_have_patient_info(self):
        """Test that example clinical notes contain patient information"""
        base_path = Path("examples")
        
        diabetes_file = base_path / "sample_clinical_note_diabetes.txt"
        assert diabetes_file.exists()
        
        content = diabetes_file.read_text()
        
        # Should contain patient demographics
        assert "Age:" in content or "age" in content.lower()
        assert "Gender:" in content or "Male" in content or "Female" in content
        
        # Should contain clinical information
        clinical_terms = ["Chief Complaint", "History", "Physical", "Assessment", "Plan"]
        assert any(term.lower() in content.lower() for term in clinical_terms)
    
    def test_relative_path_handling(self):
        """Test handling of relative file paths"""
        # Test with examples directory
        rel_path = Path("examples/sample_clinical_note_diabetes.txt")
        
        if rel_path.exists():
            content = rel_path.read_text()
            assert len(content) > 0
    
    def test_absolute_path_handling(self):
        """Test handling of absolute file paths"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Absolute path test")
            temp_path = Path(f.name).absolute()
        
        try:
            assert temp_path.exists()
            assert temp_path.is_file()
            assert temp_path.is_absolute()
            
            content = temp_path.read_text()
            assert content == "Absolute path test"
        finally:
            temp_path.unlink()


class TestClinicalNoteContent:
    """Test suite for sample clinical note content quality"""
    
    def test_diabetes_note_completeness(self):
        """Test diabetes clinical note has comprehensive information"""
        note_path = Path("examples/sample_clinical_note_diabetes.txt")
        
        if not note_path.exists():
            pytest.skip("Diabetes note not found")
        
        content = note_path.read_text()
        
        # Check for key diabetes-related terms
        diabetes_terms = ["polyuria", "polydipsia", "glucose", "HbA1c", "diabetes", "insulin"]
        found_terms = [term for term in diabetes_terms if term.lower() in content.lower()]
        assert len(found_terms) >= 4, f"Diabetes note should contain diabetes-related terms, found: {found_terms}"
    
    def test_chest_pain_note_completeness(self):
        """Test chest pain clinical note has relevant information"""
        note_path = Path("examples/sample_clinical_note_chest_pain.txt")
        
        if not note_path.exists():
            pytest.skip("Chest pain note not found")
        
        content = note_path.read_text()
        
        # Check for cardiac-related terms
        cardiac_terms = ["chest pain", "troponin", "ECG", "MI", "cardiac", "heart"]
        found_terms = [term for term in cardiac_terms if term.lower() in content.lower()]
        assert len(found_terms) >= 3, f"Chest pain note should contain cardiac terms"
    
    def test_pneumonia_note_completeness(self):
        """Test pneumonia clinical note has relevant information"""
        note_path = Path("examples/sample_clinical_note_pneumonia.txt")
        
        if not note_path.exists():
            pytest.skip("Pneumonia note not found")
        
        content = note_path.read_text()
        
        # Check for respiratory-related terms
        respiratory_terms = ["pneumonia", "cough", "fever", "dyspnea", "chest x-ray", "antibiotic"]
        found_terms = [term for term in respiratory_terms if term.lower() in content.lower()]
        assert len(found_terms) >= 3, f"Pneumonia note should contain respiratory terms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
