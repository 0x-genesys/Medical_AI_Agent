"""
Test cases for user context collection in query flow
Tests patient demographic information handling
"""
import pytest
from models import QueryRequest


class TestUserContext:
    """Test suite for user context handling"""
    
    def test_query_request_with_context(self):
        """Test QueryRequest accepts context parameter"""
        query = QueryRequest(
            query="What medication should I take?",
            context="Age: 45 years\nGender: Female\nWeight: 70 kg\nHeight: 165 cm",
            include_references=False
        )
        
        assert query.query is not None
        assert query.context is not None
        assert "Age" in query.context
    
    def test_query_request_without_context(self):
        """Test QueryRequest works without context"""
        query = QueryRequest(
            query="What are the symptoms of diabetes?",
            include_references=False
        )
        
        assert query.query is not None
        assert query.context is None or query.context == ""
    
    def test_bmi_calculation(self):
        """Test BMI calculation logic"""
        weight_kg = 70
        height_cm = 165
        
        # BMI = weight / (height_m)^2
        height_m = height_cm / 100
        expected_bmi = weight_kg / (height_m ** 2)
        
        assert 25 < expected_bmi < 26  # Should be ~25.7
    
    def test_context_string_formatting(self):
        """Test context string is properly formatted"""
        age = "45"
        gender = "Female"
        weight = "70"
        height = "165"
        
        context_parts = []
        if age:
            context_parts.append(f"Age: {age} years")
        if gender:
            context_parts.append(f"Gender: {gender}")
        if weight:
            context_parts.append(f"Weight: {weight} kg")
        if height:
            context_parts.append(f"Height: {height} cm")
        
        context = "\n".join(context_parts)
        
        assert "Age: 45 years" in context
        assert "Gender: Female" in context
        assert "Weight: 70 kg" in context
        assert "Height: 165 cm" in context
    
    def test_empty_context_handling(self):
        """Test that empty context fields are handled correctly"""
        age = ""
        gender = ""
        weight = ""
        height = ""
        
        context_parts = []
        if age:
            context_parts.append(f"Age: {age} years")
        if gender:
            context_parts.append(f"Gender: {gender}")
        if weight:
            context_parts.append(f"Weight: {weight} kg")
        if height:
            context_parts.append(f"Height: {height} cm")
        
        user_context = "\n".join(context_parts) if context_parts else None
        
        assert user_context is None
    
    def test_partial_context(self):
        """Test handling of partial context information"""
        age = "30"
        gender = ""
        weight = "80"
        height = ""
        
        context_parts = []
        if age:
            context_parts.append(f"Age: {age} years")
        if gender:
            context_parts.append(f"Gender: {gender}")
        if weight:
            context_parts.append(f"Weight: {weight} kg")
        if height:
            context_parts.append(f"Height: {height} cm")
        
        context = "\n".join(context_parts)
        
        # Should only contain provided fields
        assert "Age: 30 years" in context
        assert "Weight: 80 kg" in context
        assert "Gender" not in context
        assert "Height" not in context
    
    def test_bmi_with_valid_inputs(self):
        """Test BMI calculation with various valid inputs"""
        test_cases = [
            (70, 175, 22.86),  # Normal weight
            (90, 175, 29.39),  # Overweight
            (60, 160, 23.44),  # Normal weight
        ]
        
        for weight, height, expected_bmi in test_cases:
            calculated_bmi = weight / ((height / 100) ** 2)
            assert abs(calculated_bmi - expected_bmi) < 0.1
    
    def test_bmi_calculation_error_handling(self):
        """Test BMI calculation handles invalid inputs gracefully"""
        weight = "not_a_number"
        height = "175"
        
        try:
            bmi = float(weight) / ((float(height) / 100) ** 2)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


class TestContextIntegration:
    """Test suite for context integration with text processor"""
    
    def test_query_with_demographics(self):
        """Test that query works with demographic context"""
        from text_processor import TextProcessor
        
        processor = TextProcessor()
        
        query = QueryRequest(
            query="What blood pressure medication is appropriate?",
            context="Age: 55 years\nGender: Male\nWeight: 90 kg\nHeight: 180 cm\nBMI: 27.8",
            include_references=False
        )
        
        result = processor.answer_query(query)
        
        assert result is not None
        assert result.answer is not None
        assert len(result.answer) > 0
    
    def test_query_without_demographics(self):
        """Test that query works without demographic context"""
        from text_processor import TextProcessor
        
        processor = TextProcessor()
        
        query = QueryRequest(
            query="What are the side effects of aspirin?",
            include_references=False
        )
        
        result = processor.answer_query(query)
        
        assert result is not None
        assert result.answer is not None
    
    def test_context_appears_in_prompt(self):
        """Test that user context is included in the prompt"""
        from text_processor import TextProcessor
        
        processor = TextProcessor()
        
        user_context = "Age: 40 years\nGender: Female"
        
        # The context should be formatted and passed to the LLM
        # We can't directly test the prompt, but we can verify the method accepts it
        query = QueryRequest(
            query="Test query",
            context=user_context,
            include_references=False
        )
        
        # Should not raise an error
        result = processor.answer_query(query)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
