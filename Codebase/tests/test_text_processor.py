"""
Unit tests for TextProcessor with BioBERT and LangChain
Tests semantic search, embeddings, and LangChain integration
"""
import pytest
from models import MedicalText, QueryRequest
from text_processor import TextProcessor


@pytest.fixture
def text_processor():
    """Fixture for TextProcessor instance with BioBERT"""
    return TextProcessor()


@pytest.fixture
def sample_clinical_text():
    """Sample clinical text for testing"""
    return """
    Patient: John Doe
    Chief Complaint: Chest pain
    History: 45-year-old male presenting with acute chest pain.
    Symptoms: Shortness of breath, diaphoresis, radiating pain to left arm.
    Medications: Aspirin 81mg daily, Lisinopril 10mg daily
    Vital Signs: BP 140/90, HR 95, RR 18, Temp 98.6F
    Lab Results: Troponin elevated at 0.5 ng/mL
    """


def test_text_processor_initialization(text_processor):
    """Test TextProcessor initializes with BioBERT and LangChain"""
    assert text_processor is not None
    assert text_processor.model_name == "llama3"
    assert text_processor.embedder is not None
    assert text_processor.llm is not None
    assert text_processor.text_splitter is not None


def test_biobert_embeddings(text_processor):
    """Test BioBERT embedding generation"""
    text = "Patient has hypertension and diabetes"
    embedding = text_processor.embedder.encode(text)
    
    assert embedding is not None
    assert len(embedding) > 0
    assert embedding.shape[0] == 768


def test_analyze_clinical_text_valid(text_processor, sample_clinical_text):
    """Test analysis with BioBERT embeddings and LangChain"""
    medical_text = MedicalText(text=sample_clinical_text)
    result = text_processor.analyze_clinical_text(medical_text)
    
    assert result is not None
    assert hasattr(result, 'summary')
    assert hasattr(result, 'chief_complaints')
    assert hasattr(result, 'symptoms')
    assert hasattr(result, 'entities')


def test_analyze_clinical_text_invalid(text_processor):
    """Test analysis with invalid text"""
    medical_text = MedicalText(text="")
    result = text_processor.analyze_clinical_text(medical_text)
    
    assert result is not None


def test_answer_query_with_langchain(text_processor):
    """Test answering query using LangChain QA chain"""
    query_request = QueryRequest(
        query="What are the symptoms of myocardial infarction?",
        include_references=True
    )
    
    result = text_processor.answer_query(query_request)
    
    assert result is not None
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.confidence >= 0.0


def test_semantic_search_with_biobert(text_processor):
    """Test semantic search using BioBERT embeddings and FAISS"""
    context = """
    Patient has a history of hypertension and diabetes.
    Recent labs show elevated glucose levels.
    Blood pressure is well controlled on current medications.
    """
    
    query = "What is the patient's blood sugar status?"
    
    docs = text_processor._semantic_search(context, query, top_k=2)
    
    assert isinstance(docs, list)
    assert len(docs) <= 2


def test_index_documents(text_processor):
    """Test document indexing with BioBERT"""
    documents = [
        "Patient has diabetes mellitus type 2",
        "Hypertension is well controlled",
        "No known drug allergies"
    ]
    
    text_processor.index_documents(documents)
    
    assert text_processor.documents is not None
    assert len(text_processor.documents) > 0


def test_extract_entities_with_langchain(text_processor, sample_clinical_text):
    """Test entity extraction using LangChain"""
    entities = text_processor.extract_entities(sample_clinical_text)
    
    assert isinstance(entities, list)


def test_answer_query_with_context(text_processor):
    """Test query with additional context using LangChain"""
    query_request = QueryRequest(
        query="What is the diagnosis?",
        context="Patient with elevated troponin and chest pain",
        include_references=False
    )
    
    result = text_processor.answer_query(query_request)
    
    assert result is not None
    assert len(result.references) == 0
