"""
Data models for Multimodal Medical Assistant
Following SOLID principles with clear data structures
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DataType(Enum):
    """
    Enumeration of supported medical data types.
    
    Defines the various types of medical data that can be processed
    by the multimodal medical assistant system.
    
    Values:
        TEXT: Plain text medical data (notes, reports)
        IMAGE: Medical images (X-ray, MRI, CT scans)
        DICOM: DICOM format medical imaging files
        REPORT: Structured medical reports
        MULTIMODAL: Combined text and image data
    """
    TEXT = "text"
    IMAGE = "image"
    DICOM = "dicom"
    REPORT = "report"
    MULTIMODAL = "multimodal"


class ImageModality(Enum):
    """
    Enumeration of supported medical imaging modalities.
    
    Represents different types of medical imaging techniques used
    in healthcare for diagnosis and treatment planning.
    
    Values:
        XRAY: X-ray radiography
        MRI: Magnetic Resonance Imaging
        CT: Computed Tomography
        ULTRASOUND: Ultrasound imaging
        PATHOLOGY: Pathology slide images
        UNKNOWN: Unspecified or unknown modality
    """
    XRAY = "xray"
    MRI = "mri"
    CT = "ct"
    ULTRASOUND = "ultrasound"
    PATHOLOGY = "pathology"
    UNKNOWN = "unknown"


@dataclass
class MedicalText:
    """Structured medical text data"""
    text: str
    data_type: str = "clinical_note"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """
        Validate if the medical text data is valid for processing.
        
        Checks that the text field is not None, empty, or contains only whitespace.
        
        Returns:
            bool: True if text is valid (non-empty and not just whitespace), False otherwise
        """
        return bool(self.text and self.text.strip())


@dataclass
class MedicalImage:
    """
    Structured medical image data container.
    
    Stores metadata and path information for medical images including
    imaging modality, body part, and additional metadata.
    
    Attributes:
        image_path (str): File system path to the image file
        modality (ImageModality): Type of medical imaging used
        body_part (Optional[str]): Anatomical region imaged (e.g., 'chest', 'knee')
        timestamp (datetime): When the image data was created
        metadata (Dict[str, Any]): Additional custom metadata
    """
    image_path: str
    modality: ImageModality = ImageModality.UNKNOWN
    body_part: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        return bool(self.image_path)


@dataclass
class ClinicalEntity:
    """
    Represents a single extracted clinical entity from medical text.
    
    Clinical entities are medical concepts identified in text such as
    symptoms, medications, diagnoses, procedures, etc.
    
    Attributes:
        entity_type (str): Type of entity (e.g., 'symptom', 'medication', 'diagnosis')
        value (str): The actual text value of the entity
        confidence (float): Confidence score of the extraction (0.0 to 1.0)
        context (Optional[str]): Surrounding context where entity was found
    """
    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


@dataclass
class TextAnalysisResult:
    """
    Result container for clinical text analysis.
    
    Stores all extracted information from analyzing medical text including
    structured fields, clinical entities, and raw LLM responses.
    
    Attributes:
        chief_complaints (List[str]): Main patient complaints or reasons for visit
        symptoms (List[str]): Reported or observed symptoms
        medical_history (List[str]): Relevant past medical history items
        medications (List[str]): Current or mentioned medications
        lab_findings (List[str]): Laboratory test results mentioned
        entities (List[ClinicalEntity]): Extracted clinical entities with metadata
        summary (Optional[str]): Brief clinical summary of the text
        raw_response (Optional[str]): Unprocessed LLM response for transparency
    """
    chief_complaints: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    lab_findings: List[str] = field(default_factory=list)
    entities: List[ClinicalEntity] = field(default_factory=list)
    summary: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class ImageAnalysisResult:
    """
    Result container for medical image analysis.
    
    Stores observations, findings, and recommendations from analyzing
    medical images using vision models.
    
    Attributes:
        observations (List[str]): Visual observations from the image
        potential_findings (List[str]): Potential clinical findings identified
        abnormalities (List[str]): Detected abnormalities or irregularities
        confidence_score (float): Overall confidence in the analysis (0.0 to 1.0)
        recommendations (List[str]): Suggested follow-up actions
        raw_response (Optional[str]): Unprocessed LLM response for transparency
    """
    observations: List[str] = field(default_factory=list)
    potential_findings: List[str] = field(default_factory=list)
    abnormalities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None


@dataclass
class MultimodalAnalysisResult:
    """
    Result container for integrated multimodal analysis.
    
    Combines results from both text and image analysis into a unified
    clinical assessment with differential diagnosis and recommendations.
    
    Attributes:
        text_analysis (Optional[TextAnalysisResult]): Results from text analysis
        image_analysis (Optional[ImageAnalysisResult]): Results from image analysis
        integrated_assessment (Optional[str]): Combined clinical assessment
        differential_diagnosis (List[str]): Possible diagnoses based on all data
        recommended_workup (List[str]): Suggested additional tests or procedures
        clinical_summary (Optional[str]): Overall clinical summary
        confidence_level (str): Confidence in integrated assessment ('low', 'medium', 'high')
        raw_response (Optional[str]): Unprocessed LLM response for transparency
    """
    text_analysis: Optional[TextAnalysisResult] = None
    image_analysis: Optional[ImageAnalysisResult] = None
    integrated_assessment: Optional[str] = None
    differential_diagnosis: List[str] = field(default_factory=list)
    recommended_workup: List[str] = field(default_factory=list)
    clinical_summary: Optional[str] = None
    confidence_level: str = "medium"
    raw_response: Optional[str] = None


@dataclass
class QueryRequest:
    """
    Request container for medical knowledge queries.
    
    Encapsulates a user's medical question along with optional context
    and preferences for the response.
    
    Attributes:
        query (str): The medical question or query text
        context (Optional[str]): Additional context for the query
        include_references (bool): Whether to include references in response
    """
    query: str
    context: Optional[str] = None
    include_references: bool = True


@dataclass
class QueryResponse:
    """
    Response container for medical knowledge queries.
    
    Contains the answer to a medical query along with confidence score,
    references, and metadata.
    
    Attributes:
        answer (str): The generated answer to the query
        confidence (float): Confidence in the answer (0.0 to 1.0)
        references (List[str]): Supporting references or citations
        timestamp (datetime): When the response was generated
    """
    answer: str
    confidence: float
    references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
