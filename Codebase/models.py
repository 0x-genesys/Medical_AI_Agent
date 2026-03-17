"""
Data models for Multimodal Medical Assistant
Following SOLID principles with clear data structures
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DataType(Enum):
    """Types of medical data"""
    TEXT = "text"
    IMAGE = "image"
    DICOM = "dicom"
    REPORT = "report"
    MULTIMODAL = "multimodal"


class ImageModality(Enum):
    """Medical image modalities"""
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
        return bool(self.text and self.text.strip())


@dataclass
class MedicalImage:
    """Structured medical image data"""
    image_path: str
    modality: ImageModality = ImageModality.UNKNOWN
    body_part: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        return bool(self.image_path)


@dataclass
class ClinicalEntity:
    """Extracted clinical entity"""
    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


@dataclass
class TextAnalysisResult:
    """Result of text analysis"""
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
    """Result of image analysis"""
    observations: List[str] = field(default_factory=list)
    potential_findings: List[str] = field(default_factory=list)
    abnormalities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None


@dataclass
class MultimodalAnalysisResult:
    """Combined multimodal analysis result"""
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
    """User query request"""
    query: str
    context: Optional[str] = None
    include_references: bool = True


@dataclass
class QueryResponse:
    """Response to user query"""
    answer: str
    confidence: float
    references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
