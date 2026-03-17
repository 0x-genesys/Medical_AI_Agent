"""
Image Processing Service using Microsoft BiomedCLIP
Follows PDF requirements: uses specialized medical vision models for clinical images
"""
# CRITICAL: Set OpenMP environment before ANY imports
# Multiple libraries (FAISS, PyTorch, NumPy, scikit-learn) load different OpenMP versions
# This must happen before torch, faiss, numpy imports to prevent SIGSEGV crashes
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

# Suppress deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from models import MedicalImage, ImageAnalysisResult, ImageModality
from logger import get_logger
from config import config
from session_manager import SessionManager


class ImageProcessor:
    """
    Processes medical images using MedCLIP for specialized clinical image understanding
    Uses LangChain for orchestration as per PDF requirements
    """
    
    def __init__(self, model_name: Optional[str] = None, vision_model: str = "BiomedCLIP", session_manager=None):
        self.model_name = model_name or config.model.llm_model
        self.logger = get_logger(__name__)
        
        # Load all conditions from medical knowledge base for BiomedCLIP matching
        self.medical_conditions = self._load_medical_conditions()
        self.logger.info(f"Loaded {len(self.medical_conditions)} conditions from knowledge base for BiomedCLIP matching")
        
        self.logger.info(f"Loading vision model: {vision_model}")
        
        # Device selection - auto-detect GPU (CUDA preferred, then MPS, then CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"BiomedCLIP will use CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("BiomedCLIP will use CPU (no GPU detected)")
        
        try:
            self.logger.info(f"Initializing BiomedCLIP on {self.device}")
            
            # Load Microsoft BiomedCLIP from HuggingFace
            # This model is pretrained on PMC-15M (15M medical image-caption pairs)
            self.clip_model, self.preprocess = create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.tokenizer = get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            self.context_length = 256  # BiomedCLIP context length
            
            self.logger.info("✓ BiomedCLIP loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load BiomedCLIP: {e}. Image analysis will be limited.")
            self.clip_model = None
            self.preprocess = None
            self.tokenizer = None
        
        self.llm = OllamaLLM(model=self.model_name, temperature=config.model.temperature)
        
        # Unified session manager (shared or create new)
        from session_manager import SessionManager
        self.session_manager = session_manager if session_manager else SessionManager()
        
        if not PYDICOM_AVAILABLE:
            self.logger.warning("pydicom not available. DICOM support disabled.")
        if not CV2_AVAILABLE:
            self.logger.warning("opencv-python not available. Advanced image processing disabled.")
        
        self.logger.info("Initialized ImageProcessor with MedCLIP and LangChain")
    
    def _load_medical_conditions(self) -> list:
        """
        Load imaging-specific medical conditions for MedCLIP matching
        Uses dedicated imaging_conditions.txt optimized for CT, X-ray, and MRI analysis
        
        Returns:
            List of condition names
        """
        conditions = []
        imaging_conditions_file = Path(__file__).parent / "data" / "imaging_conditions.txt"
        
        try:
            if not imaging_conditions_file.exists():
                self.logger.warning(f"Imaging conditions file not found: {imaging_conditions_file}")
                return []
            
            with open(imaging_conditions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        conditions.append(line)
            
            self.logger.info(f"Loaded {len(conditions)} imaging conditions from {imaging_conditions_file}")
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error loading medical conditions: {e}")
            return []
    
    def analyze_medical_image(
        self, 
        medical_image: MedicalImage, 
        clinical_context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ImageAnalysisResult:
        """
        Analyze medical image using MedCLIP and LangChain
        
        Args:
            medical_image: MedicalImage object
            clinical_context: Optional clinical context for interpretation
            
        Returns:
            ImageAnalysisResult with findings
        """
        if not medical_image.is_valid():
            self.logger.error("Invalid medical image provided")
            return ImageAnalysisResult()
        
        self.logger.info(f"Analyzing medical image with MedCLIP: {medical_image.image_path}")
        
        try:
            image_data, metadata = self._load_image(medical_image.image_path)
            
            medical_image.metadata.update(metadata)
            
            image_features = None
            medclip_findings = []
            retrieved_knowledge = ""
            
            # Step 1: Identify likely findings using BiomedCLIP
            if self.clip_model is not None:
                try:
                    # Step 2: Score image against ALL medical conditions from knowledge base
                    if not self.medical_conditions:
                        self.logger.warning("No medical conditions loaded - skipping MedCLIP matching")
                    else:
                        similarities = self.compute_image_text_similarity(
                            medical_image.image_path, 
                            self.medical_conditions
                        )
                        
                        # Get top 3 matches from entire knowledge base (reduced from 5 for focus)
                        top_indices = np.argsort(similarities)[-3:][::-1]
                        medclip_findings = [
                            (self.medical_conditions[idx], float(similarities[idx]))
                            for idx in top_indices
                        ]
                    
                    self.logger.info(f"MedCLIP top findings: {medclip_findings}")
                    
                    # Step 3: Query FAISS knowledge base directly (no LLM call)
                    try:
                        from text_processor import TextProcessor
                        
                        text_processor = TextProcessor()
                        
                        # Use FAISS directly without LLM - just semantic search
                        for finding, score in medclip_findings:
                            # Direct FAISS retrieval using the condition name
                            query_text = f"{finding} clinical features diagnosis treatment"
                            rag_context = text_processor._retrieve_rag_context(query_text, top_k=2)
                            
                            if rag_context and "No relevant" not in rag_context:
                                retrieved_knowledge += f"\n{'='*60}\n"
                                retrieved_knowledge += f"Knowledge for '{finding}' (MedCLIP confidence: {score:.2f}):\n"
                                retrieved_knowledge += f"{rag_context}\n"
                        
                        if retrieved_knowledge:
                            self.logger.info(f"Retrieved {len(retrieved_knowledge)} chars of knowledge from FAISS")
                        else:
                            self.logger.warning("No relevant knowledge retrieved from FAISS")
                            
                    except Exception as rag_error:
                        self.logger.warning(f"Could not query FAISS knowledge base: {rag_error}")
                        retrieved_knowledge = ""
                        
                except Exception as medclip_error:
                    self.logger.warning(f"MedCLIP processing failed: {medclip_error}")
            
            # Build enriched prompt with MedCLIP findings and retrieved knowledge
            prompt_template = PromptTemplate(
                input_variables=["modality", "body_part", "metadata", "context", "medclip_findings", "retrieved_knowledge", "conversation_context"],
                template="""As a medical imaging AI assistant, analyze this medical image.
        
        Image Information:
        - Modality: {modality}
        - Body Part: {body_part}
        - Metadata: {metadata}
        
        MedCLIP Visual Analysis (AI-detected findings):
        {medclip_findings}
        
        Relevant Medical Knowledge from Database:
        {retrieved_knowledge}
        
        Clinical Context:
        {context}
        
        Previous Conversation Context:
        {conversation_context}
        
        Based on the MedCLIP visual analysis and medical knowledge retrieved, provide a detailed analysis including:
        1. Key observations and anatomical structures visible
        2. Potential findings or abnormalities
        3. Differential considerations based on imaging findings
        4. Recommendations for follow-up or additional imaging
        5. Confidence level (low/medium/high)
        
        Return as JSON:
        {{
            "observations": ["list of key observations"],
            "potential_findings": ["list of findings"],
            "abnormalities": ["list of abnormalities if any"],
            "recommendations": ["list of recommendations"],
            "confidence_score": 0.85
        }}
        
        Important: This is for educational purposes. Always consult qualified radiologists.
        Return only valid JSON."""
            )
            
            chain = prompt_template | self.llm
            
            context_str = clinical_context if clinical_context else "No clinical context provided"
            
            # Format MedCLIP findings for display
            if medclip_findings:
                findings_str = "\n".join([
                    f"  - {finding} (confidence: {score:.2f})" 
                    for finding, score in medclip_findings
                ])
            else:
                findings_str = "MedCLIP analysis not available"
            
            # Get conversation context from session
            conversation_context = ""
            if session_id:
                conversation_context = self.session_manager.get_context(session_id)
                self.logger.info(f"Image analysis using session context: {len(conversation_context)} chars")
            
            # Use invoke() instead of deprecated run() to avoid config errors
            # Handle modality as either string or enum
            modality_str = medical_image.modality.value if hasattr(medical_image.modality, 'value') else str(medical_image.modality)
            
            result_dict = chain.invoke({
                "modality": modality_str,
                "body_part": medical_image.body_part or "Unknown",
                "metadata": str(medical_image.metadata),
                "context": context_str,
                "medclip_findings": findings_str,
                "retrieved_knowledge": retrieved_knowledge if retrieved_knowledge else "No relevant knowledge found in database",
                "conversation_context": conversation_context or "No previous context."
            })
            
            # Extract text from response dict
            response = result_dict.get('text', str(result_dict)) if isinstance(result_dict, dict) else result_dict
            
            result = self._parse_image_analysis_response(response)
            
            # Save interaction to session
            if session_id:
                self.session_manager.add_interaction(
                    session_id=session_id,
                    user_input=f"Image: {medical_image.image_path}, Modality: {modality_str}",
                    ai_response=response,
                    flow_type="image_analysis"
                )
            
            self.logger.info("Image analysis completed successfully with MedCLIP")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
            return ImageAnalysisResult(raw_response=str(e))
    
    def _extract_biomedclip_features(self, image_path: str) -> np.ndarray:
        """
        Extract features using BiomedCLIP model
        
        Args:
            image_path: Path to image
            
        Returns:
            Feature vector as numpy array
        """
        if self.clip_model is None:
            raise ValueError("BiomedCLIP model not loaded")
        
        # Load and preprocess image using BiomedCLIP's preprocess function
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
        
        return image_features.cpu().numpy()
    
    def compute_image_text_similarity(self, image_path: str, text_descriptions: list) -> np.ndarray:
        """
        Compute similarity between image and text descriptions using BiomedCLIP
        
        Args:
            image_path: Path to medical image
            text_descriptions: List of text descriptions
            
        Returns:
            Similarity scores as numpy array
        """
        if self.clip_model is None:
            raise ValueError("BiomedCLIP model not loaded")
        
        self.logger.info(f"Computing BiomedCLIP similarity for {len(text_descriptions)} descriptions")
        
        # Preprocess image using BiomedCLIP's preprocess function
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize text descriptions
        texts = self.tokenizer(text_descriptions, context_length=self.context_length).to(self.device)
        
        # Compute features and similarity
        with torch.no_grad():
            image_features, text_features, logit_scale = self.clip_model(image_tensor, texts)
            
            # Compute similarity scores using logit_scale (learned temperature parameter)
            logits = (logit_scale * image_features @ text_features.t()).detach()
            similarities = logits.squeeze(0).cpu().numpy()
        
        self.logger.info(f"BiomedCLIP similarity scores computed: min={similarities.min():.2f}, max={similarities.max():.2f}")
        return similarities
    
    def process_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process DICOM medical image
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM processing")
        
        self.logger.info(f"Processing DICOM file: {dicom_path}")
        
        ds = pydicom.dcmread(dicom_path)
        
        image_array = ds.pixel_array
        
        image_array = self._normalize_image(image_array)
        
        metadata = {
            'patient_id': str(ds.get('PatientID', 'Unknown')),
            'study_date': str(ds.get('StudyDate', 'Unknown')),
            'modality': str(ds.get('Modality', 'Unknown')),
            'body_part': str(ds.get('BodyPartExamined', 'Unknown')),
            'institution': str(ds.get('InstitutionName', 'Unknown')),
            'image_shape': image_array.shape
        }
        
        self.logger.info(f"DICOM processed: {metadata['modality']}, {metadata['body_part']}")
        
        return image_array, metadata
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for analysis
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        if CV2_AVAILABLE:
            image_array = self._enhance_contrast(image_array)
        
        return image_array
    
    def _load_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image and extract metadata"""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if path.suffix.lower() == '.dcm':
            if PYDICOM_AVAILABLE:
                return self.process_dicom(image_path)
            else:
                raise ImportError("pydicom required for DICOM files")
        else:
            image_array = self.preprocess_image(image_path)
            metadata = {
                'filename': path.name,
                'format': path.suffix.lower(),
                'image_shape': image_array.shape
            }
            return image_array, metadata
    
    def _normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image array to 0-255 range"""
        if image_array.max() > image_array.min():
            normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            return (normalized * 255).astype(np.uint8)
        return image_array.astype(np.uint8)
    
    def _enhance_contrast(self, image_array: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if not CV2_AVAILABLE:
            return image_array
        
        if len(image_array.shape) == 3:
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image_array)
    
    def _parse_image_analysis_response(self, response: str) -> ImageAnalysisResult:
        """Robust parser - handles markdown code blocks"""
        import re
        import json
        try:
            cleaned = response.strip()
            data = None
            
            # Extract from markdown code blocks
            if "```" in cleaned:
                pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                matches = re.findall(pattern, cleaned, re.DOTALL)
                for match in matches:
                    match = match.strip()
                    if match.startswith("{"):
                        try:
                            data = json.loads(match)
                            self.logger.info("Extracted JSON from markdown")
                            break
                        except:
                            continue
            
            # Find JSON by matching braces
            if data is None:
                start = cleaned.find("{")
                if start != -1:
                    count, end = 0, start
                    for i in range(start, len(cleaned)):
                        if cleaned[i] == '{': count += 1
                        elif cleaned[i] == '}':
                            count -= 1
                            if count == 0:
                                end = i
                                break
                    if end > start:
                        try:
                            data = json.loads(cleaned[start:end+1])
                            self.logger.info("Extracted JSON from text")
                        except:
                            pass
            
            if data is None:
                data = json.loads(cleaned)
            
            return ImageAnalysisResult(
                observations=data.get("observations", []),
                potential_findings=data.get("potential_findings", []),
                abnormalities=data.get("abnormalities", []),
                confidence_score=data.get("confidence_score", 0.5),
                recommendations=data.get("recommendations", []),
                raw_response=response
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(f"Could not parse JSON: {str(e)}")
            return ImageAnalysisResult(raw_response=response)
