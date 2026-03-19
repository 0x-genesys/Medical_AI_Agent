"""
Multimodal Fusion Service using LangChain
Integrates text and image pipelines using LangChain chains as per PDF requirements
No custom orchestration - uses LangChain's built-in chain mechanisms
"""
import json
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from models import (
    MedicalText, MedicalImage, MultimodalAnalysisResult,
    TextAnalysisResult, ImageAnalysisResult
)
from logger import get_logger
from config import config


class MultimodalFusion:
    """
    Fuses multimodal information using LangChain chains.
    
    Integrates text and image analysis pipelines using LangChain's orchestration
    capabilities. Implements cross-modal queries and retrieval combining BioBERT
    text embeddings with BiomedCLIP image embeddings for comprehensive medical
    assessment.
    
    Uses Dependency Injection pattern to share processor instances across the
    application, ensuring consistent session management and avoiding duplicate
    model loading.
    
    Attributes:
        text_processor (TextProcessor): Shared text processing instance
        image_processor (ImageProcessor): Shared image processing instance
        session_manager (SessionManager): Shared session management instance
        llm (OllamaLLM): LangChain LLM for synthesis
    """
    
    def __init__(self, text_processor, image_processor, session_manager, model_name: Optional[str] = None):
        """
        Initialize multimodal fusion with shared dependencies (Dependency Injection).
        
        Accepts pre-initialized processor instances to ensure single model loading
        and shared session management across all analysis flows.
        
        Args:
            text_processor (TextProcessor): Shared TextProcessor instance with BioBERT
            image_processor (ImageProcessor): Shared ImageProcessor instance with BiomedCLIP
            session_manager (SessionManager): Shared SessionManager for cross-flow context
            model_name (Optional[str]): Override LLM model name (defaults from config)
        
        Returns:
            None
        """
        self.model_name = model_name or config.model.llm_model
        self.text_processor = text_processor  # Use shared instance
        self.image_processor = image_processor  # Use shared instance
        self.session_manager = session_manager  # Use shared instance
        self.logger = get_logger(__name__)
        
        self.llm = OllamaLLM(model=self.model_name, temperature=config.model.temperature)
        
        self.logger.info("Initialized MultimodalFusion with shared processors and session manager")
    
    def analyze_multimodal(
        self,
        medical_text: Optional[MedicalText] = None,
        medical_image: Optional[MedicalImage] = None,
        session_id: Optional[str] = None
    ) -> MultimodalAnalysisResult:
        """
        Perform integrated multimodal analysis combining text and image data.
        
        Analyzes both clinical text (using BioBERT) and medical images (using BiomedCLIP)
        then synthesizes the findings into a unified clinical assessment with differential
        diagnosis and recommendations.
        
        Process:
        1. Analyze text component with BioBERT embeddings and RAG
        2. Analyze image component with BiomedCLIP and medical condition matching
        3. Synthesize both analyses using LangChain chains
        4. Generate integrated assessment with differential diagnosis
        
        Args:
            medical_text (Optional[MedicalText]): Clinical text data to analyze
            medical_image (Optional[MedicalImage]): Medical image data to analyze
            session_id (Optional[str]): Session ID for context tracking
            
        Returns:
            MultimodalAnalysisResult: Integrated analysis with text results, image results,
                                     differential diagnosis, and recommendations
        """
        self.logger.info("Starting LangChain multimodal analysis")
        
        text_analysis = None
        image_analysis = None
        
        if medical_text and medical_text.is_valid():
            self.logger.info("Analyzing text component with BioBERT")
            text_analysis = self.text_processor.analyze_clinical_text(medical_text, session_id=session_id)
        
        if medical_image and medical_image.is_valid():
            self.logger.info("Analyzing image component with BiomedCLIP")
            clinical_context = medical_text.text if medical_text else None
            image_analysis = self.image_processor.analyze_medical_image(
                medical_image,
                clinical_context=clinical_context,
                session_id=session_id
            )
        
        if not text_analysis and not image_analysis:
            self.logger.error("No valid data provided for analysis")
            return MultimodalAnalysisResult()
        
        integrated_result = self._synthesize_with_langchain(text_analysis, image_analysis, session_id)
        
        result = MultimodalAnalysisResult(
            text_analysis=text_analysis,
            image_analysis=image_analysis,
            integrated_assessment=integrated_result.get("integrated_assessment"),
            differential_diagnosis=integrated_result.get("differential_diagnosis", []),
            recommended_workup=integrated_result.get("recommended_workup", []),
            clinical_summary=integrated_result.get("clinical_summary"),
            confidence_level=integrated_result.get("confidence_level", "medium"),
            raw_response=integrated_result.get("raw_response")
        )
        
        self.logger.info("LangChain multimodal analysis completed successfully")
        return result
    
    def _synthesize_with_langchain(
        self,
        text_analysis: Optional[TextAnalysisResult],
        image_analysis: Optional[ImageAnalysisResult],
        session_id: Optional[str] = None
    ) -> dict:
        """
        Synthesize text and image findings using LangChain sequential chain.
        
        Implements cross-modal integration by combining text analysis results
        (symptoms, medications, history) with image analysis results (observations,
        findings) into a unified clinical assessment.
        
        Uses LangChain's prompt templates and chain orchestration to generate:
        - Integrated clinical picture
        - Differential diagnosis list (ranked)
        - Recommended diagnostic workup
        - Clinical summary
        - Confidence level
        
        Args:
            text_analysis (Optional[TextAnalysisResult]): Results from BioBERT text analysis
            image_analysis (Optional[ImageAnalysisResult]): Results from BiomedCLIP image analysis
            session_id (Optional[str]): Session ID to retrieve conversation context
            
        Returns:
            dict: Dictionary containing integrated_assessment, differential_diagnosis,
                 recommended_workup, clinical_summary, confidence_level, and raw_response
        """
        self.logger.info("Synthesizing multimodal findings with LangChain chains")
        
        text_summary = self._prepare_text_summary(text_analysis)
        image_summary = self._prepare_image_summary(image_analysis)
        
        # Get conversation context from shared session
        conversation_context = ""
        if session_id:
            conversation_context = self.session_manager.get_context(session_id)
            self.logger.info(f"Multimodal synthesis using session context: {len(conversation_context)} chars")
        
        synthesis_prompt = PromptTemplate(
            input_variables=["text_findings", "image_findings", "conversation_context"],
            template="""As a clinical decision support system, synthesize these multimodal findings into a comprehensive assessment.
        
        Previous Conversation:
        {conversation_context}
        
        Clinical Text Analysis:
        {text_findings}
        
        Imaging Analysis:
        {image_findings}
        
        Provide an integrated analysis with:
        1. Integrated clinical picture (synthesize text and imaging findings)
        2. Most likely differential diagnoses (ranked by probability)
        3. Recommended diagnostic workup (labs, additional imaging, consultations)
        4. Clinical summary (2-3 sentences)
        5. Confidence level (low/medium/high)
        
        Return as JSON:
        {{
            "integrated_assessment": "comprehensive integrated assessment",
            "differential_diagnosis": ["diagnosis 1", "diagnosis 2", "diagnosis 3"],
            "recommended_workup": ["test 1", "test 2", "consultation"],
            "clinical_summary": "concise clinical summary",
            "confidence_level": "medium"
        }}
        
        Important: For educational purposes only. Clinical decisions require qualified healthcare providers.
        Return only valid JSON."""
        )
        
        chain = synthesis_prompt | self.llm
        
        try:
            result = chain.invoke({
                "text_findings": text_summary, 
                "image_findings": image_summary,
                "conversation_context": conversation_context or "No previous context."
            })
            
            # Extract string from response
            response_str = result.get('text', str(result)) if isinstance(result, dict) else result
            
            parsed_result = self._parse_synthesis_response(response_str)
            parsed_result["raw_response"] = response_str  # Store raw response
            
            # Save to session
            if session_id:
                self.session_manager.add_interaction(
                    session_id=session_id,
                    user_input=f"Multimodal Analysis: Text={'Yes' if text_analysis else 'No'}, Image={'Yes' if image_analysis else 'No'}",
                    ai_response=response_str,
                    flow_type="multimodal"
                )
            
            self.logger.info("LangChain synthesis completed successfully")
            return parsed_result
        except Exception as e:
            self.logger.error(f"Error during LangChain synthesis: {str(e)}", exc_info=True)
            return self._create_fallback_synthesis(text_analysis, image_analysis)
    
    def create_multimodal_chain(self):
        """
        Create a LangChain runnable sequence for multimodal processing.
        
        Demonstrates modern LangChain Expression Language (LCEL) orchestration
        using RunnableParallel to process text and image in parallel, then
        synthesize results in a sequential chain.
        
        This method showcases LangChain's composability and is an example of
        how to build complex AI pipelines declaratively.
        
        Returns:
            RunnableSequence: LangChain runnable that processes multimodal inputs
                            and returns integrated analysis
        """
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        
        text_prompt = PromptTemplate(
            input_variables=["clinical_text"],
            template="Analyze this clinical text and extract key findings: {clinical_text}"
        )
        
        image_prompt = PromptTemplate(
            input_variables=["image_description"],
            template="Analyze these imaging findings: {image_description}"
        )
        
        synthesis_prompt = PromptTemplate(
            input_variables=["text_findings", "image_findings"],
            template="""Synthesize these multimodal findings:
            
            Text: {text_findings}
            Image: {image_findings}
            
            Provide integrated assessment and diagnosis."""
        )
        
        # Create parallel analysis chains
        text_chain = text_prompt | self.llm
        image_chain = image_prompt | self.llm
        
        # Create overall chain using RunnableParallel and synthesis
        overall_chain = (
            RunnableParallel(
                text_findings=text_chain,
                image_findings=image_chain
            )
            | synthesis_prompt
            | self.llm
        )
        
        self.logger.info("Created LangChain RunnableSequence for multimodal processing")
        return overall_chain
    
    def _prepare_text_summary(self, text_analysis: Optional[TextAnalysisResult]) -> str:
        """
        Prepare a concise text analysis summary for multimodal synthesis.
        
        Extracts key information from text analysis results and formats it
        into a compact summary string suitable for inclusion in the synthesis
        prompt.
        
        Args:
            text_analysis (Optional[TextAnalysisResult]): Text analysis results
        
        Returns:
            str: Formatted summary of text findings or "No text analysis available"
        """
        if not text_analysis:
            return "No text analysis available"
        
        parts = []
        if text_analysis.chief_complaints:
            parts.append(f"Chief Complaints: {', '.join(text_analysis.chief_complaints[:3])}")
        if text_analysis.symptoms:
            parts.append(f"Symptoms: {', '.join(text_analysis.symptoms[:5])}")
        if text_analysis.summary:
            parts.append(f"Summary: {text_analysis.summary}")
        
        return " | ".join(parts) if parts else "No significant text findings"
    
    def _prepare_image_summary(self, image_analysis: Optional[ImageAnalysisResult]) -> str:
        """
        Prepare a concise image analysis summary for multimodal synthesis.
        
        Extracts key observations, findings, and abnormalities from image analysis
        and formats them into a compact summary string.
        
        Args:
            image_analysis (Optional[ImageAnalysisResult]): Image analysis results
        
        Returns:
            str: Formatted summary of imaging findings or "No image analysis available"
        """
        if not image_analysis:
            return "No image analysis available"
        
        parts = []
        if image_analysis.observations:
            parts.append(f"Observations: {', '.join(image_analysis.observations[:3])}")
        if image_analysis.potential_findings:
            parts.append(f"Findings: {', '.join(image_analysis.potential_findings[:3])}")
        if image_analysis.abnormalities:
            parts.append(f"Abnormalities: {', '.join(image_analysis.abnormalities[:3])}")
        
        return " | ".join(parts) if parts else "No significant imaging findings"
    
    def _parse_synthesis_response(self, response: str) -> dict:
        """
        Parse LLM synthesis response into structured dictionary.
        
        Attempts to parse the LLM's JSON response. If parsing fails,
        creates a fallback dictionary with truncated response text.
        
        Args:
            response (str): Raw LLM response string (expected to be JSON)
        
        Returns:
            dict: Parsed synthesis results with keys: integrated_assessment,
                 differential_diagnosis, recommended_workup, clinical_summary,
                 confidence_level
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse JSON response, using fallback")
            return {
                "integrated_assessment": response[:500],
                "differential_diagnosis": [],
                "recommended_workup": [],
                "clinical_summary": response[:200],
                "confidence_level": "medium"
            }
    
    def _create_fallback_synthesis(
        self,
        text_analysis: Optional[TextAnalysisResult],
        image_analysis: Optional[ImageAnalysisResult]
    ) -> dict:
        """
        Create a basic fallback synthesis when LangChain processing fails.
        
        Provides a simple text-based synthesis by concatenating summaries
        from text and image analysis without complex integration. Used as
        a graceful degradation when the LLM synthesis fails.
        
        Args:
            text_analysis (Optional[TextAnalysisResult]): Text analysis results
            image_analysis (Optional[ImageAnalysisResult]): Image analysis results
        
        Returns:
            dict: Basic synthesis dictionary with minimal integration
        """
        summary_parts = []
        
        if text_analysis and text_analysis.summary:
            summary_parts.append(f"Clinical: {text_analysis.summary}")
        
        if image_analysis and image_analysis.observations:
            summary_parts.append(f"Imaging: {', '.join(image_analysis.observations[:2])}")
        
        return {
            "integrated_assessment": " | ".join(summary_parts),
            "differential_diagnosis": [],
            "recommended_workup": ["Further clinical evaluation recommended"],
            "clinical_summary": " ".join(summary_parts),
            "confidence_level": "low"
        }
