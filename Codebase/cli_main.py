"""
CLI-Based Healthcare Medical Assistant
Focus: Interactive command-line healthcare solution

Features:
1. Unified session management across all clinical workflows
2. Smart routing: text queries, clinical report files, medical images
3. CLI-only interface for healthcare professionals
4. HIPAA-compliant PHI handling
5. Multimodal analysis: text, images, and integrated assessments
"""
# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
# CRITICAL FIX: Disable tokenizers parallelism to prevent BiomedCLIP crash
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datetime import datetime
from typing import Optional, Dict
from pathlib import Path
import uuid

from text_processor import TextProcessor
from image_processor import ImageProcessor
from multimodal_fusion import MultimodalFusion
from models import (
    MedicalText, MedicalImage, ImageModality,
    QueryRequest
)
from logger import get_logger

logger = get_logger(__name__)


class MedicalAssistantOrchestrator:
    """
    Healthcare CLI Orchestrator
    Provides intelligent routing and unified session management for clinical workflows
    """
    
    def __init__(self):
        # Create SHARED session manager first
        from session_manager import SessionManager
        self.shared_session_manager = SessionManager()
        
        # Pass shared session manager to all processors
        self.text_processor = TextProcessor(session_manager=self.shared_session_manager)
        self.image_processor = ImageProcessor(session_manager=self.shared_session_manager)
        
        # Inject shared dependencies into MultimodalFusion (Dependency Injection)
        self.multimodal_fusion = MultimodalFusion(
            text_processor=self.text_processor,
            image_processor=self.image_processor,
            session_manager=self.shared_session_manager
        )
        
        # Shared session for unified context across ALL flows
        self._session_id = None
        self._session_name = None
        self._interaction_count = 0
        logger.info("✓ Medical Assistant initialized with SHARED session management across all processors")
    
    def get_session_id(self) -> str:
        """Get or create shared session ID for all flows"""
        if self._session_id is None:
            self._session_id = str(uuid.uuid4())
            self._session_name = f"Session-{datetime.now().strftime('%H:%M')}"
            self._interaction_count = 0
            logger.info(f"✓ Created new session: {self._session_name} ({self._session_id[:8]}...)")
        return self._session_id
    
    def get_session_info(self) -> dict:
        """Get current session information"""
        if self._session_id is None:
            return {"active": False, "name": "No active session", "id": None, "interactions": 0}
        return {
            "active": True,
            "name": self._session_name,
            "id": self._session_id[:8],
            "interactions": self._interaction_count
        }
    
    def reset_session(self):
        """Clear current session and start fresh"""
        if self._session_id:
            # Clear from shared session manager
            self.shared_session_manager.clear_session(self._session_id)
            logger.info(f"Cleared session: {self._session_name}")
        
        self._session_id = None
        self._session_name = None
        self._interaction_count = 0
        logger.info("✓ Session reset - ready for new conversation")
    
    def set_session_name(self, name: str):
        """Set intelligent session name based on content"""
        if self._session_id and not self._session_name.startswith("Session-"):
            return  # Already has custom name
        self._session_name = name[:50]  # Limit length
        logger.info(f"Session renamed to: {self._session_name}")
    
    def increment_interaction(self):
        """Track interaction count for session"""
        self._interaction_count += 1
    
    def cleanup(self):
        """Cleanup resources to prevent semaphore leaks"""
        try:
            logger.info("Cleaning up resources...")
            
            # Clear CUDA cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✓ Cleared CUDA cache")
            
            # Clear session manager
            if self._session_id:
                self.shared_session_manager.clear_session(self._session_id)
            
            # Delete heavy model references
            del self.text_processor
            del self.image_processor
            del self.multimodal_fusion
            
            logger.info("✓ Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup error (non-critical): {e}")
    
    def analyze_report_flow(self, file_path: str, data_type: str = "clinical_note") -> dict:
        """
        Flow 1: Analyze clinical report FILES only (.txt, .doc, etc)
        For raw text queries, automatically routes to answer_query_flow
        """
        logger.info("=== Starting Report Analysis Flow ===")
        logger.info(f"📄 Analyzing file: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return {"error": f"File not found: {file_path}", "flow": "report_analysis"}
        
        try:
            with open(path, 'r') as f:
                text = f.read()
            logger.info(f"✓ Loaded {len(text)} characters from file")
        except Exception as e:
            logger.error(f"❌ Could not read file: {e}")
            return {"error": f"Could not read file: {e}", "flow": "report_analysis"}
        
        # Use shared session
        session_id = self.get_session_id()
        self.increment_interaction()
        
        # Intelligent session naming from first report
        if self._interaction_count == 1:
            filename = path.stem[:30]  # Use filename as session name
            self.set_session_name(f"Report: {filename}")
        
        medical_text = MedicalText(text=text, data_type=data_type)
        result = self.text_processor.analyze_clinical_text(medical_text, session_id=session_id)
        
        if result.summary or result.chief_complaints or result.symptoms:
            output = {
                "flow": "report_analysis",
                "chief_complaints": result.chief_complaints,
                "symptoms": result.symptoms,
                "medical_history": result.medical_history,
                "medications": result.medications,
                "lab_findings": result.lab_findings,
                "summary": result.summary,
                "raw_response": result.raw_response,
                "parsed": True
            }
        else:
            output = {
                "flow": "report_analysis",
                "raw_response": result.raw_response,
                "parsed": False,
                "summary": "Unable to parse structured output."
            }
        
        logger.info("=== Report Analysis Flow Completed ===")
        return output
    
    def answer_query_flow(self, query: str, context: Optional[str] = None) -> dict:
        """
        Flow 2: Answer medical questions with RAW TEXT (not files)
        Uses conversational context from shared session
        """
        logger.info("=== Starting Query Flow ===")
        logger.info(f"💬 Query: {query[:80]}...")
        
        # Use shared session for context
        session_id = self.get_session_id()
        self.increment_interaction()
        
        # Intelligent session naming from first query
        if self._interaction_count == 1:
            query_summary = query[:40] + "..." if len(query) > 40 else query
            self.set_session_name(f"Query: {query_summary}")
        
        query_request = QueryRequest(query=query, context=context or "")
        result = self.text_processor.answer_query(query_request, session_id=session_id)
        
        output = {
            "flow": "query",
            "query": query,
            "answer": result.answer,
            "confidence": result.confidence,
            "references": result.references
        }
        
        logger.info("=== Query Flow Completed ===")
        return output
    
    def analyze_image_flow(
        self, 
        image_path: str, 
        modality: str = "xray",
        body_part: Optional[str] = None,
        clinical_context: Optional[str] = None
    ) -> dict:
        """
        Flow 3: Analyze medical IMAGE files only
        """
        logger.info("=== Starting Image Analysis Flow ===")
        logger.info(f"🖼️  Analyzing image: {image_path}")
        logger.info(f"📋 Modality: {modality}, Body part: {body_part or 'unspecified'}")
        
        # Use shared session
        session_id = self.get_session_id()
        self.increment_interaction()
        
        medical_image = MedicalImage(
            image_path=image_path,
            modality=ImageModality[modality.upper()],
            body_part=body_part
        )
        
        result = self.image_processor.analyze_medical_image(
            medical_image,
            clinical_context=clinical_context,
            session_id=session_id
        )
        
        output = {
            "flow": "image_analysis",
            "observations": result.observations,
            "potential_findings": result.potential_findings,
            "abnormalities": result.abnormalities,
            "confidence_score": result.confidence_score,
            "recommendations": result.recommendations,
            "raw_response": result.raw_response
        }
        
        logger.info("=== Image Analysis Flow Completed ===")
        return output
    
    def multimodal_analysis_flow(
        self,
        user_input: Optional[str] = None,
        modality: str = "xray",
        body_part: Optional[str] = None,
        text_file: Optional[str] = None,
        image_file: Optional[str] = None
    ) -> Dict:
        """
        Flow 4: Smart multimodal routing based on input type
        - File path (.txt, .doc) → analyze_report_flow
        - Image path (.jpg, .png) → analyze_image_flow  
        - Raw text → answer_query_flow
        """
        logger.info("=== Starting Multimodal Analysis Flow ===")
        
        # True multimodal fusion: both text and image provided
        if text_file and image_file:
            logger.info("🔬 True multimodal fusion: Text + Image")
            session_id = self.get_session_id()
            self.increment_interaction()
            
            # Load text
            try:
                with open(text_file, 'r') as f:
                    text_content = f.read()
                medical_text = MedicalText(text=text_content)
            except Exception as e:
                return {"error": f"Error reading text file: {e}", "flow": "multimodal_fusion"}
            
            # Create image object
            medical_image = MedicalImage(
                image_path=image_file,
                modality=ImageModality[modality.upper()],
                body_part=body_part
            )
            
            # Perform fusion analysis
            result = self.multimodal_fusion.analyze_multimodal(
                medical_text=medical_text,
                medical_image=medical_image,
                session_id=session_id
            )
            
            return {
                "flow": "multimodal_fusion",
                "integrated_assessment": result.integrated_assessment,
                "differential_diagnosis": result.differential_diagnosis,
                "recommended_workup": result.recommended_workup,
                "clinical_summary": result.clinical_summary,
                "confidence_level": result.confidence_level,
                "raw_response": result.raw_response
            }
        
        # Single input routing
        if user_input:
            logger.info("🔀 Smart routing based on input type...")
            path = Path(user_input)
            
            # Route 1: File exists and is a document → Report analysis
            if path.exists() and path.is_file():
                ext = path.suffix.lower()
                
                # Image file extensions
                image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.dcm', '.tif', '.tiff'}
                # Text/document file extensions
                doc_exts = {'.txt', '.doc', '.docx', '.pdf', '.rtf'}
                
                if ext in image_exts:
                    logger.info(f"  ➜ Detected IMAGE file → Routing to analyze_image_flow")
                    return self.analyze_image_flow(user_input, modality, body_part)
                elif ext in doc_exts:
                    logger.info(f"  ➜ Detected DOCUMENT file → Routing to analyze_report_flow")
                    return self.analyze_report_flow(user_input)
                else:
                    logger.warning(f"  ⚠️  Unknown file type: {ext}")
                    return {"error": f"Unsupported file type: {ext}", "flow": "multimodal"}
            
            # Route 2: Raw text → Query flow
            else:
                logger.info(f"  ➜ Detected RAW TEXT → Routing to answer_query_flow")
                return self.answer_query_flow(user_input)
        
        return {"error": "No input provided", "flow": "multimodal"}


def run_cli_mode():
    """
    Interactive Healthcare CLI Mode
    Primary interface for healthcare professionals to analyze clinical data
    """
    orchestrator = MedicalAssistantOrchestrator()
    
    print("\n" + "="*70)
    print("  🏥 Healthcare Medical Assistant - CLI Interface")
    print("="*70)
    print("\n💡 Interactive Memory-Driven Clinical Assistant")
    print("   All interactions preserved in session for context continuity")
    print("="*70 + "\n")
    
    while True:
        try:
            # Display active session info
            session_info = orchestrator.get_session_info()
            if session_info['active']:
                print(f"\n📝 Active Session: {session_info['name']} | ID: {session_info['id']} | Interactions: {session_info['interactions']}")
            else:
                print("\n🆕 No active session - will create on first interaction")
            
            print("\n📋 Available Flows:")
            print("  1. Report Analysis   - Analyze clinical report FILES (.txt, .doc)")
            print("  2. Query             - Ask medical questions (raw text)")
            print("  3. Image Analysis    - Analyze medical images (.jpg, .png, .dcm)")
            print("  4. Multimodal        - Smart routing based on input type")
            print("\n🔄 Session Management:")
            print("  6. New Session       - Start fresh session (for new patient/report)")
            print("  7. Reset Session     - Clear current session history")
            print("\n  5. Exit              - Exit application")
            
            command = input("\n👉 Select option (1-7): ").strip()
            
            if command == "5" or command.lower() == "exit":
                print("\n✓ Goodbye!")
                break
            
            elif command == "6":
                print("\n--- New Session ---")
                old_session = orchestrator.get_session_info()
                orchestrator.reset_session()
                orchestrator.get_session_id()  # Create new session immediately
                new_session = orchestrator.get_session_info()
                print(f"✓ Created new session: {new_session['name']}")
                if old_session['active']:
                    print(f"  (Previous: {old_session['name']} with {old_session['interactions']} interactions)")
                print("\n📌 Ready for fresh patient/report analysis with clean context")
            
            elif command == "7":
                print("\n--- Reset Session ---")
                session_info = orchestrator.get_session_info()
                if session_info['active']:
                    confirm = input(f"Clear session '{session_info['name']}' with {session_info['interactions']} interactions? (y/n): ").strip().lower()
                    if confirm == 'y':
                        orchestrator.reset_session()
                        print("✓ Session history cleared - context reset")
                    else:
                        print("  Cancelled - session preserved")
                else:
                    print("⚠️  No active session to reset")
            
            elif command == "1":
                print("\n--- Report Analysis Flow ---")
                file_path = input("Enter report file path: ").strip()
                result = orchestrator.analyze_report_flow(file_path)
                
                if "error" in result:
                    print(f"\n❌ Error: {result['error']}")
                elif result.get('parsed', True):
                    print("\n=== 📊 Parsed Clinical Analysis ===")
                    print(f"\n📋 Summary:\n{result['summary']}")
                    print(f"\n🔍 Chief Complaints: {', '.join(result['chief_complaints']) if result['chief_complaints'] else 'None'}")
                    print(f"\n🩺 Symptoms: {', '.join(result['symptoms']) if result['symptoms'] else 'None'}")
                    print(f"\n📜 Medical History: {', '.join(result['medical_history']) if result['medical_history'] else 'None'}")
                    print(f"\n💊 Medications: {', '.join(result['medications']) if result['medications'] else 'None'}")
                    print(f"\n🔬 Lab Findings: {', '.join(result['lab_findings']) if result['lab_findings'] else 'None'}")
                    
                    # Show raw LLM output for transparency
                    if result.get('raw_response'):
                        print("\n\n=== 🤖 Raw LLM Output ===")
                        print("─" * 80)
                        print(result['raw_response'])
                        print("─" * 80)
                else:
                    print("\n⚠️  Could not parse structured output. Showing raw response:\n")
                    print("─" * 80)
                    print(result.get('raw_response', 'No response'))
                    print("─" * 80)
            
            elif command == "2":
                print("\n--- Query Flow ---")
                query = input("Enter your medical question: ").strip()
                result = orchestrator.answer_query_flow(query)
                
                print("\n=== 📊 Query Results ===")
                print(f"\n💡 Answer:\n{result['answer']}")
                print(f"\n🎯 Confidence: {result['confidence']:.2f}")
                if result['references']:
                    print(f"\n📚 References: {', '.join(result['references'][:3])}")
            
            elif command == "3":
                print("\n--- Image Analysis Flow ---")
                image_path = input("Enter image path: ").strip()
                modality = input("Enter modality (xray/mri/ct) [xray]: ").strip() or "xray"
                body_part = input("Enter body part (optional): ").strip() or None
                
                result = orchestrator.analyze_image_flow(image_path, modality, body_part)
                
                print("\n=== 📊 Parsed Image Analysis ===")
                print(f"\n👁️  Observations ({len(result['observations'])}):")
                for i, obs in enumerate(result['observations'][:5], 1):
                    print(f"  {i}. {obs}")
                
                if result['potential_findings']:
                    print(f"\n🔍 Potential Findings ({len(result['potential_findings'])}):")
                    for i, finding in enumerate(result['potential_findings'][:3], 1):
                        print(f"  {i}. {finding}")
                
                print(f"\n📊 Confidence: {result['confidence_score']}")
                
                # Show raw LLM output for transparency
                if result.get('raw_response'):
                    print("\n\n=== 🤖 Raw LLM Output ===")
                    print("─" * 80)
                    print(result['raw_response'])
                    print("─" * 80)
            
            elif command == "4":
                print("\n--- Multimodal Fusion Analysis ---")
                print("Supports: Text+Image combinations for integrated assessment")
                
                # Collect inputs
                text_file = input("Enter report/text file path (or press Enter to skip): ").strip()
                image_file = input("Enter image file path (or press Enter to skip): ").strip()
                
                if not text_file and not image_file:
                    print("⚠️  At least one input (text or image) required")
                    continue
                
                # If only one input, route to specific flow
                if text_file and not image_file:
                    result = orchestrator.multimodal_analysis_flow(text_file, "xray", None)
                elif image_file and not text_file:
                    modality = input("Enter modality (xray/mri/ct) [xray]: ").strip() or "xray"
                    body_part = input("Enter body part (optional): ").strip() or None
                    result = orchestrator.multimodal_analysis_flow(image_file, modality, body_part)
                else:
                    # Both inputs - true multimodal fusion
                    modality = input("Enter image modality (xray/mri/ct) [xray]: ").strip() or "xray"
                    body_part = input("Enter body part (optional): ").strip() or None
                    result = orchestrator.multimodal_analysis_flow(
                        user_input=None,  # Not used for fusion
                        modality=modality,
                        body_part=body_part,
                        text_file=text_file,
                        image_file=image_file
                    )
                
                print("\n=== 📊 Multimodal Analysis Results ===")
                
                # Display routed flow results
                if result.get('flow') == 'query':
                    print(f"\n💡 Answer:\n{result['answer']}")
                    print(f"\n🎯 Confidence: {result['confidence']:.2f}")
                elif result.get('flow') == 'image_analysis':
                    print(f"\n👁️  Image Observations: {len(result['observations'])}")
                    for i, obs in enumerate(result['observations'][:3], 1):
                        print(f"  {i}. {obs}")
                elif result.get('flow') == 'report_analysis':
                    print(f"\n📋 Report Summary:\n{result.get('summary', 'N/A')}")
                elif result.get('flow') == 'multimodal_fusion':
                    # Integrated multimodal results
                    print("\n🔬 Integrated Clinical Assessment:")
                    print(f"{result.get('integrated_assessment', 'N/A')}")
                    
                    if result.get('differential_diagnosis'):
                        print(f"\n🩺 Differential Diagnosis:")
                        for i, dx in enumerate(result['differential_diagnosis'][:3], 1):
                            print(f"  {i}. {dx}")
                    
                    if result.get('recommended_workup'):
                        print(f"\n🔍 Recommended Workup:")
                        for i, workup in enumerate(result['recommended_workup'][:3], 1):
                            print(f"  {i}. {workup}")
                    
                    print(f"\n📊 Confidence: {result.get('confidence_level', 'N/A')}")
                    
                    # Show raw LLM output
                    if result.get('raw_response'):
                        print("\n\n=== 🤖 Raw LLM Output ===")
                        print("─" * 80)
                        print(result['raw_response'])
                        print("─" * 80)
                
                if 'error' in result:
                    print(f"\n❌ Error: {result['error']}")
            
            else:
                print("⚠️  Invalid command. Please select 1-7.")
        
        except KeyboardInterrupt:
            print("\n\n✓ Interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in CLI: {str(e)}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
    
    # Cleanup resources before exit
    print("\n🧹 Cleaning up resources...")
    orchestrator.cleanup()
    print("✓ Cleanup complete. Goodbye!\n")


if __name__ == "__main__":
    import sys
    
    # CLI-only healthcare interface
    if len(sys.argv) > 1:
        print("⚠️  Healthcare Assistant operates in CLI mode only")
        print("Usage: python main.py")
        sys.exit(1)
    
    try:
        run_cli_mode()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
