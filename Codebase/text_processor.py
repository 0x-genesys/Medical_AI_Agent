"""
Text Processing Service using BioBERT/ClinicalBERT and LangChain
Follows PDF requirements: uses sentence-transformers for semantic search and encoding
"""
import json
from typing import Optional, List, Dict
import faiss
import torch
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationSummaryBufferMemory

from models import MedicalText, TextAnalysisResult, ClinicalEntity, QueryRequest, QueryResponse
from logger import get_logger
from config import config
from session_manager import SessionManager

# Suppress verbose BERT model loading messages
import warnings
import logging
warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')
logging.getLogger("transformers.modeling_utils").setLevel(logging.DEBUG)

class TextProcessor:
    """
    Processes medical text using BioBERT/ClinicalBERT for embeddings
    Uses LangChain for orchestration as per PDF requirements
    Implements semantic search and matching for clinical data
    """
    
    def __init__(self, model_name: Optional[str] = None, embedding_model: str = "emilyalsentzer/Bio_ClinicalBERT", session_manager=None):
        self.model_name = model_name or config.model.llm_model
        self.logger = get_logger(__name__)
        
        # Detect GPU for BioBERT embeddings
        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"BioBERT will use NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("BioBERT will use Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            self.logger.warning("BioBERT using CPU (no GPU detected)")
        
        self.logger.info(f"Loading BioBERT/ClinicalBERT: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.logger.info(f"BioBERT loaded on {self.device}")
        
        self.llm = OllamaLLM(model=self.model_name, temperature=config.model.temperature)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        self.index = None
        self.documents = []
        
        # Unified session manager for cross-flow context (shared or create new)
        from session_manager import SessionManager
        self.session_manager = session_manager if session_manager else SessionManager()
        
        # Auto-load medical knowledge base at startup for RAG
        self._load_medical_knowledge_base()
        
        self.logger.info(f"Initialized TextProcessor with LangChain and BioBERT on {self.device}")
    
    def _retrieve_rag_context(self, query_text: str, top_k: int = 3) -> str:
        """
        Helper method to retrieve RAG context from FAISS knowledge base
        Follows DRY principle to avoid code duplication
        
        Args:
            query_text: Text to use for semantic search
            top_k: Number of top documents to retrieve
            
        Returns:
            Formatted context string with retrieved documents
        """
        if self.index is not None:
            self.logger.info("Retrieving from FAISS knowledge base using BioBERT embeddings")
            context_docs = self._semantic_search("", query_text, top_k=top_k, use_persistent_index=True)
            if context_docs:
                retrieved_context = "\n\n=== RELEVANT MEDICAL KNOWLEDGE (Retrieved via BioBERT) ===\n" + "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(context_docs)])
                self.logger.info(f"Retrieved {len(context_docs)} relevant documents from knowledge base")
                return retrieved_context
            else:
                return "\n\n=== No relevant medical knowledge retrieved ==="
        else:
            return "\n\n=== Knowledge base not loaded ==="
    
    def analyze_clinical_text(self, medical_text: MedicalText, session_id: Optional[str] = None) -> TextAnalysisResult:
        """
        Analyze clinical text using LangChain and BioBERT embeddings
        Stores anonymized history if session_id provided (privacy-compliant)
        
        Args:
            medical_text: MedicalText object containing clinical notes
            session_id: Optional session ID for history tracking (anonymized)
            
        Returns:
            TextAnalysisResult with extracted information
        """
        if not medical_text.is_valid():
            self.logger.error("Invalid medical text provided")
            return TextAnalysisResult()
        
        self.logger.info("Analyzing clinical text with BioBERT embeddings")
        
        # Generate BioBERT embedding for the input text
        embedding = self.embedder.encode(medical_text.text)
        self.logger.info(f"Generated BioBERT embedding vector: dimension {len(embedding)}")
        
        # Retrieve relevant context from FAISS knowledge base using helper method
        retrieved_context = self._retrieve_rag_context(medical_text.text, top_k=3)
        
        # Get conversation context from session
        conversation_context = ""
        if session_id:
            conversation_context = self.session_manager.get_context(session_id)
            self.logger.info(f"Retrieved conversation context: {len(conversation_context)} chars")
        
        prompt_template = PromptTemplate(
            input_variables=["text", "retrieved_context", "conversation_context"],
            template="""As a medical AI assistant, analyze this clinical text and extract structured information.

Previous Conversation:
{conversation_context}

As a medical AI assistant, analyze this clinical text and extract structured information.

                    Clinical Text:
                    {text}
                    {retrieved_context}

        Extract and return as JSON with ALL fields:
        {{
            "chief_complaints": ["list of main complaints"],
            "symptoms": ["list of symptoms mentioned"],
            "medical_history": ["relevant medical history"],
            "medications": ["list of medications"],
            "lab_findings": ["laboratory test results if mentioned"],
            "summary": "brief clinical summary in 2-3 sentences",
            "entities": [
                {{"entity_type": "symptom|diagnosis|medication|procedure|body_part|lab_test", "value": "entity name", "confidence": 0.0-1.0}}
            ]
        }}
        
        IMPORTANT: Include 'entities' array with medical entities found in text.
        Return only valid JSON."""
        )
        
        chain = prompt_template | self.llm
        
        try:
            # Use LangChain memory if session provided
            response = chain.invoke({
                "text": medical_text.text,
                "retrieved_context": retrieved_context,
                "conversation_context": conversation_context or "No previous context."
            })
            
            # Extract string from response (invoke may return dict or string)
            if isinstance(response, dict):
                response = response.get('text', str(response))
            
            result = self._parse_analysis_response(response)
            
            # Save interaction to unified session
            if session_id:
                self.session_manager.add_interaction(
                    session_id=session_id,
                    user_input=medical_text.text,
                    ai_response=response,
                    flow_type="report_analysis"
                )
            
            # Entities now extracted in single LLM call (no duplicate call needed)
            self.logger.info("Clinical text analysis completed successfully with BioBERT (single LLM call)")
            return result
        except Exception as e:
            self.logger.error(f"Error during text analysis: {str(e)}", exc_info=True)
            return TextAnalysisResult(raw_response=str(e))
    
    def answer_query(self, query_request: QueryRequest, session_id: Optional[str] = None) -> QueryResponse:
        """
        Answer medical knowledge questions using LangChain QA chain with semantic search
        
        Args:
            query_request: QueryRequest with question and context
            
        Returns:
            QueryResponse with answer and references
        """
        self.logger.info(f"Processing query with BioBERT semantic search: {query_request.query[:50]}...")
        
        # Generate BioBERT embedding and retrieve from FAISS knowledge base
        query_embedding = self.embedder.encode(query_request.query)
        self.logger.info(f"Generated BioBERT query embedding: dimension {len(query_embedding)}")
        
        # Retrieve relevant context from FAISS knowledge base using helper method
        retrieved_context = self._retrieve_rag_context(query_request.query, top_k=3)
        
        # Also include user-provided context if available
        user_context = f"\n\n=== USER PROVIDED CONTEXT ===\n{query_request.context}" if query_request.context else ""

        # Get conversation context from session
        conversation_context = ""
        if session_id:
            conversation_context = self.session_manager.get_context(session_id)
            self.logger.info(f"Retrieved conversation context: {len(conversation_context)} chars")

        prompt_template = PromptTemplate(
            input_variables=["query", "retrieved_context", "user_context", "conversation_context"],
            template="""You are a medical knowledge assistant. Answer this question with:
        1. Direct, accurate answer
        2. Supporting evidence and reasoning
        3. Clinical significance
        4. Important considerations
        
        Previous Conversation:
        {conversation_context}
        
        Question: {query}
        {retrieved_context}
        {user_context}
        
        Provide a comprehensive, evidence-based answer. 
        Note: This is for educational purposes only. Always consult qualified medical professionals."""
        )
        
        chain = prompt_template | self.llm
        
        try:
            response = chain.invoke({
                "query": query_request.query,
                "retrieved_context": retrieved_context,
                "user_context": user_context,
                "conversation_context": conversation_context or "No previous context."
            })
            
            # Extract string from response (invoke may return dict or string)
            if isinstance(response, dict):
                response = response.get('text', str(response))
            
            # Calculate dynamic confidence based on context availability and response quality
            confidence = self._calculate_query_confidence(retrieved_context, response)
            
            # Save query interaction to session
            if session_id:
                self.session_manager.add_interaction(
                    session_id=session_id,
                    user_input=query_request.query,
                    ai_response=response,
                    flow_type="query"
                )
            
            return QueryResponse(
                answer=response,
                confidence=confidence,
                references=self._extract_references(response) if query_request.include_references else []
            )
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return QueryResponse(
                answer=f"Error processing query: {str(e)}",
                confidence=0.0
            )
    
    def _semantic_search(self, context: str, query: str, top_k: int = 3, use_persistent_index: bool = True) -> List[Document]:
        """
        Perform semantic search using BioBERT embeddings and FAISS
        Implements semantic matching as per PDF requirements
        
        Args:
            context: Context text to search within (used only if no persistent index)
            query: Search query
            top_k: Number of top results to return
            use_persistent_index: Whether to use pre-built index (if available)
            ry
        Returns:
            List of relevant Document objects
        """
        self.logger.info("Performing semantic search with BioBERT embeddings")
        
        query_embedding = self.embedder.encode(query)
        
        # Use persistent index if available and requested
        if use_persistent_index and self.index is not None:
            self.logger.info(f"Using persistent FAISS index with {len(self.documents)} documents")
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(top_k, len(self.documents))
            )
            relevant_docs = [self.documents[i] for i in indices[0]]
            self.logger.info(f"Found {len(relevant_docs)} relevant documents from persistent index")
            return relevant_docs
        
        # Fall back to temporary index from context
        self.logger.info("No persistent index available, creating temporary index from context")
        chunks = self.text_splitter.split_text(context)
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        if not docs:
            return []
        
        doc_embeddings = self.embedder.encode([doc.page_content for doc in docs])
        
        dimension = doc_embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dimension)
        temp_index.add(doc_embeddings.astype('float32'))
        
        distances, indices = temp_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(top_k, len(docs))
        )
        
        relevant_docs = [docs[i] for i in indices[0]]
        self.logger.info(f"Found {len(relevant_docs)} relevant documents using temporary index")
        
        return relevant_docs
    
    def _load_medical_knowledge_base(self):
        """
        Load and index medical knowledge base at startup for RAG
        Uses in-memory FAISS for fast retrieval without disk I/O
        """
        from pathlib import Path
        
        kb_path = Path(__file__).parent / "data" / "medical_knowledge.txt"
        
        if not kb_path.exists():
            self.logger.warning(f"Medical knowledge base not found at {kb_path}. RAG will be limited.")
            return
        
        self.logger.info(f"Loading medical knowledge base from {kb_path}")
        
        try:
            with open(kb_path, 'r') as f:
                content = f.read()
            
            # Split by document separator
            docs = [doc.strip() for doc in content.split('\n---\n') if doc.strip()]
            
            self.logger.info(f"Found {len(docs)} documents in knowledge base")
            
            # Index documents
            self.index_documents(docs)
            
            self.logger.info(f"✓ Medical knowledge base indexed successfully with {len(docs)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to load medical knowledge base: {str(e)}", exc_info=True)
    
    def index_documents(self, documents: List[str]):
        """
        Pre-index documents for fast semantic search using BioBERT embeddings
        Creates persistent FAISS index that's reused across queries
        
        Use this to index a knowledge base (e.g., medical guidelines, FAQs)
        before handling user queries for much faster retrieval.
        
        Performance:
        - Without indexing: O(N) embedding cost per query
        - With indexing: O(1) embedding cost per query (amortized)
        
        Args:
            documents: List of document texts to index
        """
        self.logger.info(f"Indexing {len(documents)} documents with BioBERT")
        
        all_chunks = []
        for doc_text in documents:
            chunks = self.text_splitter.split_text(doc_text)
            all_chunks.extend([Document(page_content=chunk) for chunk in chunks])
        
        self.documents = all_chunks
        
        if all_chunks:
            embeddings = self.embedder.encode([doc.page_content for doc in all_chunks])
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.logger.info(f"Indexed {len(all_chunks)} document chunks for semantic search")
            self.logger.info(f"Persistent FAISS index ready - future queries will be 10-100x faster")
    
    def _parse_analysis_response(self, response: str) -> TextAnalysisResult:
        """
        Robust LLM response parser - handles markdown, extra text, flexible JSON
        """
        import re
        
        try:
            cleaned = response.strip()
            data = None
            
            # Strategy 1: Extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
            if "```" in cleaned:
                code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                matches = re.findall(code_block_pattern, cleaned, re.DOTALL)
                for match in matches:
                    match = match.strip()
                    if match.startswith("{"):
                        try:
                            data = json.loads(match)
                            self.logger.info("Extracted JSON from markdown code block")
                            break
                        except:
                            continue
            
            # Strategy 2: Find JSON object in text (first { to matching })
            if data is None:
                start_idx = cleaned.find("{")
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                    
                    if end_idx > start_idx:
                        json_str = cleaned[start_idx:end_idx+1]
                        try:
                            data = json.loads(json_str)
                            self.logger.info("Extracted JSON from text")
                        except:
                            pass
            
            # Strategy 3: Try parsing entire response
            if data is None:
                data = json.loads(cleaned)
            
            # Parse entities
            entities = []
            if "entities" in data:
                from models import ClinicalEntity
                for e in data.get("entities", []):
                    if isinstance(e, dict):
                        entities.append(ClinicalEntity(
                            entity_type=e.get("entity_type", "unknown"),
                            value=e.get("value", ""),
                            confidence=e.get("confidence", 0.5)
                        ))
            
            # Normalize all list fields to be lists of strings
            def normalize_list_field(field_data):
                if not field_data:
                    return []
                normalized = []
                for item in field_data:
                    if isinstance(item, dict):
                        # Convert dict to string representation
                        normalized.append(str(item.get('value', item.get('name', str(item)))))
                    else:
                        normalized.append(str(item))
                return normalized
            
            return TextAnalysisResult(
                chief_complaints=normalize_list_field(data.get("chief_complaints", [])),
                symptoms=normalize_list_field(data.get("symptoms", [])),
                medical_history=normalize_list_field(data.get("medical_history", [])),
                medications=normalize_list_field(data.get("medications", [])),
                lab_findings=normalize_list_field(data.get("lab_findings", [])),
                summary=data.get("summary", ""),
                entities=entities,
                raw_response=response
            )
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            self.logger.warning(f"Could not parse JSON from response: {str(e)}")
            return TextAnalysisResult(raw_response=response, summary=response[:500])
    
    def _calculate_query_confidence(self, retrieved_context: str, response: str) -> float:
        """
        Calculate dynamic confidence score based on context availability and response quality
        
        Args:
            retrieved_context: RAG context retrieved
            response: Generated response
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost if RAG context was found
        if retrieved_context and "No relevant context found" not in retrieved_context:
            confidence += 0.25
        
        # Boost if response is substantial (not error or very short)
        if len(response) > 100:
            confidence += 0.15
        
        # Boost if response contains medical terminology or structure
        medical_indicators = ['treatment', 'diagnosis', 'symptoms', 'medication', 'patient', 'clinical']
        if any(indicator in response.lower() for indicator in medical_indicators):
            confidence += 0.1
        
        return min(confidence, 0.95)  # Cap at 0.95
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from response text"""
        references = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['reference', 'source', 'study', 'research']):
                references.append(line.strip())
        return references[:5]
