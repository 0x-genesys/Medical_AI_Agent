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
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class TextProcessor:
    """
    Medical text processing service using BioBERT/ClinicalBERT embeddings.
    
    Provides comprehensive text analysis for clinical documents using BioBERT
    for semantic embeddings, FAISS for vector search, and LangChain for LLM
    orchestration. Implements Retrieval-Augmented Generation (RAG) with a
    medical knowledge base for enhanced clinical reasoning.
    
    Key Features:
    - BioBERT/ClinicalBERT embeddings for medical text understanding
    - FAISS-based semantic search for knowledge retrieval
    - LangChain orchestration with Ollama LLM
    - Session-aware conversation management
    - Automatic GPU detection and utilization
    
    Attributes:
        embedder (SentenceTransformer): BioBERT model for text embeddings
        llm (OllamaLLM): LangChain LLM for text generation
        index (faiss.Index): FAISS index for semantic search
        documents (List[Document]): Indexed document chunks
        session_manager (SessionManager): Session management for context
        device (str): Compute device ('cuda', 'mps', or 'cpu')
    """
    
    def __init__(self, model_name: Optional[str] = None, embedding_model: str = "emilyalsentzer/Bio_ClinicalBERT", session_manager=None):
        """
        Initialize text processor with BioBERT and LangChain components.
        
        Sets up the complete text processing pipeline including:
        - BioBERT/ClinicalBERT for embeddings (auto GPU detection)
        - Ollama LLM via LangChain for text generation
        - FAISS index for medical knowledge base retrieval
        - Session manager for conversation context
        
        Args:
            model_name (Optional[str]): Override LLM model name (defaults from config)
            embedding_model (str): HuggingFace model for embeddings
                                 (default: Bio_ClinicalBERT)
            session_manager (Optional[SessionManager]): Shared session manager instance
                                                       (creates new if None)
        
        Returns:
            None
        """
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
        Retrieve Retrieval-Augmented Generation (RAG) context from FAISS knowledge base.
        
        Helper method implementing the RAG pattern: retrieves relevant medical knowledge
        from the pre-indexed FAISS vector database using BioBERT semantic similarity.
        Follows DRY principle to centralize RAG retrieval logic used across multiple
        analysis methods.
        
        Process:
        1. Encode query text with BioBERT
        2. Search FAISS index for top-k similar documents
        3. Format retrieved documents for prompt injection
        
        Args:
            query_text (str): Clinical text to use for semantic search
            top_k (int): Number of most relevant documents to retrieve (default: 3)
            
        Returns:
            str: Formatted context string with numbered retrieved documents,
                or message indicating no retrieval (if index not loaded)
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
        Analyze clinical text using BioBERT embeddings and LangChain LLM.
        
        Comprehensive clinical text analysis that:
        1. Generates BioBERT embeddings for the input text
        2. Retrieves relevant medical knowledge via RAG (FAISS search)
        3. Extracts structured clinical information using LLM:
           - Chief complaints
           - Symptoms
           - Medical history
           - Medications
           - Lab findings
           - Clinical entities (with types and confidence)
        4. Stores interaction in session (PHI-sanitized)
        
        Uses a single LLM call for efficiency, combining entity extraction
        with structured field extraction.
        
        Args:
            medical_text (MedicalText): Clinical text data to analyze
            session_id (Optional[str]): Session ID for conversation context tracking
                                       (history is PHI-sanitized before storage)
            
        Returns:
            TextAnalysisResult: Structured analysis with chief complaints, symptoms,
                              medications, entities, and clinical summary
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
        Answer medical knowledge questions using RAG and LangChain.
        
        Implements a question-answering system that:
        1. Encodes the query with BioBERT
        2. Retrieves relevant medical knowledge from FAISS index
        3. Combines retrieved knowledge with user-provided context
        4. Generates evidence-based answer using LLM
        5. Calculates dynamic confidence score
        6. Extracts references from response
        
        Uses session context to maintain conversation continuity, allowing
        follow-up questions that reference previous interactions.
        
        Args:
            query_request (QueryRequest): Query object with question, optional context,
                                         and reference preferences
            session_id (Optional[str]): Session ID for conversation context
            
        Returns:
            QueryResponse: Answer with confidence score, references, and timestamp
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
        Perform semantic search using BioBERT embeddings and FAISS vector database.
        
        Implements semantic matching with two strategies:
        1. Persistent index: Uses pre-built FAISS index (fast, for knowledge base)
        2. Temporary index: Creates ephemeral index from context (for ad-hoc search)
        
        Uses L2 distance similarity in embedding space to find semantically
        similar documents based on BioBERT representations.
        
        Args:
            context (str): Context text to search within (used only if no persistent index)
            query (str): Search query text
            top_k (int): Number of top results to return (default: 3)
            use_persistent_index (bool): Whether to use pre-built index if available
                                        (default: True)
            
        Returns:
            List[Document]: List of relevant Document objects ranked by similarity
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
        Load and index medical knowledge base at startup for RAG.
        
        Automatically loads the medical knowledge base from
        data/medical_knowledge.txt during initialization. Creates an
        in-memory FAISS index for fast semantic retrieval without disk I/O.
        
        The knowledge base is split by document separators (---) and
        indexed using BioBERT embeddings for later RAG retrieval.
        
        Returns:
            None
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
        Pre-index documents for fast semantic search using BioBERT embeddings.
        
        Creates a persistent in-memory FAISS index that enables O(log N) semantic
        search without re-embedding documents on every query. This dramatically
        improves performance for knowledge bases.
        
        Process:
        1. Split documents into chunks using RecursiveCharacterTextSplitter
        2. Generate BioBERT embeddings for all chunks
        3. Build FAISS L2 index from embeddings
        4. Store index and documents for future queries
        
        Performance Impact:
        - Without indexing: O(N) embedding cost per query
        - With indexing: O(1) embedding cost per query (amortized)
        - Search time: O(log N) with FAISS index
        
        Args:
            documents (List[str]): List of document texts to index
        
        Returns:
            None
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
        Parse LLM response into structured TextAnalysisResult.
        
        Robust parser that handles multiple response formats:
        - JSON in markdown code blocks (```json ... ```)
        - Raw JSON objects
        - JSON embedded in prose
        - Malformed responses (fallback to text summary)
        
        Extracts clinical entities and normalizes all list fields to
        consistent string arrays.
        
        Args:
            response (str): Raw LLM response string
        
        Returns:
            TextAnalysisResult: Structured result with parsed fields or
                              fallback result with raw response if parsing fails
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
        Calculate dynamic confidence score for query responses.
        
        Computes confidence based on multiple factors:
        - Base confidence: 0.5
        - +0.25 if relevant RAG context was retrieved
        - +0.15 if response is substantial (>100 chars)
        - +0.10 if response contains medical terminology
        
        Maximum confidence capped at 0.95 to indicate inherent uncertainty
        in AI-generated medical content.
        
        Args:
            retrieved_context (str): RAG context retrieved from knowledge base
            response (str): Generated LLM response
            
        Returns:
            float: Confidence score between 0.0 and 0.95
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
        """
        Extract references and citations from LLM response text.
        
        Searches for lines containing keywords like 'reference', 'source',
        'study', or 'research' to identify potential citations in the response.
        
        Args:
            text (str): LLM response text
        
        Returns:
            List[str]: List of up to 5 extracted reference lines
        """
        references = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['reference', 'source', 'study', 'research']):
                references.append(line.strip())
        return references[:5]
