"""
Unified Session Manager for cross-flow context continuity
Maintains conversation history across report, query, and image flows
"""
from typing import Dict, Optional
from langchain_classic.memory import ConversationBufferMemory
from privacy_utils import sanitize_phi
from logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages conversation sessions across all flows (text, image, query)
    Uses ConversationBufferMemory to preserve full conversation history
    (Llama3 has 8K+ token context, no need for premature summarization)
    """
    
    def __init__(self):
        """
        Initialize session manager with full conversation buffer
        """
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        logger.info("✓ SessionManager initialized with full conversation buffer (no summarization)")
    
    def get_or_create_session(self, session_id: str) -> ConversationBufferMemory:
        """
        Get existing session or create new one
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ConversationBufferMemory instance with full history
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            logger.info(f"✓ Created new session memory: {session_id[:8]}...")
        
        return self.sessions[session_id]
    
    def add_interaction(
        self,
        session_id: str,
        user_input: str,
        ai_response: str,
        flow_type: str = "general"
    ):
        """
        Add user-AI interaction to session memory with PHI sanitization
        
        Args:
            session_id: Session identifier
            user_input: User's input (will be sanitized)
            ai_response: AI's response (will be sanitized)
            flow_type: Type of flow (report/query/image)
        """
        memory = self.get_or_create_session(session_id)
        
        # Sanitize PHI before storing
        sanitized_input = sanitize_phi(user_input)
        sanitized_response = sanitize_phi(ai_response)
        
        # Add flow type marker for context
        tagged_input = f"[{flow_type.upper()}] {sanitized_input}"
        
        # Let LangChain's ConversationSummaryBufferMemory handle compression
        memory.save_context(
            {"input": tagged_input},
            {"output": sanitized_response}
        )
        
        logger.info(f"✓ Saved {flow_type} interaction to session {session_id[:8]}...")
    
    def get_context(self, session_id: str) -> str:
        """
        Get conversation context for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation history string
        """
        if session_id not in self.sessions:
            return "No previous context in this session."
        
        memory = self.sessions[session_id]
        
        try:
            # Load memory variables
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            if not chat_history:
                return "No previous context in this session."
            
            # Format as context string
            context_parts = []
            for msg in chat_history[-5:]:  # Last 5 messages
                if hasattr(msg, 'type'):
                    role = "User" if msg.type == "human" else "Assistant"
                    context_parts.append(f"{role}: {msg.content}")
            
            context = "\n".join(context_parts)
            logger.info(f"Retrieved {len(chat_history)} messages from session {session_id[:8]}...")
            
            return context
        
        except Exception as e:
            logger.warning(f"Could not retrieve context: {e}")
            return "Could not retrieve previous context."
    
    def clear_session(self, session_id: str):
        """
        Clear a session's memory
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"✓ Cleared session {session_id[:8]}...")
    
    def get_session_summary(self, session_id: str) -> str:
        """
        Get a summary of the session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Human-readable summary
        """
        if session_id not in self.sessions:
            return "No active session."
        
        context = self.get_context(session_id)
        memory = self.sessions[session_id]
        
        return f"Session {session_id[:8]}: {len(context)} chars of context"
