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
    Manages conversation sessions across all flows (text, image, query).
    
    Provides unified session management for maintaining conversation context
    across different analysis flows. Uses LangChain's ConversationBufferMemory
    to preserve full conversation history without premature summarization.
    
    The session manager ensures:
    - Cross-flow context continuity (queries can reference previous reports)
    - Privacy-compliant storage (PHI sanitized before storage)
    - Multi-session support (multiple concurrent patient contexts)
    
    Attributes:
        sessions (Dict[str, ConversationBufferMemory]): Map of session IDs to memory buffers
    """
    
    def __init__(self):
        """
        Initialize session manager with empty session dictionary.
        
        Creates the internal sessions dictionary for storing conversation
        buffers keyed by session ID.
        
        Returns:
            None
        """
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        logger.info("✓ SessionManager initialized with full conversation buffer (no summarization)")
    
    def get_or_create_session(self, session_id: str) -> ConversationBufferMemory:
        """
        Get existing session memory or create a new one if it doesn't exist.
        
        Retrieves the conversation buffer for the specified session ID.
        If no session exists with that ID, creates a new ConversationBufferMemory
        instance configured with message return format.
        
        Args:
            session_id (str): Unique session identifier (typically UUID)
            
        Returns:
            ConversationBufferMemory: LangChain memory buffer for this session
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
        Add a user-AI interaction to session memory with automatic PHI sanitization.
        
        Stores the conversation turn in the session's memory after removing PHI
        to ensure HIPAA compliance. Tags the input with the flow type (e.g., [REPORT],
        [QUERY]) to provide context for future interactions.
        
        Args:
            session_id (str): Session identifier
            user_input (str): User's input text (will be sanitized for PHI)
            ai_response (str): AI's response text (will be sanitized for PHI)
            flow_type (str): Type of flow that generated this interaction
                           (e.g., 'report', 'query', 'image')
        
        Returns:
            None
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
        Get formatted conversation context for a session.
        
        Retrieves the conversation history from the session's memory and
        formats it as a readable string. Returns the last 5 messages to
        keep context relevant without overwhelming the prompt.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            str: Formatted conversation history with role labels (User/Assistant)
                Returns "No previous context" if session doesn't exist or is empty
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
        Clear and delete a session's memory.
        
        Removes the session from the internal sessions dictionary,
        freeing memory and resetting context for that session ID.
        
        Args:
            session_id (str): Session identifier to clear
        
        Returns:
            None
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"✓ Cleared session {session_id[:8]}...")
    
    def get_session_summary(self, session_id: str) -> str:
        """
        Get a human-readable summary of the session.
        
        Provides basic information about the session including its ID
        and the amount of context stored.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            str: Summary string with session info or "No active session"
        """
        if session_id not in self.sessions:
            return "No active session."
        
        context = self.get_context(session_id)
        memory = self.sessions[session_id]
        
        return f"Session {session_id[:8]}: {len(context)} chars of context"
