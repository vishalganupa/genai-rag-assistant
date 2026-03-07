from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class RAGSystem:
    """Simple RAG system using template-based responses"""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "simple",
        temperature: float = 0.7,
        max_tokens: int = 300
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = "template-based"
        
        logger.info("✅ Initialized simple RAG system (template-based responses)")
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """Generate response using retrieved context"""
        
        if conversation_history is None:
            conversation_history = []
        
        try:
            if not retrieved_docs:
                return {
                    'reply': self.generate_fallback_response(threshold_not_met=True),
                    'tokens_used': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'model': self.model_name
                }
            
            # Extract content from top retrieved document
            top_doc = retrieved_docs[0]
            context = top_doc['document']
            
            # Simple extraction: get first few sentences as answer
            sentences = re.split(r'(?<=[.!?])\s+', context)
            
            # Build a natural response
            if len(sentences) >= 3:
                answer = ' '.join(sentences[:3])
            else:
                answer = context[:500]  # First 500 chars
            
            # Add helpful framing
            response = self._format_response(answer, query)
            
            # Calculate tokens
            tokens = len(response.split()) + len(query.split())
            
            result = {
                'reply': response,
                'tokens_used': tokens,
                'prompt_tokens': len(query.split()),
                'completion_tokens': len(response.split()),
                'model': self.model_name
            }
            
            logger.info("✅ Response generated from retrieved context")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'reply': "I apologize, but I encountered an error. Please try again.",
                'error': str(e),
                'tokens_used': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'model': self.model_name
            }
    
    def _format_response(self, answer: str, query: str) -> str:
        """Format the response to be more natural"""
        
        # Clean up the answer
        answer = answer.strip()
        
        # Make sure it ends with punctuation
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def generate_fallback_response(self, threshold_not_met: bool = False) -> str:
        """Generate fallback response"""
        if threshold_not_met:
            return """I apologize, but I couldn't find relevant information in our documentation to answer your question confidently.

Could you please:
1. Rephrase your question, or
2. Contact our support team at support@example.com for personalized assistance

They'll be happy to help!"""
        else:
            return "I apologize, but I'm having trouble accessing our documentation right now. Please try again."