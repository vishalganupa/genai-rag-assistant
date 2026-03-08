from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class RAGSystem:
    """Simple RAG system with keyword matching fallback"""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "simple",
        temperature: float = 0.7,
        max_tokens: int = 300
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = "template-based-with-fallback"
        
        logger.info("✅ Initialized simple RAG system")
    
    def _keyword_match(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Fallback keyword matching when embeddings don't work"""
        query_lower = query.lower()
        
        # Keyword patterns
        patterns = {
            'password': ['password', 'reset', 'forgot', 'credentials', 'login'],
            'payment': ['payment', 'pay', 'billing', 'invoice', 'credit', 'card'],
            'subscription': ['subscription', 'plan', 'pricing', 'tier', 'cost', 'price'],
            'support': ['support', 'help', 'contact', 'assistance', 'customer'],
            'account': ['account', 'signup', 'register', 'create', 'profile'],
            'security': ['security', 'privacy', 'data', 'encryption', 'safe'],
            'api': ['api', 'developer', 'integration', 'webhook'],
            'mobile': ['mobile', 'app', 'ios', 'android', 'phone'],
        }
        
        # Score documents by keyword matches
        scored_docs = []
        for doc in documents:
            score = 0
            doc_lower = doc['document'].lower()
            title_lower = doc['metadata']['title'].lower()
            
            # Check query keywords
            for category, keywords in patterns.items():
                if any(kw in query_lower for kw in keywords):
                    # Boost if keywords in title
                    if any(kw in title_lower for kw in keywords):
                        score += 0.5
                    # Check if keywords in content
                    if any(kw in doc_lower for kw in keywords):
                        score += 0.3
            
            if score > 0:
                scored_docs.append({
                    'document': doc['document'],
                    'score': min(score, 0.99),  # Cap at 99%
                    'metadata': doc['metadata']
                })
        
        # Sort by score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:3]
    
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """Generate response from retrieved documents"""
        
        if conversation_history is None:
            conversation_history = []
        
        try:
            # If no documents retrieved by embeddings, try keyword matching
            if not retrieved_docs:
                logger.warning("No docs from embeddings, trying keyword match")
                # We don't have access to all docs here, so just return fallback
                return {
                    'reply': self.generate_fallback_response(threshold_not_met=True),
                    'tokens_used': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'model': self.model_name
                }
            
            # Extract content from top document
            top_doc = retrieved_docs[0]
            context = top_doc['document']
            
            # Extract first 2-3 sentences as answer
            sentences = re.split(r'(?<=[.!?])\s+', context)
            
            if len(sentences) >= 3:
                answer = ' '.join(sentences[:3])
            elif len(sentences) >= 2:
                answer = ' '.join(sentences[:2])
            else:
                answer = context[:400]  # First 400 chars
            
            # Clean up
            answer = answer.strip()
            if answer and answer[-1] not in '.!?':
                answer += '.'
            
            # Calculate tokens
            tokens = len(answer.split()) + len(query.split())
            
            result = {
                'reply': answer,
                'tokens_used': tokens,
                'prompt_tokens': len(query.split()),
                'completion_tokens': len(answer.split()),
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