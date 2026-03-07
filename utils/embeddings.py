import numpy as np
from typing import List
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Improved embedding generator using TF-IDF approach"""
    
    def __init__(self, api_key: str = None, model: str = "simple"):
        self.model_name = "tfidf-based"
        self.embedding_dim = 384
        logger.info("✅ Initialized TF-IDF embedding generator")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text.split())
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text into smaller pieces"""
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        text = text.lower()
        # Extract words and common bigrams
        words = re.findall(r'\b\w+\b', text)
        
        # Add important bigrams for better matching
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
        
        return words + bigrams
    
    def _create_vocabulary_vector(self, tokens: List[str]) -> np.ndarray:
        """Create a sparse vector representation"""
        # Count term frequencies
        term_freq = Counter(tokens)
        
        # Create fixed-size vector
        vector = np.zeros(self.embedding_dim)
        
        # Use multiple hash functions for better distribution
        for term, freq in term_freq.items():
            # Use term as bytes for hashing
            term_bytes = term.encode('utf-8')
            
            # Create multiple hash values
            for i in range(3):  # Use 3 hash functions
                hash_val = hash(term_bytes + bytes([i]))
                pos = abs(hash_val) % self.embedding_dim
                vector[pos] += freq * (1.0 / (1 + i))  # Weighted by hash function
        
        # Apply sublinear TF scaling (log normalization)
        vector = np.log1p(vector)
        
        # L2 normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided")
                return [0.0] * self.embedding_dim
            
            # Tokenize
            tokens = self._tokenize(text)
            
            # Create vector
            vector = self._create_vocabulary_vector(tokens)
            
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            logger.error(f"Text: {text[:100]}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = []
            for i, text in enumerate(texts):
                logger.debug(f"Generating embedding {i+1}/{len(texts)}")
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'type': 'tfidf-hash-hybrid'
        }