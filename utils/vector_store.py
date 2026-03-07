import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """In-memory vector storage with similarity search"""
    
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadata = []
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict]
    ):
        """
        Add documents with their embeddings to the store
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dicts for each document
        """
        if not (len(documents) == len(embeddings) == len(metadata)):
            raise ValueError("Documents, embeddings, and metadata must have same length")
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for most similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
        
        Returns:
            List of dicts containing document, score, and metadata
        """
        if not self.embeddings:
            return []
        
        # Convert to numpy arrays
        query_vector = np.array(query_embedding).reshape(1, -1)
        doc_vectors = np.array(self.embeddings)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # Apply threshold
            if score >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'score': score,
                    'metadata': self.metadata[idx]
                })
        
        logger.info(f"Similarity search returned {len(results)} results above threshold {threshold}")
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
        }
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")