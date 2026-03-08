import os

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from typing import Dict
import json
import logging
from datetime import datetime
import uuid

from config import Config
from utils.embeddings import EmbeddingGenerator
from utils.vector_store import VectorStore
from utils.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Validate configuration
try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Initialize components
embedding_generator = EmbeddingGenerator(
    api_key=Config.GEMINI_API_KEY,
    model=Config.EMBEDDING_MODEL
)

vector_store = VectorStore()

rag_system = RAGSystem(
    api_key=Config.GEMINI_API_KEY,
    model=Config.LLM_MODEL,
    temperature=Config.TEMPERATURE,
    max_tokens=Config.MAX_TOKENS
)

# Session storage (in-memory - use Redis/DB for production)
sessions = {}

def expand_query(query: str) -> str:
    """
    Expand query with related terms for better matching
    
    Args:
        query: Original user query
    
    Returns:
        Expanded query string
    """
    query_lower = query.lower()
    
    # Expansion dictionary
    expansions = {
        'subscription': 'subscription plan pricing tier package cost',
        'payment': 'payment billing invoice transaction pay charge',
        'password': 'password reset credentials login authentication access',
        'account': 'account registration signup profile user create',
        'support': 'support help assistance customer service contact',
        'api': 'api developer integration webhook endpoint',
        'security': 'security privacy protection encryption data safe',
        'mobile': 'mobile app application ios android phone',
        'price': 'price pricing cost fee subscription plan',
        'contact': 'contact support help email phone reach',
    }
    
    # Check for matching keywords and expand
    for keyword, expansion in expansions.items():
        if keyword in query_lower:
            expanded = f"{query} {expansion}"
            logger.info(f"Query expanded: '{query}' → '{expanded}'")
            return expanded
    
    return query

def load_and_index_documents():
    """Load documents from JSON and create embeddings"""
    try:
        logger.info("Loading documents from docs.json")
        
        with open(Config.DOCS_PATH, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} documents")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        for doc in documents:
            # Chunk the document
            chunks = embedding_generator.chunk_text(
                doc['content'],
                chunk_size=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP
            )
            
            logger.info(f"Document '{doc['title']}': {len(chunks)} chunks created")
            
            # Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                try:
                    embedding = embedding_generator.generate_embedding(chunk)
                    
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    all_metadata.append({
                        'doc_id': doc['id'],
                        'title': doc['title'],
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
                    
                    logger.info(f"  Generated embedding for chunk {i+1}/{len(chunks)}")
                    
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {i}: {e}")
                    continue
        
        # Add to vector store
        if all_chunks:
            vector_store.add_documents(all_chunks, all_embeddings, all_metadata)
            
            stats = vector_store.get_stats()
            logger.info(f"Indexing complete: {stats}")
            return True
        else:
            logger.error("No chunks were successfully embedded")
            return False
        
    except FileNotFoundError:
        logger.error(f"Documents file not found: {Config.DOCS_PATH}")
        return False
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return False

def get_or_create_session(session_id: str) -> dict:
    """Get existing session or create new one"""
    if session_id not in sessions:
        sessions[session_id] = {
            'id': session_id,
            'created_at': datetime.now().isoformat(),
            'conversation_history': [],
            'total_queries': 0
        }
    return sessions[session_id]

@app.route('/')
def index():
    """Render chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat messages with RAG
    
    Expected JSON:
        {
            "sessionId": "string",
            "message": "string"
        }
    
    Returns JSON:
        {
            "reply": "string",
            "tokensUsed": int,
            "retrievedChunks": int,
            "sources": [...],
            "relevanceScores": [...]
        }
    """
    try:
        # Validate request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        session_id = data.get('sessionId')
        user_message = data.get('message', '').strip()
        
        if not session_id:
            return jsonify({'error': 'sessionId is required'}), 400
        
        if not user_message:
            return jsonify({'error': 'message cannot be empty'}), 400
        
        logger.info(f"Chat request - Session: {session_id}, Message: {user_message[:50]}...")
        
        # Get or create session
        session = get_or_create_session(session_id)
        
        # Expand query for better matching
        expanded_message = expand_query(user_message)
        
        # Generate query embedding
        try:
            query_embedding = embedding_generator.generate_embedding(expanded_message)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return jsonify({
                'error': 'Failed to process query',
                'message': str(e)
            }), 500
        
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(
            query_embedding,
            top_k=Config.TOP_K_CHUNKS,
            threshold=Config.SIMILARITY_THRESHOLD
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents above threshold {Config.SIMILARITY_THRESHOLD}")
        
        # Log similarity scores for debugging
        if retrieved_docs:
            logger.info("Top retrieved documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"  {i}. '{doc['metadata']['title']}' - Score: {doc['score']:.4f}")
        else:
            logger.warning(f"⚠️ No documents found above threshold {Config.SIMILARITY_THRESHOLD}")
            logger.warning(f"Query was: '{user_message}'")
            logger.warning(f"Expanded to: '{expanded_message}'")
        
        # Check if we have relevant documents
        if not retrieved_docs:
            response_text = rag_system.generate_fallback_response(threshold_not_met=True)
            result = {
                'reply': response_text,
                'tokensUsed': 0,
                'retrievedChunks': 0,
                'sources': [],
                'relevanceScores': [],
                'warning': 'No relevant documents found above threshold'
            }
        else:
            # Generate response using RAG
            rag_response = rag_system.generate_response(
                query=user_message,  # Use original query for response generation
                retrieved_docs=retrieved_docs,
                conversation_history=session['conversation_history'][-Config.MAX_CONVERSATION_HISTORY:]
            )
            
            # Update conversation history
            session['conversation_history'].append({
                'user': user_message,
                'assistant': rag_response['reply'],
                'timestamp': datetime.now().isoformat()
            })
            
            session['total_queries'] += 1
            
            # Build response
            result = {
                'reply': rag_response['reply'],
                'tokensUsed': rag_response.get('tokens_used', 0),
                'promptTokens': rag_response.get('prompt_tokens', 0),
                'completionTokens': rag_response.get('completion_tokens', 0),
                'retrievedChunks': len(retrieved_docs),
                'sources': [
                    {
                        'title': doc['metadata']['title'],
                        'chunkIndex': doc['metadata']['chunk_index'],
                        'totalChunks': doc['metadata']['total_chunks']
                    }
                    for doc in retrieved_docs
                ],
                'relevanceScores': [
                    {
                        'title': doc['metadata']['title'],
                        'score': round(doc['score'], 4)
                    }
                    for doc in retrieved_docs
                ]
            }
        
        logger.info(f"Response generated successfully for session {session_id}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear conversation history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        
        if not session_id:
            return jsonify({'error': 'sessionId is required'}), 400
        
        if session_id in sessions:
            sessions[session_id]['conversation_history'] = []
            logger.info(f"Cleared conversation history for session {session_id}")
        
        return jsonify({'status': 'success', 'message': 'Conversation cleared'})
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/session/new', methods=['POST'])
def new_session():
    """Create a new session"""
    try:
        session_id = str(uuid.uuid4())
        session = get_or_create_session(session_id)
        
        return jsonify({
            'sessionId': session_id,
            'createdAt': session['created_at']
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    stats = vector_store.get_stats()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'vectorStore': stats,
        'activeSessions': len(sessions),
        'config': {
            'embeddingModel': Config.EMBEDDING_MODEL,
            'llmModel': Config.LLM_MODEL,
            'similarityThreshold': Config.SIMILARITY_THRESHOLD,
            'topK': Config.TOP_K_CHUNKS
        }
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    total_queries = sum(s['total_queries'] for s in sessions.values())
    
    return jsonify({
        'totalSessions': len(sessions),
        'totalQueries': total_queries,
        'vectorStore': vector_store.get_stats()
    })

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load and index documents on startup
    logger.info("=" * 60)
    logger.info("Starting GenAI RAG Assistant")
    logger.info("Using local embeddings")
    logger.info("=" * 60)
    
    success = load_and_index_documents()
    
    if not success:
        logger.error("Failed to load documents. Exiting.")
        exit(1)
    
    logger.info("=" * 60)
    logger.info("Document indexing complete. Starting Flask server...")
    logger.info("=" * 60)
    
    # Run the application
    # Get port from environment variable (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Always False in production
    )