// Session management
let sessionId = localStorage.getItem('chatSessionId');
if (!sessionId) {
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('chatSessionId', sessionId);
}

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const newChatButton = document.getElementById('new-chat-button');
const clearButton = document.getElementById('clear-button');

// State
let isProcessing = false;

// Event listeners
sendButton.addEventListener('click', sendMessage);
clearButton.addEventListener('click', clearConversation);
newChatButton.addEventListener('click', startNewChat);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Suggestion buttons
document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        userInput.value = btn.textContent;
        userInput.focus();
    });
});

// Functions
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message || isProcessing) return;
    
    // Set processing state
    isProcessing = true;
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.innerHTML = '<span class="spinner"></span> Sending...';
    
    // Display user message
    addMessage(message, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Show typing indicator
    const typingIndicator = showTypingIndicator();
    
    try {
        // Send to backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sessionId: sessionId,
                message: message
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Request failed');
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        // Display assistant response with metadata
        addMessage(
            data.reply, 
            'assistant', 
            {
                sources: data.sources,
                relevanceScores: data.relevanceScores,
                tokensUsed: data.tokensUsed,
                retrievedChunks: data.retrievedChunks
            }
        );
        
    } catch (error) {
        console.error('Error:', error);
        typingIndicator.remove();
        addMessage(
            `I apologize, but I encountered an error: ${error.message}. Please try again.`,
            'assistant',
            { error: true }
        );
    } finally {
        // Reset processing state
        isProcessing = false;
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.innerHTML = '<span>Send</span>';
        userInput.focus();
    }
}

function addMessage(text, type, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const label = type === 'user' ? 'You' : 'Assistant';
    const timestamp = new Date().toLocaleTimeString();
    
    contentDiv.innerHTML = `
        <div class="message-header">
            <strong>${label}</strong>
            <span class="timestamp">${timestamp}</span>
        </div>
        <div class="message-text">${escapeHtml(text)}</div>
    `;
    
    messageDiv.appendChild(contentDiv);
    
    // Add metadata for assistant messages
    if (type === 'assistant' && metadata && !metadata.error) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        
        let metadataHTML = '';
        
        // Sources
        if (metadata.sources && metadata.sources.length > 0) {
            metadataHTML += '<div class="sources"><strong>📚 Sources:</strong> ';
            metadata.sources.forEach((source, idx) => {
                metadataHTML += `<span class="source-tag">${source.title}</span>`;
            });
            metadataHTML += '</div>';
        }
        
        // Relevance scores
        if (metadata.relevanceScores && metadata.relevanceScores.length > 0) {
            metadataHTML += '<div class="relevance-scores"><strong>🎯 Relevance:</strong> ';
            metadata.relevanceScores.forEach(score => {
                const percentage = (score.score * 100).toFixed(1);
                metadataHTML += `<span class="score-tag">${score.title}: ${percentage}%</span>`;
            });
            metadataHTML += '</div>';
        }
        
        // Token usage
        if (metadata.tokensUsed) {
            metadataHTML += `<div class="token-info">🔢 Tokens: ${metadata.tokensUsed} | Chunks: ${metadata.retrievedChunks}</div>`;
        }
        
        metadataDiv.innerHTML = metadataHTML;
        messageDiv.appendChild(metadataDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

async function clearConversation() {
    if (!confirm('Clear this conversation? This action cannot be undone.')) {
        return;
    }
    
    try {
        await fetch('/api/session/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sessionId: sessionId
            })
        });
        
        // Clear UI
        chatMessages.innerHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <div class="message-header">
                        <strong>Assistant</strong>
                    </div>
                    <div class="message-text">
                        Conversation cleared. How can I help you today?
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Error clearing conversation:', error);
        alert('Failed to clear conversation. Please try again.');
    }
}

async function startNewChat() {
    if (!confirm('Start a new chat? Current conversation will be saved but not visible.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/session/new', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        sessionId = data.sessionId;
        localStorage.setItem('chatSessionId', sessionId);
        
        // Clear UI
        chatMessages.innerHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <div class="message-header">
                        <strong>Assistant</strong>
                    </div>
                    <div class="message-text">
                        New conversation started. How can I help you today?
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Error starting new chat:', error);
        alert('Failed to start new chat. Please try again.');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load health check on startup
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('System health:', data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Initialize
window.addEventListener('load', () => {
    userInput.focus();
    checkHealth();
});