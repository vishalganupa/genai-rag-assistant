from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class LocalLLM:
    """Local LLM using HuggingFace transformers - No API needed!"""
    
    def __init__(self, model_name: str = "facebook/opt-350m"):
        """
        Initialize local LLM
        
        Model options:
        - facebook/opt-350m (350M params) - Fast, lightweight
        - facebook/opt-1.3b (1.3B params) - Better quality, slower
        - google/flan-t5-base (250M params) - Good for Q&A
        """
        logger.info(f"Loading local LLM: {model_name}")
        logger.info("This may take a few minutes on first run (downloading model)...")
        
        try:
            # Use pipeline for simplicity
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=-1,  # CPU (-1), use 0 for GPU
                max_length=1024,
                truncation=True
            )
            
            logger.info("✅ Local LLM loaded successfully")
            logger.info(f"   Model: {model_name}")
            logger.info("   Device: CPU")
            
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
        """
        Generate response using local LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Generated text
        """
        try:
            logger.info("Generating response with local LLM...")
            
            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the prompt)
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            logger.info("✅ Response generated")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."