import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key: {api_key[:20]}...\n")

genai.configure(api_key=api_key)

# List available models first
print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")

print("\n" + "="*60 + "\n")

# Try different models
models_to_try = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
]

for model_name in models_to_try:
    print(f"Testing {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        
        prompt = """Based on this context:

CONTEXT:
We accept multiple payment methods including credit cards (Visa, MasterCard, American Express), 
PayPal, and bank transfers.

USER QUESTION: What payment methods do you accept?

Answer based on the context above."""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200,
            )
        )
        
        print(f"✅ {model_name} WORKS!")
        print(f"\nResponse:\n{response.text}\n")
        print("="*60 + "\n")
        break
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}\n")
        continue