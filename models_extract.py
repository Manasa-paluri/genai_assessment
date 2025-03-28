import google.generativeai as genai
from dotenv import load_dotenv 
import os 
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY) 
try:
    available_models = genai.list_models()
    for model in available_models:
        print(model.name)  # Print available models
except Exception as e:
    print(f"Error fetching available models: {e}")
 