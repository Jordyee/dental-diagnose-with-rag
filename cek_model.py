import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key dari file .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Mencari model yang tersedia untuk API Key Anda...\n")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Bisa digunakan: {m.name}")
except Exception as e:
    print(f"Error mengakses API: {e}")