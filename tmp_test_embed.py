import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

try:
    print("Testing gemini-embedding-2-preview...")
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=["Đây là một câu test."],
    )
    print(f"Success! Vector length: {len(result.embeddings[0].values)}")
except Exception as e:
    print(f"Error: {e}")
