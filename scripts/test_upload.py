import requests
import json

url = "http://localhost:8000/session/preview_ocr"
file_path = r"D:\academic chatbot\data\Adaptive-RAG Learning to Adapt Retrieval-Augmented Large Language Models.pdf"

try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    with open('scripts/response.txt', 'w', encoding='utf-8') as f:
        try:
             json.dump(response.json(), f, indent=2)
        except:
             f.write(response.text)

except Exception as e:
    print(f"An error occurred: {e}")
