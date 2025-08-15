import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv('API_1')

genai.configure(api_key=API_KEY)

models = genai.list_models()
for model in models:
    print(model.name)

# Google Generative AI Python SDK에서 지원하는 모델 리스트를 가져오는 예시