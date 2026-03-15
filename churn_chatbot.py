import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3-flash-preview")


def churn_chat(question, customer_data, prediction):

    prompt = f"""
You are a banking data analyst.

Customer data:
{customer_data}

Churn probability:
{prediction}

Answer the user's question and explain churn risk clearly.

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text