import os
import requests
from PIL import Image, ImageDraw
from google import genai

# ---------------- CONFIG ----------------
TOPICS = [
    "Benefits of soaked almonds",
    "High protein vegetarian foods",
    "Foods that improve gut health"
]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- GEMINI SETUP ----------------
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate_image_prompt(topic):
    response = client.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=f"""
Create a minimal flat lay Instagram food photography prompt.

Topic: {topic}

Rules:
- Clean white or pastel background
- Soft natural lighting
- Professional food photography
- No text in image
"""
    )
    return response.text.strip()

def generate_nutrition_text(topic):
    response = client.models.generate_content(
        model="mode
