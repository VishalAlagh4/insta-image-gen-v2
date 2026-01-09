import os
import requests
from PIL import Image, ImageDraw
import google.generativeai as genai

# ---------------- CONFIG ----------------
TOPICS = [
    "Benefits of soaked almonds",
    "High protein vegetarian foods",
    "Foods that improve gut health"
]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_image_prompt(topic):
    response = model.generate_content(
        f"Create a minimal flat lay Instagram food photography prompt for: {topic}. No text in image."
    )
    return response.text.strip()

def generate_nutrition_text(topic):
    response = model.generate_content(
        f"""
Write short Instagram nutrition content.

Topic: {topic}
Format:
Title
• Point 1
• Point 2
• Point 3
"""
    )
    return response.text.strip()

# ---------------- IMAGE GENERATION ----------------
HF_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

def generate_image(prompt, path):
    r = requests.post(HF_URL, headers=HEADERS, json={"inputs": prompt}, timeout=60)
    with open(path, "wb") as f:
        f.write(r.content)

def format_and_overlay(image_path, text, out_path):
    img = Image.open(image_path).resize((1080, 1080))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((40, 720), text, fill="black", spacing=10)
    img.save(out_path)

# ---------------- PIPELINE ----------------
def run():
    for idx, topic in enumerate(TOPICS):
        prompt = generate_image_prompt(topic)
        text = generate_nutrition_text(topic)

        raw = f"{OUTPUT_DIR}/raw_{idx}.png"
        final = f"{OUTPUT_DIR}/post_{idx}.png"

        generate_image(prompt, raw)
        format_and_overlay(raw, text, final)

        print("Created:", final)

if __name__ == "__main__":
    run()
