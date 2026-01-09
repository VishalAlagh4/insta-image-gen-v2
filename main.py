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

GEMINI_MODEL = "models/gemini-1.0-pro"  # <-- THIS IS THE FIX

def generate_image_prompt(topic):
    prompt = (
        "Create a minimal flat lay Instagram food photography prompt. "
        f"Topic: {topic}. "
        "Clean white background, soft natural lighting, "
        "professional food photography, no text in image."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text.strip()

def generate_nutrition_text(topic):
    text_prompt = (
        "Write short Instagram nutrition content. "
        f"Topic: {topic}. "
        "Format: Title, then three bullet points."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=text_prompt
    )
    return response.text.strip()

# ---------------- IMAGE GENERATION ----------------
HF_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

def generate_image(prompt, path):
    r = requests.post(
        HF_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=120
    )

    if r.headers.get("content-type") != "image/png":
        raise RuntimeError("Hugging Face did not return an image.")

    with open(path, "wb") as f:
        f.write(r.content)

def format_and_overlay(image_path, text, out_path):
    img = Image.open(image_path).convert("RGB").resize((1080, 1080))
    draw = ImageDraw.Draw(img)

    draw.multiline_text(
        (40, 720),
        text,
        fill="black",
        spacing=12
    )

    img.save(out_path)

# ---------------- PIPELINE ----------------
def run():
    for idx, topic in enumerate(TOPICS):
        print(f"Processing: {topic}")

        prompt = generate_image_prompt(topic)
        text = generate_nutrition_text(topic)

        raw = f"{OUTPUT_DIR}/raw_{idx}.png"
        final = f"{OUTPUT_DIR}/post_{idx}.png"

        generate_image(prompt, raw)
        format_and_overlay(raw, text, final)

        print("Created:", final)

if __name__ == "__main__":
    run()
