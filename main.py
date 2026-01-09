import os
import requests
import io
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# ---------------- CONFIG ----------------
TOPICS = [
    "Benefits of soaked almonds",
    "High protein vegetarian foods",
    "Foods that improve gut health",
]

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANVAS_SIZE = (1080, 1080)
CAPTION_AREA_Y = 720
FONT_SIZE = 36

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_image_prompt(topic: str) -> str:
    prompt = (
        "Create a minimal flat lay Instagram food photography prompt. "
        f"Topic: {topic}. "
        "Clean white background, soft natural lighting, "
        "professional food photography, sharp focus, high resolution, no text or logos in the image."
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()

def generate_nutrition_text(topic: str) -> str:
    text_prompt = (
        "Write short Instagram nutrition content. "
        f"Topic: {topic}. "
        "Format: Title, then three bullet points. Keep it concise and positive."
    )
    response = model.generate_content(text_prompt)
    return (response.text or "").strip()

# ---------------- HUGGING FACE FREE IMAGE GENERATION ----------------
HF_MODEL = "stabilityai/stable-diffusion-2-1"  # free model hosted on Hugging Face
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}

def generate_image(prompt: str, path: str):
    r = requests.post(HF_URL, headers=HF_HEADERS, json={"inputs": prompt}, timeout=120)
    if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"Hugging Face did not return an image. Response: {r.text}")
    with open(path, "wb") as f:
        f.write(r.content)

def format_and_overlay(image_path: str, text: str, out_path: str):
    img = Image.open(image_path).convert("RGB").resize(CANVAS_SIZE)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.multiline_text((40, CAPTION_AREA_Y), text, fill="black", spacing=12, font=font)
    img.save(out_path)

# ---------------- PIPELINE ----------------
def run():
    for idx, topic in enumerate(TOPICS):
        print(f"\n=== Processing: {topic} ===")
        try:
            prompt = generate_image_prompt(topic)
            print("Gemini prompt:\n", prompt)

            caption = generate_nutrition_text(topic)
            print("Gemini caption:\n", caption)

            raw = os.path.join(OUTPUT_DIR, f"raw_{idx}.png")
            final = os.path.join(OUTPUT_DIR, f"post_{idx}.png")

            generate_image(prompt, raw)
            format_and_overlay(raw, caption, final)

            print("Created:", final)
        except Exception as e:
            print(f"Error processing '{topic}': {e}")

if __name__ == "__main__":
    run()
