import os
import time
import textwrap
import requests
from PIL import Image, ImageDraw, ImageFont
import io
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
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty prompt text.")
    return text

def generate_nutrition_text(topic: str) -> str:
    text_prompt = (
        "Write short Instagram nutrition content. "
        f"Topic: {topic}. "
        "Format: Title, then three bullet points. Keep it concise and positive."
    )
    response = model.generate_content(text_prompt)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty caption text.")
    return text

# ---------------- HUGGING FACE ROUTER (FREE) ----------------
HF_URL = "https://router.huggingface.co"
HF_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
    "Accept": "image/png",
    "Content-Type": "application/json",
}

# Choose a free, widely available model
# sdxl-turbo is fast and often available; fallback to stable-diffusion-2-1 if needed
HF_PRIMARY_MODEL = "stabilityai/sdxl-turbo"
HF_FALLBACK_MODEL = "stabilityai/stable-diffusion-2-1"

def hf_generate_image(prompt: str, model_name: str, timeout: int = 120) -> bytes:
    """
    Calls Hugging Face Router with explicit model selection.
    Returns raw PNG bytes if successful.
    """
    if not HF_HEADERS["Authorization"]:
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    payload = {
        "inputs": prompt,
        "model": model_name
    }

    r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=timeout)

    # Some routers may return 200 with JSON error; check content-type
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "image/png" not in ctype:
        raise RuntimeError(f"Hugging Face did not return an image. Status: {r.status_code}. Response: {r.text}")

    return r.content

def generate_image_with_retry(prompt: str, retries: int = 2, delay: int = 3) -> bytes:
    """
    Try primary model first; if it fails, fallback to a secondary model.
    Retries transient failures.
    """
    last_err = None

    # Try primary model
    for attempt in range(1, retries + 2):
        try:
            return hf_generate_image(prompt, HF_PRIMARY_MODEL)
        except Exception as e:
            last_err = e
            print(f"[Primary {HF_PRIMARY_MODEL} attempt {attempt}] failed: {e}")
            if attempt <= retries:
                time.sleep(delay)

    # Fallback model
    for attempt in range(1, retries + 2):
        try:
            return hf_generate_image(prompt, HF_FALLBACK_MODEL)
        except Exception as e:
            last_err = e
            print(f"[Fallback {HF_FALLBACK_MODEL} attempt {attempt}] failed: {e}")
            if attempt <= retries:
                time.sleep(delay)

    raise RuntimeError(f"Hugging Face generation failed after retries: {last_err}")

# ---------------- IMAGE POST-PROCESSING ----------------
def format_and_overlay(image_bytes: bytes, text: str) -> Image.Image:
    """
    - Loads PNG bytes into PIL
    - Resizes to Instagram square
    - Overlays wrapped caption text near the bottom
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(CANVAS_SIZE)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=40)

    draw.multiline_text(
        (40, CAPTION_AREA_Y),
        wrapped,
        fill="black",
        spacing=12,
        font=font,
    )

    return img

# ---------------- PIPELINE ----------------
def run():
    for idx, topic in enumerate(TOPICS):
        print(f"\n=== Processing: {topic} ===")

        try:
            prompt = generate_image_prompt(topic)
            print("Gemini prompt:\n", prompt)

            caption = generate_nutrition_text(topic)
            print("Gemini caption:\n", caption)

            # Generate image via HF Router
            raw_png_bytes = generate_image_with_retry(prompt)

            raw_path = os.path.join(OUTPUT_DIR, f"raw_{idx}.png")
            with open(raw_path, "wb") as f:
                f.write(raw_png_bytes)

            final_img = format_and_overlay(raw_png_bytes, caption)
            final_path = os.path.join(OUTPUT_DIR, f"post_{idx}.png")
            final_img.save(final_path, format="PNG")

            print("Created:", final_path)

        except Exception as e:
            print(f"Error processing '{topic}': {e}")

if __name__ == "__main__":
    run()
