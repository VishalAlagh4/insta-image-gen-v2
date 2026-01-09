import os
import time
import textwrap
import base64
import io
import requests
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

# Image render settings
CANVAS_SIZE = (1080, 1080)  # Instagram square
CAPTION_AREA_Y = 720        # Start Y for caption overlay
CAPTION_WIDTH = 1000        # Max text width for wrapping
FONT_PATH = None            # Use default PIL font; set a TTF path if you want custom
FONT_SIZE = 36

# Stability AI settings
STABILITY_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
STABILITY_MODEL = "stable-diffusion-xl-1024-v1-0"
STABILITY_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('STABILITY_API_KEY', '')}",
    "Accept": "application/json",
}

# Gemini setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------- GEMINI HELPERS ----------------
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


# ---------------- STABILITY AI IMAGE GENERATION ----------------
def stability_generate_image(prompt: str, timeout: int = 120) -> bytes:
    """
    Calls Stability AI to render an image from a text prompt.
    Returns raw PNG bytes.
    Handles both 'image' and 'images' response formats.
    """
    if not STABILITY_HEADERS["Authorization"]:
        raise RuntimeError("Missing STABILITY_API_KEY environment variable.")

    files = {
        "prompt": (None, prompt),
        "output_format": (None, "png"),
        "model": (None, STABILITY_MODEL),
    }

    r = requests.post(STABILITY_URL, headers=STABILITY_HEADERS, files=files, timeout=timeout)

    if r.status_code != 200:
        raise RuntimeError(f"Stability AI failed: {r.status_code} {r.text}")

    data = r.json()
    # Debug preview (trim to avoid huge logs)
    print("Stability response keys:", list(data.keys()))

    # Prefer 'images' list if present; otherwise fallback to 'image'
    image_base64 = None
    if "images" in data and isinstance(data["images"], list) and data["images"]:
        # Expect objects like {"base64": "..."}
        first = data["images"][0]
        image_base64 = first.get("base64")
    elif "image" in data and isinstance(data["image"], str):
        image_base64 = data["image"]

    if not image_base64:
        raise RuntimeError(f"No image returned from Stability: {data}")

    return base64.b64decode(image_base64)


def generate_image_with_retry(prompt: str, retries: int = 2, delay: int = 3) -> bytes:
    """
    Retry wrapper for Stability image generation to handle transient failures.
    """
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            return stability_generate_image(prompt)
        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}] Stability generation failed: {e}")
            if attempt <= retries:
                time.sleep(delay)
    raise RuntimeError(f"Stability generation failed after retries: {last_err}")


# ---------------- IMAGE POST-PROCESSING ----------------
def format_and_overlay(image_bytes: bytes, text: str) -> Image.Image:
    """
    - Loads PNG bytes into PIL
    - Resizes to Instagram square
    - Overlays wrapped caption text near the bottom
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(CANVAS_SIZE)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if FONT_PATH is None else ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Wrap text to fit caption area
    wrapped = textwrap.fill(text, width=40)

    # Draw caption
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
            # 1) Gemini prompt + caption
            prompt = generate_image_prompt(topic)
            print("Gemini prompt:\n", prompt)

            caption = generate_nutrition_text(topic)
            print("Gemini caption:\n", caption)

            # 2) Stability render
            raw_png_bytes = generate_image_with_retry(prompt)

            # 3) Save raw image
            raw_path = os.path.join(OUTPUT_DIR, f"raw_{idx}.png")
            with open(raw_path, "wb") as f:
                f.write(raw_png_bytes)

            # 4) Overlay caption and save final
            final_img = format_and_overlay(raw_png_bytes, caption)
            final_path = os.path.join(OUTPUT_DIR, f"post_{idx}.png")
            final_img.save(final_path, format="PNG")

            print("Created:", final_path)

        except Exception as e:
            print(f"Error processing '{topic}': {e}")


if __name__ == "__main__":
    run()
