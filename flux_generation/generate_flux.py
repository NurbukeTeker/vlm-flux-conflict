import os
import time
import random
from tqdm import tqdm
from huggingface_hub import InferenceClient
from PIL import Image
import io


# ------------------------------
# Load prompts
# ------------------------------
def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ------------------------------
# Generate a single image (with retry safety)
# ------------------------------
def generate_single_image(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            result = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )
            return result
        except Exception as e:
            print(f"⚠ Error: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(2)

    raise RuntimeError("Image generation failed after all retries.")


# ------------------------------
# Generate images for a category
# ------------------------------
def generate_category(client, prompts, out_dir, num_images=20):
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Generating {out_dir}"):
        prompt = random.choice(prompts)
        image = generate_single_image(client, prompt)

        image.save(os.path.join(out_dir, f"sample_{i:03d}.png"))


# ------------------------------
# MAIN
# ------------------------------
def main():
    print("🚀 Loading Nebius FLUX client…")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise EnvironmentError(
            "❌ HF_TOKEN environment variable not found. Run: setx HF_TOKEN \"your_token\""
        )

    client = InferenceClient(
        provider="nebius",
        api_key=HF_TOKEN
    )

    # Your categories
    categories = {
       # "traffic": "flux_generation/prompts/traffic_signs.txt",
        "ui": "flux_generation/prompts/ui_warnings.txt",
        "medical": "flux_generation/prompts/medicine_labels.txt",
        "packaging": "flux_generation/prompts/packaging_conflicts.txt",
    }

    for name, path in categories.items():
        print(f"\n📁 Generating category: {name}")
        prompts = load_prompts(path)
        generate_category(
            client,
            prompts,
            out_dir=os.path.join("outputs", name),
            num_images=20  # Later set to 50 / 100
        )

    print("\n🎉 All images generated successfully!")


if __name__ == "__main__":
    main()
