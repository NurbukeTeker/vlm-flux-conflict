import os
import random
from tqdm import tqdm
from diffusers import FluxPipeline
from PIL import Image

def load_prompts(path):
    """Load prompt lines from a .txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def generate_images(model, prompts, out_dir, num_images=20):
    """Generate images using Flux from a prompt list."""
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Generating {out_dir}"):
        prompt = random.choice(prompts)
        result = model(prompt)
        img = result.images[0]
        img.save(os.path.join(out_dir, f"sample_{i:03d}.png"))

def main():
    # Load Flux model
    print("Loading Flux model…")
    model = FluxPipeline.from_pretrained("black-forest-labs/flux-1-dev")

    categories = {
        "traffic": "flux_generation/prompts/traffic_signs.txt",
        "ui": "flux_generation/prompts/ui_warnings.txt",
        "medical": "flux_generation/prompts/medicine_labels.txt",
        "packaging": "flux_generation/prompts/packaging_conflicts.txt",
    }

    # Generate 20 images per category (change to 50/100 later)
    for name, path in categories.items():
        prompts = load_prompts(path)
        generate_images(model, prompts, os.path.join("outputs", name), num_images=20)

if __name__ == "__main__":
    main()
