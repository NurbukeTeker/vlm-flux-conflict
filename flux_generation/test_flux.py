import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="nebius",
    api_key=HF_TOKEN,
)

image = client.text_to_image(
    prompt="Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-schnell",
)

image.save("test_flux.png")
print("Image saved as test_flux.png")
