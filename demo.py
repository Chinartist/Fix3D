from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("/nvme0/public_data/Occupancy/proj/cache/google/siglip2-so400m-patch16-512")
processor = AutoProcessor.from_pretrained("/nvme0/public_data/Occupancy/proj/cache/google/siglip2-so400m-patch16-512")

url = "/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/3DRealCar_Renders/image_pairs_output/2024_05_26_19_01_07/gt/00000.png"
image = Image.open(url)

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
