import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_metric
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./models/blip_finetuned"  
model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
processor = BlipProcessor.from_pretrained(model_path)

print(" Fine-tuned model loaded!")

test_dataset_root = "./dataset" 

class CaptionDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.processor = processor
        self.data = []

        # Traverse through all categories
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            captions_file = os.path.join(category_path, "captions.json")

            # Check if captions.json exists
            if not os.path.exists(captions_file):
                print(f"⚠ WARNING: Skipping {category_path}, no captions.json found.")
                continue

            with open(captions_file, "r") as f:
                captions_data = json.load(f)

            # Iterate over images and their captions
            for item in captions_data:
                image_filename = item.get("image_filename", "")
                image_path = os.path.join(category_path, image_filename)  # Full image path
                text = item.get("caption", "")  # Caption text

                # Validate file existence
                if not os.path.exists(image_path):
                    print(f"⚠ WARNING: Image {image_path} not found, skipping.")
                    continue

                self.data.append({"image_path": image_path, "caption": text})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        text = item["caption"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")

        return {"inputs": inputs, "caption": text, "image_path": image_path}

test_dataset = CaptionDataset(test_dataset_root, processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f" Test dataset loaded! Found {len(test_dataset)} samples.")


predictions = []
references = []

for batch in test_dataloader:
    inputs = {k: v.to(device) for k, v in batch["inputs"].items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_length=30)

    predicted_caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    ground_truth_caption = batch["caption"][0]  

    predictions.append(predicted_caption)
    references.append([ground_truth_caption])  

meteor = load_metric("meteor")
cider = load_metric("cider")
spice = load_metric("spice")
bertscore = load_metric("bertscore")

meteor_result = meteor.compute(predictions=predictions, references=references)
cider_result = cider.compute(predictions=predictions, references=references)
spice_result = spice.compute(predictions=predictions, references=references)
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")

meteor_mean = np.mean(meteor_result["meteor"])
cider_mean = np.mean(cider_result["cider"])
spice_mean = np.mean(spice_result["spice"])
bertscore_mean = np.mean(bertscore_result["f1"])  

print("\n Evaluation Results:")
print(f" METEOR Score (Mean): {meteor_mean:.4f}")
print(f" CIDEr Score (Mean): {cider_mean:.4f}")
print(f" SPICE Score (Mean): {spice_mean:.4f}")
print(f" BERTScore (Mean F1): {bertscore_mean:.4f}")
