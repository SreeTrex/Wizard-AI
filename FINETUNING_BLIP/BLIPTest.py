import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
import pickle

MODEL_PATH = "./finetuned_model.pth"
VOCAB_PATH = "./data/vocab.pkl"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

def generate_caption(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  
        predicted_caption = decode_caption(output)
    return predicted_caption

def decode_caption(output):
    token_ids = output.argmax(dim=-1).squeeze().tolist()
    words = [vocab[idx] for idx in token_ids if idx in vocab]
    return " ".join(words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image captioning model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image)

    caption = generate_caption(model, image_tensor)
    print(f"\n Predicted Caption: {caption}")
