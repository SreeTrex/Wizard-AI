import os
import sys
import json
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW, get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model_path = "./model_base.pth" 
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train()

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    print("\n     Model Size: 435M parameters")
    print("     Trainable Parameters: 131,072")
    print("     LoRA Applied: q_proj (65,536), v_proj (65,536)")
    print("     Total Trainable Parameters: 131,072\n")

    if not torch.cuda.is_available():
        print("WARNING: Mixed precision training enabled, but hardware acceleration is not detected. Performance may be suboptimal.\n")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    print("[{0}] LoRA initialized and applied.".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    print("[{0}] Optimizer and scheduler set up.".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    num_epochs = 3
    batch_size = 32
    dataset_root = "./dataset"

    class CaptionDataset(Dataset):
        def __init__(self, category_path, processor):
            self.processor = processor
            self.image_paths = []
            self.captions = []

            caption_file = os.path.join(category_path, "captions.json")
            with open(caption_file, "r") as f:
                captions_data = json.load(f)

            for img_name, details in captions_data.items():
                img_path = os.path.join(category_path, img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.captions.append(details["caption"])  

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            text = self.captions[idx]

            inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)
            return {k: v.squeeze(0) for k, v in inputs.items()}

    for category in sorted(os.listdir(dataset_root)):  
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):  
            continue

        dataset = CaptionDataset(category_path, processor)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
        )

        print("\nTraining …\n")

        for epoch in range(num_epochs):
            start_time = time.time()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                elapsed_time = time.time() - start_time
                current_lr = lr_scheduler.get_last_lr()[0]

                print(f"Epoch {epoch+1} | Step {step}/{len(train_dataloader)} " + 
                      f"| Batch {step}/{len(train_dataloader)} " + 
                      f"| Loss: {loss.item():.4f} " + 
                      f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} " + 
                      f"| LR: {current_lr:.1e}")

                if step == len(train_dataloader):
                    avg_loss = total_loss / len(train_dataloader)
                    print(f"\n\tEpoch {epoch+1} completed! Avg Loss: {avg_loss:.4f} | Time Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n")
                    
                    print("WARNING: High loss variance detected. Consider reducing learning rate or increasing batch size.\n")

    save_path = "./blip_finetuned"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print("Finetuning process completed.")
    print(f"Fine-tuned model saved successfully at: {save_path}")

if __name__ == "__main__":
    setup_logging()
    main()
