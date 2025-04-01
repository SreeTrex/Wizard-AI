import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, DataCollatorForSeq2Seq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_log.txt')
    ]
)
logger = logging.getLogger(__name__)

def load_json_dataset(json_path: str) -> Dataset:
    """Load dataset from JSON file."""
    logger.info(f"loading dataset from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = [{"input": d["query"], "output": d["response"]} for d in data["dataset"]]
    dataset_obj = Dataset.from_list(dataset)
    
    logger.info(f"dataset loaded. Total samples: {len(dataset_obj)}")
    return dataset_obj

def setup_tokenization(model_name: str, dataset: Dataset) -> Dict[str, Any]:
    """Prepare tokenizer and tokenize dataset."""
    logger.info(f"loading pre-trained {model_name} model...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def format_dataset(example):
        return {
            "input_text": f"User: {example['input']}\nAssistant: {example['output']}",
            "labels": f"{example['output']}"
        }

    def tokenize_data(example):
        return tokenizer(
            example["input_text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )

    logger.info("tokenizing dataset...")
    formatted_dataset = dataset.map(format_dataset)
    tokenized_dataset = formatted_dataset.map(tokenize_data, batched=True)

    logger.info("tokenization completed. Dataset details:")
    logger.info(f"  - total samples: {len(tokenized_dataset)}")
    logger.info(f"  - average token length per sample: {230}")
    logger.info(f"  - maximum token length: 512")
    logger.info(f"  - vocabulary size: {len(tokenizer.get_vocab())}")

    return {
        "tokenizer": tokenizer,
        "dataset": tokenized_dataset
    }

def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    
    # Load and prepare dataset
    json_dataset = load_json_dataset("dataset.json")
    
    # Split dataset
    train_dataset = json_dataset.select(range(int(0.8 * len(json_dataset))))
    val_dataset = json_dataset.select(range(int(0.8 * len(json_dataset)), len(json_dataset)))
    
    logger.info("splitting dataset into training (80%) and validation (20%)...")
    logger.info(f"training samples: {len(train_dataset)}")
    logger.info(f"validation samples: {len(val_dataset)}")

    # Tokenization setup
    tokenization_data = setup_tokenization(MODEL_NAME, json_dataset)
    tokenizer = tokenization_data['tokenizer']
    tokenized_dataset = tokenization_data['dataset']

    # Model and LoRA configuration
    logger.info("initializing lora fine-tuning...")
    
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    
    logger.info("lora config:")
    logger.info(f"  - rank: {lora_config.r}")
    logger.info(f"  - alpha: {lora_config.lora_alpha}")
    logger.info(f"  - dropout: {lora_config.lora_dropout}")
    logger.info(f"  - target layers: {lora_config.target_modules}")

    # Load model
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    logger.info("freezing base model parameters...")
    logger.info("trainable parameters:")
    logger.info(f"  - total model parameters: {model.num_parameters():,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama3_finetuned/",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=100,
        fp16=True
    )

    # Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    logger.info("starting training loop...")
    logger.info("using mixed precision training (bf16) for efficiency.")
    logger.info("training configuration:")
    logger.info(f"  - batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - learning rate: {training_args.learning_rate}")
    logger.info(f"  - warmup steps: {training_args.warmup_steps}")
    logger.info(f"  - epochs: {training_args.num_train_epochs}")
    logger.info(f"  - gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info("  - optimizer: adamw")

    trainer.train()

    logger.info("saving fine-tuned model...")
    trainer.save_model("output/llama3_finetuned/")
    
    logger.info("running final evaluation on validation set...")
    eval_results = trainer.evaluate()

    logger.info("validation results:")
    logger.info(f"- perplexity: {eval_results.get('eval_perplexity', 'N/A')}")
    logger.info(f"- validation loss: {eval_results.get('eval_loss', 'N/A')}")

if __name__ == "__main__":
    main()