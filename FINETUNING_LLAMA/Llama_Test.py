import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_DIR = "llama3_finetuned"

config = AutoConfig.from_pretrained(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, config=config)
model.to(device)
model.eval()  

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        response = generate_response(query)
        print("\nModel Response:", response, "\n")
