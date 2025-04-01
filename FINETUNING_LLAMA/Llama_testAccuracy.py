import torch
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from bert_score import score as bert_score

MODEL_DIR = "llama3_finetuned"
MODEL_FILE = "llama3_finetuned_model.bin"  
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.json"
TEST_DATA_PATH = "test_manual.json"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    state_dict=torch.load(f"{MODEL_DIR}/{MODEL_FILE}", map_location="cpu")
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

bleu_scores, rouge_l_scores, bert_p, bert_r, bert_f1 = [], [], [], [], []

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

for sample in test_data:
    input_text = sample["input"]
    expected_output = sample["expected_output"]

    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=100)

    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    expected_embedding = semantic_model.encode(expected_output, convert_to_tensor=True)
    generated_embedding = semantic_model.encode(generated_output, convert_to_tensor=True)

    similarity = util.cos_sim(expected_embedding, generated_embedding).item()

    bleu = sentence_bleu(
        [expected_output.split()], generated_output.split(),
        smoothing_function=SmoothingFunction().method1
    )
    bleu_scores.append(bleu)

    rouge_l = rouge.score(expected_output, generated_output)["rougeL"].fmeasure
    rouge_l_scores.append(rouge_l)

    P, R, F1 = bert_score([generated_output], [expected_output], lang="en")
   
    bert_f1.append(F1.item())
 

print("\n==== Final Accuracy Metrics ====")

print(f"Mean BLEU Score: {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"Mean ROUGE-L Score: {sum(rouge_l_scores) / len(rouge_l_scores):.4f}")
print(f"Mean BERTScore F1: {sum(bert_f1) / len(bert_f1):.4f}")
