from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import csv
import time

# Load model / tokenizer 
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="auto"
)
streamer = TextStreamer(tokenizer)

# Define prompt
REWRITE_PROMPT = """
Please rewrite the following text according to the rewriting principles used in calibration datasets for quantized LLMs:

Rewriting goals:
- Preserve all important information (e.g., names, locations, products)
- Make the text fluent, grammatically correct, and natural sounding
- Avoid repetitive or template-like structures
- Promote diversity in phrasing and sentence construction
- Do not remove or skip any critical details from the original

Text to rewrite:
"{text}"
"""

# Text generation function
def rewrite_text(text, max_tokens=256):
    prompt = REWRITE_PROMPT.format(text=text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Text to rewrite:")[1].strip() if "Text to rewrite:" in decoded else decoded.strip()


input_file = "../data/clean_1024.txt"
output_file = "output_deepseek.csv"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline='', encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["Origin", "Modified", "Label"])

    for i, row in enumerate(reader):
        original = row["Original"]
        print(f"Processing line {i+1}...")
        rewritten = rewrite_text(original)
        writer.writerow([original, rewritten, "0"])
        time.sleep(0.1)  # Optional: adjust based on model speed
