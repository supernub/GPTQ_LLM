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

    #  "Text to rewrite" 之后的部分作为结果（更可靠）
    if "Text to rewrite:" in decoded:
        parts = decoded.split("Text to rewrite:")[-1]
        cleaned = parts.strip().strip('"')
    else:
        cleaned = decoded.strip()

    #  保留最后一句（通常为改写结果），去掉“Sure! Here's...”这类冗余
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    if len(lines) == 1:
        return lines[0]
    else:
        return lines[-1] 

input_file = "../data/clean_10_test.txt"
output_file = "output_ds_10test.csv"


with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline='', encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Original", "Modified", "Label"])

    for i, line in enumerate(infile):
        original = line.strip()
        if not original:
            continue
        print(f"Processing line {i+1}: {original[:40]}...")
        rewritten = rewrite_text(original)
        writer.writerow([original, rewritten, "0"])
        time.sleep(0.1)
