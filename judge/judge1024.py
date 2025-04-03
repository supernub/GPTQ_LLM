import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
FEWSHOT_FILE = "fewshot.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_response(prompt, max_tokens=256, temperature=0.7):

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if prompt in decoded:
        response = decoded.split(prompt)[-1].strip()
    else:
        response = decoded
    return response

def load_fewshot_prompt(prompt_file):
    """ few-shot """
    with open(prompt_file, "r", encoding="utf-8") as f:
        fewshot_text = f.read()
    return fewshot_text

def judge_semantic_equivalence(origin_text, changed_text, fewshot_text):
    """
    根据 few-shot prompt + 两段文本, 输出 "YES"/"NO" 
    """
    prompt = f"""{fewshot_text}

Original: {origin_text}
Changed: {changed_text}
Answer:
""".strip()
    result = generate_response(prompt)
    return result.strip()  # e.g. "YES" or "NO"

def main():
    # ========== 需要修改的部分：输入/输出 CSV 路径 ==========
    input_csv = "test_fewshot_optimized.csv"  #  origin, optimized, label
    output_csv = "judge_fewshot-test_result.csv"
    
    # 如果要使用 few-shot, 则加载 prompt
    fewshot_text = load_fewshot_prompt(FEWSHOT_FILE)

    # 读入 CSV
    rows = []
    with open(input_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            # row["origin"], row["optimized"], row["label"]
            rows.append(row)
    
    new_rows = []
    for row in rows:
        origin_text = row["origin"]
        changed_text = row["optimized"]

        # judge
        final_judgment = judge_semantic_equivalence(origin_text, changed_text, fewshot_text)
        jug = final_judgment.upper()
        if "YES" in jug:
            model_judgment = 1
        else:
            model_judgment = 0

        # label
        row["label"] = str(model_judgment)  # "0"/"1"

        new_rows.append(row)

    #  CSV，origin, optimized, label
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        fieldnames = ["origin", "optimized", "label"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in new_rows:
            writer.writerow({
                "origin": row["origin"],
                "optimized": row["optimized"],
                "label": row["label"]
            })
    
    print(f"Done. Results saved to {output_csv}.")

if __name__ == "__main__":
    main()
