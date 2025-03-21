import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
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

def extract_key_points(text, text_label="Original Text"):
    prompt = f"""
Please list the core information of the following {text_label} in bullet points.
Do not provide any extra analysis or conclusion, just the essential facts or events:

{text}

Key Points:
""".strip()
    return generate_response(prompt)

def judge_semantic_equivalence(keypointsA, keypointsB):
    prompt = f"""
Here are two sets of key points from two texts. 
Please judge whether they express the same core meaning or not.

Key Points A:
{keypointsA}

Key Points B:
{keypointsB}

If they are essentially the same, answer "YES" and briefly explain.
If they differ significantly, answer "NO" and briefly explain.

Answer:
""".strip()
    return generate_response(prompt)
 
def load_csv_as_dict(csv_path):

    samples_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row["sample_id"])
            text = row["text"]
            samples_dict[sample_id] = text
    return samples_dict

def main():
    origin_csv = "/GPTQ_llm/data/origin_samples.csv"      
    optimized_csv = "/GPTQ_llm/data/optimized_samples.csv"  
    output_csv = "semantic_result.csv"

    origin_dict = load_csv_as_dict(origin_csv)
    optimized_dict = load_csv_as_dict(optimized_csv)

    results = []

    all_ids = sorted(origin_dict.keys())
    for sample_id in all_ids:
        origin_text = origin_dict[sample_id]

        optimized_text = optimized_dict.get(sample_id, "")


        keypoints_A = extract_key_points(origin_text, text_label=f"Origin {sample_id}")
        keypoints_B = extract_key_points(optimized_text, text_label=f"Optimized {sample_id}")


        final_judgment = judge_semantic_equivalence(keypoints_A, keypoints_B)
        # 如果包含"YES"就认为一致(1), 否则认为不一致(0)
        # "YES"或"NO"开头
        judgment_upper = final_judgment.strip().upper()
        if "YES" in judgment_upper:
            consistency = 1
        else:
            consistency = 0

        results.append((sample_id, consistency))

    # 只包含 sample_id, consistency
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["sample_id", "consistency"])
        for sample_id, consistency in results:
            writer.writerow([sample_id, consistency])

    print(f"complete : {output_csv}")

if __name__ == "__main__":
    main()
