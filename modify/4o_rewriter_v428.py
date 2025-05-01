
# ────────────────────────────────────────────────────────────────
#  rewriter4o_ver428.py – GPT-4o calibration-sentence rewriter
#
#  1.  Read every row from input.csv   (expects column “Original”)
#  2.  Use the fixed GPTQ system / user prompt to rewrite it
#  3.  Append to output_4o_fullpdf.csv      columns: Original, Modified, Label
#  4.  Supports RESUME = True  → 跳过已写入的行
# ----------------------------------------------------------------

import csv, time
from pathlib import Path
from openai import OpenAI            

# ───────── USER CONFIG ──────────────────────────────────────────
API_KEY          = "YOUR_OPENAI_API_KEY"
MODEL_NAME       = "gpt-4o"
INPUT_CSV        = "input.csv"            # 原始数据，需列 “Original”
OUTPUT_CSV       = "output_4o_v428.csv"
RATE_LIMIT_SEC   = 1.2                    # 每次调用后等待
RESUME           = True                   # 若输出已存在则断点续写
TEMP             = 0.65                   # 生成温度
MAX_TOKENS       = 80                     # 够写 ≤50 词的句子
# ────────────────────────────────────────────────────────────────

client = OpenAI(api_key=API_KEY)

# ───────── 固定 system / user 上下文 ────────────────────────────
SYSTEM_CONTENT = """
You are an expert in large language model compression. I will give you the summary of a paper describing a quantization method, GPTQ. Your goal is to rewrite the calibration data used for the GPTQ compression technique to better align with the quantization objectives.

GPTQ method summary: GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that compresses large language models (LLMs) by quantizing their weights to as low as 3 or 4 bits with minimal accuracy loss. It builds upon Optimal Brain Quantization (OBQ) by approximating second-order information (using the Hessian of layer outputs) to minimize the reconstruction error during quantization. GPTQ improves scalability through several innovations: (a) it avoids costly greedy weight ordering by processing weights in a fixed order; (b) it batches updates across blocks of columns to improve GPU utilization (lazy updates); and (c) it reformulates weight updates using numerically stable Cholesky decompositions to avoid precision loss in very large models. These optimizations allow GPTQ to quantize models with hundreds of billions of parameters in a few hours while preserving performance in generation and zero-shot tasks.
""".strip()

USER_RULES = """
To better support the GPTQ quantization process, the calibration data should:
(1) stimulate realistic and diverse activations across layers, particularly those triggered in natural generative use cases;
(2) contain richer syntactic and semantic structures that mimic the model’s inference-time inputs (e.g., Wikipedia, dialogue, instructions);
(3) reflect common patterns in natural language, such as numbers, measurements, places, dates, and named entities—but embedded in fluent sentences.

### Your Task:
Rephrase the following calibration sentence so that it simulates realistic full-sentence completions with diverse vocabulary and syntactic structures, focusing on preserving semantic richness and enabling accurate layer output reconstruction for quantization. Avoid numeric-heavy or overly structured patterns. Instead, use natural language with diverse context and common word distributions seen in web-scale corpora, such as stories, conversations, or encyclopedia-style explanations.

Return exactly **one** rephrased sentence and nothing else—no quotes, no explanations, no prefixes.
""".strip()

BASE_CONTEXT = [
    {"role": "system", "content": SYSTEM_CONTENT},
    {"role": "user",   "content": USER_RULES},
]

# ───────── GPT-4o 调用封装 ────────
def gpt_rewrite(original_sentence: str) -> str:
    """Return one rewritten sentence (stripped)."""
    messages = BASE_CONTEXT + [
        {"role": "user",
         "content": f'Original calibration premise: "{original_sentence}"'}
    ]
    resp = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = messages,
        temperature = TEMP,
        max_tokens  = MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()

# ───────── 主流程：遍历 CSV & 改写 ─────────────────────────────
def rewrite_dataset():
    done = set()
    if RESUME and Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
            done = {row["Original"] for row in csv.DictReader(f)}

    with open(INPUT_CSV,  newline='', encoding='utf-8') as infile, \
         open(OUTPUT_CSV, "a", newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile,
                                fieldnames=["Original", "Modified", "Label"])
        if outfile.tell() == 0:
            writer.writeheader()

        for row in reader:
            origin = row["Original"].strip()
            if origin in done:
                continue

            try:
                modified = gpt_rewrite(origin)
            except Exception as exc:
                print(f"[ERROR] {exc} – blank output recorded.")
                modified = ""

            writer.writerow({"Original": origin,
                             "Modified": modified,
                             "Label":    0})
            print(f"✓  {origin[:45]}  →  {modified[:45]}")
            time.sleep(RATE_LIMIT_SEC)


if __name__ == "__main__":
    print("▶  GPT-4o calibration-sentence rewriting …")
    rewrite_dataset()
    print("✅  Done.  Output saved to", OUTPUT_CSV)
