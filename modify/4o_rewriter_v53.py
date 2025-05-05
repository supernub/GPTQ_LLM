
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
OUTPUT_CSV       = "output_4o_v53.csv"
RATE_LIMIT_SEC   = 1.2                    # 每次调用后等待
RESUME           = True                   # 若输出已存在则断点续写
TEMP             = 0.65                   # 生成温度
MAX_TOKENS       = 80                     # 够写 ≤50 词的句子
# ────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=API_KEY
)

# ───────── 固定 system / user 上下文 ────────────────────────────
SYSTEM_CONTENT = """
You are an expert in large language model compression. I will give you a summary of a paper describing a quantization method, GPTQ. Your goal is to rewrite the calibration data used for the GPTQ compression technique to better align with the quantization objectives.

GPTQ method summary: GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that compresses large language models (LLMs) to as low as 3–4 bits per weight with minimal accuracy loss. It extends Optimal Brain Quantization (OBQ) by using second-order approximations (layer-wise Hessians) to minimize output reconstruction error. GPTQ introduces innovations like processing weights in a fixed order, batching column updates for efficient GPU use, and applying Cholesky-based updates for numerical stability. These enable the quantization of 175B-parameter models within hours, preserving generation and zero-shot task performance, even under aggressive compression.

""".strip()

USER_RULES = """
To best support the GPTQ quantization process, the calibration data must:
(1) trigger diverse, representative activations across layers, similar to real-world generative use cases;
(2) reflect natural language patterns common in web-scale corpora, covering varied topics, structures, and entity types;
(3) avoid overly numeric, template-like, or repetitive patterns, which provide less useful reconstruction signals.

### Your Task:
Rewrite the following calibration sentence so it resembles a fluent, full-sentence natural language input typical of generative model tasks. Ensure the rephrased sentence enriches the syntactic and semantic content, contains natural variations and context, and simulates realistic model activations useful for layer-wise reconstruction during quantization.

Output exactly **one** rephrased sentence—no explanations, no quotes, no formatting.
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

