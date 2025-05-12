# 4o_rewriter_v511.py – streamlined GPT‑4o calibration‑sentence rewriter (few‑shot ready)

import csv, time, re
from pathlib import Path
from openai import OpenAI

# ───────── USER CONFIG ──────────────────────────────────────────
API_KEY          = "YOUR_OPENAI_API_KEY"
MODEL_NAME       = "gpt-4o"
INPUT_CSV        = "input.csv"            # expects column "Original"
OUTPUT_CSV       = "output_4o_v511.csv"
FEWSHOT_TXT      = "fewshot.txt"          # optional few‑shot file
NUM_FEWSHOT      = 5
RATE_LIMIT_SEC   = 1.2
RESUME           = True
TEMP             = 0.65
MAX_TOKENS       = 80

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=API_KEY
)

# ───────── PROMPT SECTIONS ──────────────────────────────────────
SYSTEM_CONTENT = 'You are an expert in large language model compression. I will give you a summary of a paper describing a quantization method, GPTQ. Your goal is to rewrite the calibration data used for the GPTQ compression technique to better align with the quantization objectives.\n\nGPTQ method summary: GPTQ (Generative Pre-trained Transformer Quantization) is a post-training quantization method that compresses large language models (LLMs) to as low as 3–4 bits per weight with minimal accuracy loss. It extends Optimal Brain Quantization (OBQ) by using second-order approximations (layer-wise Hessians) to minimize output reconstruction error. GPTQ introduces innovations like processing weights in a fixed order, batching column updates for efficient GPU use, and applying Cholesky-based updates for numerical stability. These enable the quantization of 175B-parameter models within hours, preserving generation and zero-shot task performance, even under aggressive compression.'
USER_RULES     = 'To best support the GPTQ quantization process, the calibration data must:\n(1) trigger diverse, representative activations across layers, similar to real-world generative use cases;\n(2) reflect natural language patterns common in web-scale corpora, covering varied topics, structures, and entity types;\n(3) avoid overly numeric, template-like, or repetitive patterns, which provide less useful reconstruction signals.\n\n### Your Task:\nRewrite the following calibration sentence so it resembles a fluent, full-sentence natural language input typical of generative model tasks. Ensure the rephrased sentence enriches the syntactic and semantic content, contains natural variations and context, and simulates realistic model activations useful for layer-wise reconstruction during quantization.\n\n### Rewrite Guidelines\n1. **Keep Core Facts** – never drop or invent numbers, units, names, or places.\n2. **Length Window** – keep the rewritten sentence within ±20 percent of the original token length.\n3. **Match Register** – preserve the original genre (e.g., ad copy remains promotional; notices remain declarative).\n4. **Boost Diversity** – vary syntax with conjunctions, clauses, dashes, etc.; avoid uniform templates.\n5. **Mild Denoising** – fix obvious spelling/grammar errors, remove duplicated words, trim gibberish.\n6. **No New Facts** – add only connective words needed for fluency, not additional information.\nOutput exactly **one** rewritten sentence—no explanations, no quotation marks, no markdown.'

# ───────── FEW‑SHOT LOADER ─────────────────────────────────────
def load_fewshot(path: str, n: int):
    pairs, count = [], 0
    if not Path(path).exists():
        return pairs
    txt = Path(path).read_text(encoding='utf-8')
    pattern = re.compile(r"Original:\\s*(.+?)\\s*Changed:\\s*(.+?)(?:\\n\\n|$)", re.DOTALL)
    for m in pattern.finditer(txt):
        if count >= n:
            break
        orig, changed = [s.strip() for s in m.groups()]
        pairs.append({"role": "user", "content": orig})
        pairs.append({"role": "assistant", "content": changed})
        count += 1
    return pairs

FEWSHOT_CONTEXT = load_fewshot(FEWSHOT_TXT, NUM_FEWSHOT)

BASE_CONTEXT = [{"role": "system", "content": SYSTEM_CONTENT}] + FEWSHOT_CONTEXT + [{"role": "user", "content": USER_RULES}]

# ───────── GPT‑4o CALL ─────────────────────────────────────────
def gpt_rewrite(sentence: str) -> str:
    messages = BASE_CONTEXT + [{"role": "user", "content": sentence}]
    resp = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = messages,
        temperature = TEMP,
        max_tokens  = MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()

# ───────── MAIN LOOP ───────────────────────────────────────────
def rewrite_dataset():
    done = set()
    if RESUME and Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
            done = {row["Original"] for row in csv.DictReader(f)}

    with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
         open(OUTPUT_CSV, "a", newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["Original", "Modified", "Label"])
        if outfile.tell() == 0:
            writer.writeheader()

        for row in reader:
            origin = row["Original"].strip()
            if origin in done:
                continue
            try:
                modified = gpt_rewrite(origin)
            except Exception as e:
                print(f"[ERROR] {e} – blank output recorded.")
                modified = ""
            writer.writerow({"Original": origin, "Modified": modified, "Label": 0})
            print(f"✓ {origin[:40]} → {modified[:40]}")
            time.sleep(RATE_LIMIT_SEC)

if __name__ == "__main__":
    print("▶ GPT‑4o calibration‑sentence rewriting …")
    rewrite_dataset()
    print("✅ Done. Output saved to", OUTPUT_CSV)
