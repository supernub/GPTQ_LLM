

import csv, time
from pathlib import Path
from openai import OpenAI            

# ───────── USER CONFIG ──────────────────────────────────────────
API_KEY          = "YOUR_OPENAI_API_KEY"
MODEL_NAME       = "gpt-4o"
INPUT_CSV        = "input.csv"   
OUTPUT_CSV       = "output_4o_v428.csv"
RATE_LIMIT_SEC   = 1.2
RESUME           = True
TEMP             = 0.65
MAX_TOKENS       = 80            
# ────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=API_KEY
)

SYSTEM_CONTENT = """
You are a specialist in post‑training quantization. 
Your only goal is to rewrite calibration sentences so they (a) keep every fact, quantity, and proper noun from the original, and (b) maximise activation diversity for GPTQ’s Hessian‑based 3‑4 bit weight reconstruction.

GPTQ recap (for context only): uses a second‑order error objective ΔW‖X‖², processes weights in fixed order with block‑wise lazy updates, and stabilises updates via Cholesky decomposition. Calibration text that exercises varied syntax, topics, and numerical ranges improves quantisation quality.
""".strip()

# ───────── USER_RULES ─────────────────────────────────
USER_RULES = """
Rewrite policy – follow ALL rules:

1. **Semantic fidelity** – Do not change or remove any facts, numbers, units, names, or relationships present in the original sentence.
2. **Style lottery** – For each rewrite randomly pick ONE style **(dialogue style removed)**:  
   • Narrative   • Expository   • First‑person reflection   • Instructional tip
3. **Natural flow** – No bullet points, no list markers, no template wording.
4. **Numerals** – You may include up to 3 numerals or dates; never dense tables or ID strings.
5. **Length** – Produce ONE complete English sentence ≤ 55 words. If longer, truncate gracefully.
6. **No meta‑narrative** – Do **NOT** add framing phrases like “In a lively discussion,” “One speaker asks,” stage directions, or any mention of speakers.
7. **Output format** – Return ONLY the rewritten sentence; no quotes, no markdown, no extra whitespace.

One‑shot example  
• Original: Mount Kilimanjaro stands 5,895 m tall in Tanzania.  
• Rewritten: Rising 5,895 metres above northern Tanzania, Mount Kilimanjaro towers like a silent cloud over the Serengeti plains.

(End of instructions. Start rewriting below.)
""".strip()

BASE_CONTEXT = [
    {"role": "system", "content": SYSTEM_CONTENT},
    {"role": "user",   "content": USER_RULES},
]

# ───────── GPT‑4o  ────────────────
def gpt_rewrite(original_sentence: str) -> str:
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

# ───────── CSV  ────────────────────
def rewrite_dataset():
    done = set()
    if RESUME and Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
            done = {row["Original"] for row in csv.DictReader(f)}

    with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
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
    print("▶  GPT‑4o calibration‑sentence rewriting …")
    rewrite_dataset()
    print("✅  Done.  Output saved to", OUTPUT_CSV)