"""
gptq_rewrite_pipeline.py

1. Feed a GPTQ-paper excerpt to GPT-4o ➜ get:
   • method summary
   • calibration-data requirements
   • numbered rewriting guidelines

2. Use those guidelines to rewrite every row in input.csv.
3. Save results to output_optimized.csv with columns:
   Original, Modified, Label  (Label is always 0)
"""

import csv, json, time
from pathlib import Path
import openai

# =============================== CONFIG =====================================
API_KEY           = "YOUR_OPENAI_API_KEY"          #  ←  put your key here
MODEL_NAME        = "gpt-4o"
PAPER_EXCERPT_TXT = "gptq_excerpt.txt"             # short ≤ ~15 k tokens
INPUT_CSV         = "input.csv"                    # must have column "origin"
OUTPUT_CSV        = "output_optimized.csv"
RATE_LIMIT_SEC    = 1.2                            # pause between API calls
RESUME            = True                           # skip rows already processed
# ============================================================================

openai.api_key = API_KEY


# ---------- helper --------------
def chat(messages, max_tokens=512, temperature=0.7):
    """Single OpenAI chat completion call."""
    rsp = openai.ChatCompletion.create(
        model       = MODEL_NAME,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    return rsp.choices[0].message.content.strip()


# ---------- Part 1 – obtain GPTQ guidelines -------------
def get_guidelines():
    cache = Path("guidelines_cache.json")
    if cache.exists():
        return json.loads(cache.read_text())

    excerpt = Path(PAPER_EXCERPT_TXT).read_text(encoding="utf-8")

    system_prompt = (
        "You are an expert LLM specialised in large-scale model quantisation.\n"
        "Read the GPTQ paper excerpt, then:\n"
        "Step 1 – summarise GPTQ’s core idea in 2-3 bullets.\n"
        "Step 2 – list ideal calibration-data characteristics.\n"
        "Step 3 – output a numbered GUIDELINE list for rewriting raw text.\n"
        "Return JSON with keys: "
        '{"method_summary": "...", "calib_requirements": "...", "guidelines": "..."}'
    )
    user_msg = f"[BEGIN_PAPER_EXCERPT]\n{excerpt}\n[END_PAPER_EXCERPT]"

    result  = chat(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_msg}],
        max_tokens=800
    )
    data = json.loads(result)   # will raise if malformed
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ---------- Part 2 – rewrite dataset ---------------
def rewrite_dataset(guidelines: str):
    system_msg = (
        "You are an assistant rewriting text for GPTQ calibration.\n"
        f"Follow these guidelines exactly:\n{guidelines}\n"
        "Output ONE rewritten sentence only – no markdown, no commentary."
    )

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

            user_prompt = f"Input: {origin}\nOutput:"
            try:
                modified = chat(
                    [{"role": "system", "content": system_msg},
                     {"role": "user",   "content": user_prompt}],
                    max_tokens=256
                )
            except Exception as e:
                print(f"[ERROR] {e} – writing blank output.")
                modified = ""

            writer.writerow({"Original": origin,
                             "Modified": modified,
                             "Label":    0})

            print(f"✓ {origin[:50]}  →  {modified[:50]}")
            time.sleep(RATE_LIMIT_SEC)


# ---------- MAIN -------------
if __name__ == "__main__":
    print("▶  Step 1 – querying GPT-4o for GPTQ guidelines…")
    info = get_guidelines()
    print("▶  Summary obtained.\n", info["method_summary"])
    print("▶  Rewriting dataset…")
    rewrite_dataset(info["guidelines"])
    print("✅  Completed.  Results saved to:", OUTPUT_CSV)
