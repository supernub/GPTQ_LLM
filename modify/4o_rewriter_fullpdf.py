#!/usr/bin/env python3
"""
gptq_rewrite_pipeline_fullpdf.py
================================
End-to-end pipeline that

1.  Extracts plain text from GPTQ.pdf  (PyPDF2).
2.  Sends *the whole* paper (optionally chunked) to GPT-4o to obtain:
      • method_summary
      • calib_requirements
      • numbered rewriting guidelines
3.  Uses those guidelines as a persistent system prompt to rewrite every
    row in input.csv.
4.  Saves results to output_optimized.csv with columns:
    Original, Modified, Label  (Label=0)
-----------------------------------------------------------------------
This version differs from the “excerpt” script only in Part 1.
"""

import csv, json, time, textwrap
from pathlib import Path
import openai
from PyPDF2 import PdfReader        # pip install PyPDF2

# ===================== USER CONFIG ==========================================
API_KEY          = "YOUR_OPENAI_API_KEY"
MODEL_NAME       = "gpt-4o"
PDF_PATH         = "GPTQ.pdf"                     # full paper
INPUT_CSV        = "input.csv"                    # must have column "origin"
OUTPUT_CSV       = "output_optimized.csv"
RATE_LIMIT_SEC   = 1.2
RESUME           = True                           # skip already-done rows
CHUNK_SIZE_TOK   = 6000   # ~ ≈ tokens per chunk (GPT-4o safe zone)
# ============================================================================

openai.api_key = API_KEY


# ---------- helper -----------------------------------------------------------
def chat(messages, max_tokens=700, temperature=0.7):
    rsp = openai.ChatCompletion.create(
        model       = MODEL_NAME,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    return rsp.choices[0].message.content.strip()

# ---------- Part 1  extract PDF ➜ guidelines --------------------------------
def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages  = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)

def chunk_text(raw: str, max_chars: int = 18000):
    """Rough chunking by characters so each fits under ~8k-token window."""
    raw = raw.replace("\x0c", " ")  # page-break symbols
    return textwrap.wrap(raw, max_chars)

def get_guidelines_from_pdf() -> dict:
    cache = Path("guidelines_fullpdf_cache.json")
    if cache.exists():
        return json.loads(cache.read_text())

    full_text = pdf_to_text(PDF_PATH)
    chunks    = chunk_text(full_text)   # usually 2–3 chunks

    # Accumulate paper summary across chunks (“map-reduce”)
    summaries = []
    for idx, chunk in enumerate(chunks):
        sys1 = "You are a careful researcher. Summarise the following paper chunk."
        user1 = f"[Paper chunk {idx+1}/{len(chunks)}]\n{chunk[:12000]}"
        summary = chat(
            [{"role": "system", "content": sys1},
             {"role": "user",   "content": user1}],
            max_tokens=700
        )
        summaries.append(summary)
        time.sleep(RATE_LIMIT_SEC)

    # Feed concatenated summaries to obtain final guidelines
    sys2 = (
        "You are an expert in model quantisation.\n"
        "Using the collected summaries of the GPTQ paper, produce JSON with:\n"
        " • method_summary  (2-3 bullets)\n"
        " • calib_requirements  (explain ideal calibration data)\n"
        " • guidelines  (numbered rules for rewriting raw text)\n"
        "Return **valid JSON only**."
    )
    user2 = "\n\n".join(summaries)

    result = chat(
        [{"role": "system", "content": sys2},
         {"role": "user",   "content": user2}],
        max_tokens=800
    )

    data = json.loads(result)   # will raise if JSON invalid
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ---------- Part 2  rewrite dataset (same as before) -------------------------
def rewrite_dataset(guidelines: str):
    system_msg = (
        "You are rewriting text for GPTQ calibration.\n"
        f"Follow these guidelines EXACTLY:\n{guidelines}\n"
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
                print(f"[ERROR] {e} – blank output recorded.")
                modified = ""

            writer.writerow({"Original": origin,
                             "Modified": modified,
                             "Label":    0})

            print(f"✓ {origin[:50]}  →  {modified[:50]}")
            time.sleep(RATE_LIMIT_SEC)


# ---------- MAIN ------------------------------------------------------------
if __name__ == "__main__":
    print("▶  Step 1 – extracting GPTQ.pdf & generating guidelines…")
    info = get_guidelines_from_pdf()
    print("▶  Paper analysed.  Method summary:\n", info["method_summary"])
    print("▶  Rewriting calibration dataset with extracted rules…")
    rewrite_dataset(info["guidelines"])
    print("  All done – see", OUTPUT_CSV)
