
"""
gptq_rewrite_pipeline_fullpdf.py  –  end-to-end pipeline

1.  Extract plain text from GPTQ.pdf  (PyPDF2).
2.  Feed the *entire* paper (chunked) to GPT-4o via the new OpenAI SDK:
      • method_summary
      • calib_requirements
      • numbered rewriting guidelines
3.  Use those guidelines to rewrite every row in input.csv.
4.  Save to output_4o_fullpdf.csv  →  columns:  Original, Modified, Label(=0)
"""

import csv, json, time, textwrap
from pathlib import Path
from PyPDF2 import PdfReader          # pip install PyPDF2
from openai import OpenAI             # pip install --upgrade openai

# ─────────── USER CONFIG ────────────────────────────────────────────────────
API_KEY          = "YOUR_OPENAI_API_KEY"   # or set env var OPENAI_API_KEY
MODEL_NAME       = "gpt-4o"
PDF_PATH         = "GPTQ.pdf"
INPUT_CSV        = "input.csv"             # must contain column "origin"
OUTPUT_CSV       = "output_4o_fullpdf.csv"
RATE_LIMIT_SEC   = 1.2                     # pause between requests
RESUME           = True                    # skip rows already processed
MAX_CHARS        = 18000                   # per-chunk char cap  (~6–7k tokens)
# ────────────────────────────────────────────────────────────────────────────

# initialise client (new SDK style)
client = OpenAI(api_key=API_KEY)


# ─────────── helper: single chat call ───────────────────────────────────────
def chat(messages, max_tokens=700, temperature=0.7):
    resp = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    return resp.choices[0].message.content.strip()


# ─────────── PART 1:  PDF → guidelines JSON ─────────────────────────────────
def pdf_to_text(pdf_path: str) -> str:
    pages = [page.extract_text() for page in PdfReader(pdf_path).pages]
    return "\n".join(pages).replace("\x0c", " ")

def chunk_text(raw: str, max_chars: int):
    return textwrap.wrap(raw, max_chars)

def get_guidelines() -> dict:
    cache = Path("guidelines_fullpdf_cache.json")
    if cache.exists():
        return json.loads(cache.read_text())

    full_text = pdf_to_text(PDF_PATH)
    chunks    = chunk_text(full_text, MAX_CHARS)

    # ─ map step ─  summarise each chunk
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        sys = "You are a careful researcher. Summarise this paper chunk."
        user = f"[GPTQ paper chunk {idx}/{len(chunks)}]\n{chunk}"
        summaries.append(
            chat([{"role": "system", "content": sys},
                  {"role": "user",   "content": user}],
                 max_tokens=700)
        )
        time.sleep(RATE_LIMIT_SEC)

    # ─ reduce step ─  derive final guidelines from all summaries
    sys2 = (
        "You are an expert in model quantisation.\n"
        "Using the collected summaries of GPTQ, output ONLY valid JSON with keys:\n"
        '  "method_summary": "...",\n'
        '  "calib_requirements": "...",\n'
        '  "guidelines": "1) ... 2) ..."\n'
        "No extra keys, no markdown."
    )
    user2 = "\n\n".join(summaries)
    data  = json.loads(chat(
        [{"role": "system", "content": sys2},
         {"role": "user",   "content": user2}],
        max_tokens=800
    ))

    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ─────────── PART 2:  rewrite dataset row-by-row ────────────────────────────
def rewrite_dataset(guidelines: str):
    system_msg = (
        "You are rewriting text for GPTQ calibration.\n"
        f"Follow these rules EXACTLY:\n{guidelines}\n"
        "Return ONE fluent English sentence only – no markdown, no commentary."
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
            except Exception as exc:
                print(f"[ERROR] {exc} – blank output recorded.")
                modified = ""

            writer.writerow({"Original": origin,
                             "Modified": modified,
                             "Label":    0})
            print(f"✓  {origin[:45]}  →  {modified[:45]}")
            time.sleep(RATE_LIMIT_SEC)


# ─────────── MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶  Extracting GPTQ.pdf & building guidelines …")
    info = get_guidelines()
    print("▶  Paper analysed – summary:\n", info["method_summary"])
    print("▶  Rewriting calibration dataset …")
    rewrite_dataset(info["guidelines"])
    print("✅  Finished.  Output ➜", OUTPUT_CSV)
