from openai import OpenAI
import csv
import time

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

REWRITE_PROMPT = """
You are an expert in natural language refinement. Rewrite the following text to meet the criteria used in calibration datasets for quantized language models:

Instructions:
- Preserve all key information (names, locations, products)
- Ensure fluency, grammaticality, and natural sentence structure
- Avoid repetition, unnatural phrasing, or template-based expressions
- Rewrite should be stylistically rich, not literal
- Do NOT repeat or include the original sentence
- Output ONLY the rewritten version, no headers or comments


Text to rewrite:
\"\"\"{text}\"\"\""""

# 
def rewrite_text(text):
    prompt = REWRITE_PROMPT.format(text=text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}"


input_path = "../data/clean_10_test.txt"
output_path = "output_4o.csv"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline='', encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Original", "Modified", "Label"])

    for i, line in enumerate(infile):
        original = line.strip()
        if not original:
            continue
        print(f"[{i+1}] Rewriting: {original[:50]}...")
        modified = rewrite_text(original)
        writer.writerow([original, modified, "0"])
        time.sleep(1.2) 

