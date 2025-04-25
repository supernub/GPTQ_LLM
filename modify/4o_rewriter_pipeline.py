import csv
import time
import openai

# ========== Configuration ==========
API_KEY = "your-openai-api-key"
MODEL_NAME = "gpt-4o"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output_optimized.csv"
RATE_LIMIT = 1.2                  # Seconds between requests

# ========== Initialize Client ==========
openai_client = openai.OpenAI(api_key=API_KEY)

# ========== Few-shot Prompt ==========
FEWSHOT = """
You are an expert in language model compression. Please rewrite each sentence according to the following rules:
1. Preserve the original meaning.
2. Improve fluency and readability (avoid template-like or fragmented phrases).
3. Increase linguistic diversity (use varied sentence structures and natural connectors).
4. Do not remove important details such as location names, object names, or dates.
5. Output one clean, rewritten sentence only.

Input: This is a 47.2km cycling trail in United States MI Wayne Co. Detroit. The trail was created 7/4/2017. Max elevation gain: 1.7K m.
Output: This is a 47.2 km cycling trail located in Detroit, Wayne County, Michigan, United States. The trail was created on July 4, 2017, with a maximum elevation gain of approximately 1.7 km.
"""

# ========== Generate Response ==========
def rewrite_text(input_text):
    prompt = FEWSHOT.strip() + f"\n\nInput: {input_text}\nOutput:"
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error rewriting: {e}")
        return None

# ========== Main Loop ==========
with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["Original", "Modified", "Label"])
    writer.writeheader()

    for row in reader:
        origin = row["Original"]
        print(f"Rewriting: {origin}")

        optimized = rewrite_text(origin)
        if optimized:
            writer.writerow({"Original": Original, "Modified": Modified, "Label": 0})
        else:
            writer.writerow({"Original": Original, "Modified": "", "Label": 0})

        time.sleep(RATE_LIMIT)

print("✅ Done rewriting calibration dataset.")
