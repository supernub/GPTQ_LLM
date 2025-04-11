from openai import OpenAI
import csv
import time

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# prompt
REWRITE_PROMPT = """
Please rewrite the following text according to the rewriting principles used in calibration datasets for quantized LLMs:

Rewriting goals:
- Preserve all important information (e.g., names, locations, products)
- Make the text fluent, grammatically correct, and natural sounding
- Avoid repetitive or template-like structures
- Promote diversity in phrasing and sentence construction
- Do not remove or skip any critical details from the original

Text to rewrite:
"{text}"
"""

# 文本重写函数
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
        print(f"Error: {e}")
        return ""


input_path = "clean_10_test.txt"
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

print(f"\n Finished! Output saved to: {output_csv}")
